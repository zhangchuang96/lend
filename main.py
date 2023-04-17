from __future__ import print_function
import sys
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import time
import datetime
import argparse
import numpy as np
from cifar10.cifar import *
from torch.utils.data import DataLoader
from models.preact_resnet import PreActResNet18
from models.resnet import resnet18
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--momentum', default=True, type=bool, help='momentum in KNN labels')
parser.add_argument('--noise_rate', default=0.5, type=float, help='noise rate')
parser.add_argument('--noise_type', default='rcn', type=str, help='noise type (ccn, rcn)')
parser.add_argument('--K', default=3, type=int, help='k nearest neighbors')
parser.add_argument('--alpha', default=1, type=float, help='label sharpening')
parser.add_argument('--beta', default=0.9, type=float, help='exponential moving average')
parser.add_argument('--batch_size', default=256, type=int, help='training batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=200, type=int, help='training epochs')
parser.add_argument('--id', default='', help='training mark')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int, help='number of classes')
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num', default='1', type=str, help='training round')
args = parser.parse_args()
print(args)


def dist_matrix(emb):
    sqrt_sum = (emb ** 2).sum(1, keepdim = True)
    return sqrt_sum + sqrt_sum.t() - 2 * emb @ emb.t()

def KNN_labels(embs, targets, epoch, K):
    global propgated_labels
    _, nearest_neighbors = dist_matrix(embs.detach()).topk(k = K, largest = False)
    nearest_targets = targets[nearest_neighbors] 
    prop_in_batch = t.zeros_like(preds)
    for iii in range(K): # prop_in_batch: one-hot label of the batch data
        prop_in_batch += eye[nearest_targets[:,iii]]
    prop_in_batch /= K
    return prop_in_batch

def prop_label(embs, cur, K, alpha = 1):
    _, nearest_neighbors = dist_matrix(embs.detach()).topk(k = K, largest = False)
    A = ((embs @ embs.t()) / embs.size(0)).clamp(min = 0) ** 3
    A = A * (1 - t.eye(A.size(0)).cuda())
    MASK = t.ones_like(A)
    for _ in range(nearest_neighbors.size(0)): MASK[_][nearest_neighbors[_]] = 0
    MASK = 1 - MASK
    A *= MASK
    W = A @ A.t()
    D = (1 / W.sum(1, keepdim = True).sqrt()) * t.eye(W.size(0)).cuda()
    W = D @ W @ D
    Z = (t.eye(A.size(0)).cuda() - 0.99 * W).inverse() @ cur
    if (Z!=Z).long().sum() != 0: Z = cur
    return (Z * alpha).softmax(1)

def cross_entropy(preds, targets, eye):
    return -(preds.log_softmax(1) * eye[targets]).sum(-1)

def manifold_mixup_criterion(emb, y, model, alpha = 4):
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = t.randperm(emb.size(0)).cuda()
    mixed_emb = lam * emb + (1 - lam) * emb[index, :]
    pred = model.linear(mixed_emb)
    y_a, y_b = y, y[index]
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# save path -----------#
theme = ''
save_dir = os.getcwd() + '/CPU_time/'
now_time=datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
file_name = 'lend' + '_' + args.dataset + '_' + args.noise_type + '_' + \
    str(args.noise_rate) + '_K' + str(args.K) + '_' + str(args.batch_size) + '_beta' + str(args.beta) + '_alpha' + str(args.alpha) + '_' + args.num +'.txt'
txt_file = save_dir + file_name
with open(txt_file, "w") as myfile:
    myfile.write(now_time + '\n')
    myfile.write('test batchsize' + '\n')
    myfile.write(str(args) + '\n')

# load data ------------#
if args.dataset == 'cifar100':
    trainset, validset = CIFAR100('./data/cifar100', train = True), CIFAR100('./data/cifar100', train = False)
    num_classes = 100
elif args.dataset == 'cifar10':
    trainset, validset = CIFAR10('./data/cifar10', train = True), CIFAR10('./data/cifar10', train = False)
    num_classes = 10

# generate noise -------#    
trainset.noisy_targets, noise_name = syn_noise(dataset = args.dataset, target = trainset.targets, noise_rate = args.noise_rate, type = args.noise_type)
trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
validloader = DataLoader(validset, batch_size = args.batch_size, shuffle = False)
num_classes = len(set(trainset.targets))
eye = t.eye(num_classes).cuda()

# model details --------#
model = PreActResNet18(num_classes = num_classes).cuda()
optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay=5e-4, momentum = 0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100], 0.1)

# define global propgated labels
propgated_labels = t.zeros((len(trainset),num_classes)).cuda() 
print('Training...\n')
for epoch in range(args.num_epochs):
    # lr scheduler
    if epoch != 0: 
        scheduler.step()
    model.train()
    # train
    loss_total, correct_total, prop_correct_total, total = 0, 0, 0, 0
    weights_t, weights_n, true_total = 0, 0, 0
    for idx, (inputs, ground_truth, targets, indies) in enumerate(trainloader):
        time1 = time.time()
        inputs, ground_truth, targets, indies = inputs.cuda(), ground_truth.cuda(), targets.cuda(), indies.cuda()
        optimizer.zero_grad()
        preds, embs = model(inputs)
        time2 = time.time()
        if epoch < 10:
            batch_prop_labels = prop_label(embs, eye[targets], args.K, args.alpha).detach()
        else:
            batch_prop_labels = prop_label(embs, args.beta * eye[targets] + (1 - args.beta) * propgated_labels[indies], args.K, args.alpha).detach()
        time3 = time.time()

        propgated_labels[indies] = batch_prop_labels
        batch_prop_labels = Variable(batch_prop_labels).cuda()

        weights = (batch_prop_labels * eye[targets]).sum(-1).detach()  #** 2
        # weights = weights**(1/0.5) # temparature sharpening 
        # weights = t.pow(weights,2)
        # weights /= t.sum(weights, axis=0)

        loss = weights * cross_entropy(preds, targets, eye)
        loss.mean().backward()
        optimizer.step()
        
        loss_total += loss.sum().item()
        correct_total += (preds.argmax(1) == ground_truth).sum().item()
        prop_correct_total += (batch_prop_labels.argmax(1) == ground_truth).sum().item()
        total += inputs.size(0)
        weights_t += weights[ground_truth == targets].sum().item()
        weights_n += weights[ground_truth != targets].sum().item()
        true_total += (ground_truth == targets).sum().item()
        time4= time.time()

        sys.stdout.write('\r [train] epoch %d (%d / %d) | loss %.2f | accu %.2f%% | prop accu %.2f%%' 
            % (epoch, idx + 1, len(trainloader), loss_total / total, correct_total * 100 / total, prop_correct_total * 100 / total))

    with open(txt_file, "a") as myfile:
        myfile.write('[train]' + str(int(epoch)) + ': '  + str(loss_total / total) + ' ' + str(correct_total * 100 / total) + ' ' + str(prop_correct_total * 100 / total) + "\n")
    print()


    with t.no_grad():
        model.eval()
        correct_, total_ = 0, 0
        for idx, (inputs, targets, _, indies) in enumerate(validloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            preds = model(inputs)[0].argmax(1)
            correct_ += (preds == targets).sum().item()
            total_ += inputs.size(0)
            sys.stdout.write('\r [valid] accu %.2f%%' % (correct_ * 100 / total_))
        with open(txt_file, "a") as myfile:
            myfile.write(str('[valid]') + 'accu: ' + str(correct_ * 100 / total_) + "\n\n")  
    print('\n')
