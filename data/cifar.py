from PIL import Image
import os
import os.path
import numpy as np
import copy
import pickle
from typing import Any, Callable, Optional, Tuple
from torchvision.transforms import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

import torch as t

def syn_noise(dataset, target, noise_rate, type):
    if 'rc' in type: #rcn
        return syn_rcn_noise(dataset, target, noise_rate)
    elif 'cc' in type: #ccn
        return syn_ccn_noise(dataset, target, noise_rate)
    elif 'id' in type: #idn
        return syn_idn_noise(dataset, target, noise_rate)
    else:
        RuntimeError('invalid value of key type')

def syn_rcn_noise(dataset, target, noise_rate): # generate label noise
    noise_file = dataset+'/rcn_'+'noise_'+str(noise_rate)+'.pth'
    if os.path.exists(noise_file):
        print('Load noise config...\n')
        noisy_target = t.load(dataset+'/rcn_noise_%.1f.pth' % noise_rate)
    else:
        print('Construct noise config...\n')            
        target = t.tensor(target).float()
        num_classes = target.unique().size(0)
        p = noise_rate * num_classes / (num_classes - 1) # why?
        noisy_target = t.where(t.rand_like(target) > p, target, t.rand_like(target) * num_classes).long() # randomly flip
        t.save(noisy_target, dataset + '/rcn_noise_%.1f.pth' % noise_rate)
    print('[RCN] noise_rate %.1f%%'% ((t.tensor(target).long() != noisy_target).float().mean() * 100)) # print overall noise rate
    return noisy_target.tolist(), 'rcn'

def syn_ccn_noise(dataset, target, noise_rate):
    noise_file = dataset+'/ccn_'+'noise_'+str(noise_rate)+'.pth'
    if os.path.exists(noise_file):
        print('Load noise config...\n')
        noisy_target = t.load(dataset+'/ccn_noise_%.1f.pth' % noise_rate)
    else:
        print('Construct noise config...\n')
        target = t.tensor(target).float()
        num_classes = target.unique().size(0)
        next_target = target + 1
        next_target = t.where(next_target == num_classes, t.zeros_like(next_target), next_target)
        noisy_target = t.where(t.rand_like(target) > noise_rate, target, next_target).long()
        t.save(noisy_target, dataset+'/ccn_noise_%.1f.pth' % noise_rate)
    print('[CCN] noise_rate %.1f%%'% ((t.tensor(target).long() != noisy_target).float().mean() * 100))
    return noisy_target.tolist(), 'ccn'

def syn_idn_noise(target, noise_rate):
    target = t.tensor(target)
    complete_corrupt_targets, py = t.load('data/cifar10_idn.pth')
    complete_corrupt_targets, py = complete_corrupt_targets.cpu(), py.cpu()
    corrupted_idx = np.random.choice(py.size(0), size = int(target.size(0) * noise_rate), p = py.numpy(), replace = False)
    corrupted_idx = t.tensor(corrupted_idx)
    corrupt_targets = copy.deepcopy(target)
    corrupt_targets[corrupted_idx] = complete_corrupt_targets[corrupted_idx]
    print('[IDN] noise_rate %.2f%%'% ((target.long() != corrupt_targets).float().mean() * 100))
    return corrupt_targets.tolist(), 'idn' 

class CIFAR10(VisionDataset):
   
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    
    def __init__(self, root: str, train: bool = True) -> None:

        super(CIFAR10, self).__init__(root)
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]) if train else transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #self.transform = transforms.ToTensor()

        self.train = train  # training set or test set
        self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        self.noisy_targets = []


    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
       
        img, target = self.data[index], self.targets[index]
        try: noisy_target = self.noisy_targets[index]
        except: noisy_target = -1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, target, noisy_target, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            #print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
