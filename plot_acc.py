
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-darkgrid')
# plt.figure(figsize=(7,7))

########################################################################

font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size'   : 22}
font2 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size'   : 22}


file_name1 = 'lend_cifar10_ccn_0.4_K8_256_beta0.9_1'
file_name2 = 'lend_cifar10_ccn_0.4_K8_256_beta0.8_1'
# file_name3 = 'lend_cifar100_rcn_0.6_K3_64_beta0.9_'
path1 = 'results1011/' + str(file_name1) + '.txt'
path2 = 'results1011/' + str(file_name2) + '.txt' 
# path3 = 'results1/' + str(file_name3) + '.txt'
acc1 = [] 
acc2 = []
acc3 = []

myfile1 = open(path1)
for line in myfile1.readlines():
    if 'valid' in line:
        temp = line.split()
        # print(temp)
        acc1.append(float(temp[1]))
    if len(acc1)>199:
        break
    
myfile2 = open(path2)
for line in myfile2.readlines():
    if 'valid' in line:
        temp = line.split()
        acc2.append(float(temp[1]))
    if len(acc2)>199:
        break

# myfile3 = open(path3)
# for line in myfile3.readlines():
#     if 'valid' in line:
#         temp = line.split()
#         # print(temp)
#         acc3.append(float(temp[1]))
#     if len(acc3)>199:
#         break

acc1 = np.array(acc1)
acc2= np.array(acc2)
# acc3= np.array(acc3)
x = np.linspace(1,acc1.shape[0],acc1.shape[0])
x = x-1
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(7,7))
f1, = plt.plot(x, acc1, c='red', linewidth=2, label = 'none')
f2, = plt.plot(x, acc2, c='blue', linewidth=2, label = 'sharp')
# f3, = plt.plot(x, acc3, c='black', linewidth=2, label = 'K=9')
plt.legend(handles=[f1, f2], prop=font1, loc='lower right')
# plt.savefig("figs/rcn/K8vs9.png", dpi=600, bbox_inches='tight')




# mpacc = np.array(mpacc)
# knnacc = np.array(knnacc)
# mpmean = np.mean(mpacc, axis = 0)
# knnmean = np.mean(knnacc, axis = 0)
# mpmax = np.max(mpacc, axis = 0)
# mpmin = np.min(mpacc, axis = 0)
# knnmax = np.max(knnacc, axis = 0)
# knnmin = np.min(knnacc, axis = 0)

# print(knnmean)

# x = np.linspace(1,mpacc.shape[1],mpacc.shape[1])
# # x = x-1
# plt.style.use('seaborn-darkgrid')
# plt.figure(figsize=(7,4.5))
# f1, = plt.plot(x, mpmean, c='red', label='Model predicted labels', linewidth=2)
# f2, = plt.plot(x, knnmean, c='blue', label=r'Diluted labels', linewidth=2)
# plt.fill_between(x, mpmin, mpmax, color='red', alpha=0.1)
# plt.fill_between(x, knnmin, knnmax, color='blue', alpha=0.1)
# plt.xlabel(r'Epoch', font2)
# plt.ylabel('Accuracy (%)', font2)
# plt.xlim(0, 100)
# plt.ylim(25.5, 95)
# legend = plt.legend(handles=[f1, f2], prop=font1, loc='lower right')#, loc='lower left'
# plt.savefig("acc.png", dpi=600, bbox_inches='tight')
# print


########################################################################

# color = ['r', 'b', 'y', 'c', 'g', 'k']
# for i in [4,5,6,7,8,9]:

#     lepath = 'Z:/lend/cifar/result_beta/LEND05_cifar10_ccn_0.4_K5_beta0.' + str(i) + '_.txt'
#     lefile = open(lepath)
#     leacc = []
#     for line in lefile.readlines():
#         if 'valid' in line:
#             temp = line.split()
#             leacc.append(float(temp[1]))
#         if len(leacc)>199:
#             break
#     leacc = np.array(leacc)
#     x = np.linspace(1, leacc.shape[0], leacc.shape[0])
#     plt.plot(x, leacc, color=color[i-4])
#     plt.legend(['0.'+str(i)])




# lepath = 'Z:/lend/cifar/result_beta/LEND05_cifar10_rcn_0.4_K5.txt'
# lefile = open(lepath)
# leacc1 = []
# for line in lefile.readlines():
#     if 'valid' in line:
#         temp = line.split()
#         leacc1.append(float(temp[1]))
#     if len(leacc1)>199:
#         break
# leacc1 = np.array(leacc1)
# x = np.linspace(1, leacc1.shape[0], leacc1.shape[0])




#########################################################################
# copath = 'Z:\\lend\\Co-teaching-master\\results\\cifar100\coteaching/coteachingcifar100_coteaching_pairflip_0.4.txt'
# copath = 'Z:\\lend/JoCoR-master\\results\\jocor_cifar10_pairflip_0.2.txt'
# cofile = open(copath)
# coacc = []

# for line in cofile.readlines():
#     if 'valid' in line:
#         temp = line.split()
#         coacc.append(float(temp[2]))
#     if len(coacc)>199:
#         break

# coacc = np.array(coacc)
# x = np.linspace(1, coacc.shape[0], coacc.shape[0])
# print(max(coacc))


#########################################################################
# cepath = 'Z:\\lend\\Co-teaching-master\\results\\cifar10\coteaching/coteachingcifar10_coteaching_pairflip_0.3.txt'
# cefile = open(copath)
# ceacc = []

# for line in cefile.readlines():
#     if 'valid' in line:
#         temp = line.split()
#         ceacc.append(float(temp[1]))
#     if len(ceacc)>199:
#         break

# ceacc = np.array(coacc)
# x = np.linspace(1, ceacc.shape[0], ceacc.shape[0])
# plt.style.use('seaborn-darkgrid')
# plt.figure(figsize=(7,7))


# leplot1 = plt.plot(x, leacc1, color='r')
# leplot = plt.plot(x, leacc)
# coplot = plt.plot(x, coacc)
# leplot = plt.plot(x, ceacc)




# plt.savefig("compare_cm.png", dpi=600, bbox_inches='tight') # 保存图片
plt.show() # 显示图片
