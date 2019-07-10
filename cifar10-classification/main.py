'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

## 导入包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.utils import progress_bar
from utils.dataLoder import MyDataset
import os
from models import *
#from utils.tta import ClassPredictor

# 数据增广
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_data = MyDataset(txt='./train.txt',transform=transform_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) #返回的是迭代器
test_data = MyDataset(txt='./val.txt', transform=transform_test)
test_loader = DataLoader(test_data, batch_size=64)



device = 'cuda' #if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# 模型仓库
from models import senet
from models import resnet
from models import dpn
from models import googlenet
print('==> Building model..')

# net = VGG('VGG19')
#net = resnet.ResNet18()
# net = PreActResNet18()
#net = googlenet.GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
#net = dpn.DPN92()
# net = ShuffleNetG2()
#net = senet.SENet18()
from models import shufflenetv2
#from utils2 import *
#net = shufflenetv2.ShuffleNetV2(1)
"""
tta_aug = tta_aug = [
    NoneAug(),
    Hflip(),
    Vflip(),
    Rotate(20),
    Adjustbright(0.2),
    Adjustsaturation(0.3),
    Adjustcontrast(0.2)
]
"""
LR = 0.001

# import cv2
def adjust_learning_rate(optimizer, epoch):
    """ 学习率更新规则 """
    lr = LR * (0.1 ** (epoch // 150))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
## 训练模块
def train(epoch,  net):
    print('\nEpoch: %d' % epoch)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True



    ## 优化方法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    net.train()  # 指明训练网络，dropout
    adjust_learning_rate(optimizer, epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print(inputs,targets)
        #adjust_learning_rate(optimizer, epoch)

        # cv2.imshow("img", inputs.cpu().numpy()[0][0])
        # cv2.waitKey(0)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return [1.*correct/total, train_loss/(batch_idx+1)]

## 测试模块
def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    #clsPre = ClassPredictor(net,device,tta_aug)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

           # outputs = clsPre.predict(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(loss, correct)
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 1.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc
        torch.save(net, "./model2.pkl")
    return [1.*correct/total, test_loss/(batch_idx+1)]


from models import pnasnet
net_list  = []
net_list.append(resnet.ResNet18())
net_list.append(senet.SENet18())
net_list.append(googlenet.GoogLeNet())


train_Acc=[[],[],[]]
train_loss=[[],[],[]]
val_Acc=[[],[],[]]
val_Loss=[[],[],[]]

## 开始训练和测试
for i, net in enumerate(net_list):
    for epoch in range(start_epoch, start_epoch+200):

        train_data = train(epoch, net)
        test_data = test(epoch, net)

        train_Acc[i].append(train_data[0])
        train_loss[i].append(train_data[1])
        val_Acc[i].append(test_data[0] )
        val_Loss[i].append(test_data[1])





from pylab import *
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import random

# ##### 以下设置画图后，显示中文 分辨率也比默认高很多
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300 #图片像素
#plt.rcParams['figure.dpi'] = 1200 #分辨率

pdf = PdfPages('acc-3.pdf')
x1 = range(0, 200)
plt.figure(figsize=(4,3), dpi= 400)
# 开始画图
# plt.title('Result Analysis')
plt.plot(x1, train_Acc[0], color='green',linewidth=0.5,  label='resnet18')
plt.plot(x1, train_Acc[1], color='red', linewidth=0.5,label='senet18')
plt.plot(x1, train_Acc[2], color='skyblue', linewidth=0.5,label='googlenet')

plt.plot(x1, val_Acc[0], color='green',linewidth=1.0)
plt.plot(x1, val_Acc[1], color='red', linewidth=1.0)
plt.plot(x1, val_Acc[2], color='skyblue',linewidth=1.0)
plt.yticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],size = 12)
plt.xticks( size = 22)
plt.legend()  # 显示图例

plt.xlabel('epochs',fontsize=12)
plt.ylabel('acc',fontsize=2)
plt.tight_layout()
#plt.show()
print( 'savefig...')
pdf.savefig()
plt.close()
pdf.close()
# python 一个折线图绘制多个曲线


