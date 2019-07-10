
from __future__ import print_function

####################################################### 导入包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.utils import progress_bar
from utils.dataLoder import MyDataset, MyTest
import os
import pandas as pd
import time
import numpy as np
import cv2
from models import senet
from models import pnasnet

task_flag = "train"
from utils.utils import makeLabel,makeLabelTrain3,makeLabelVal3

### 待分类别
Class = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")



###################################################### 模型训练acc 记录表格 #######################################################################

df = pd.DataFrame(columns=Class, index=Class)
#
df_name = "./result/result_train_acc___" + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()
                                                                                                       )) + ".csv"
df.to_csv(df_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



######################################################  模型仓库  ######################################################
LR = 0.001
best_acc = 0   # best test accuracy
def adjust_learning_rate(optimizer, epoch):
    """ 学习率更新规则 """
    lr = LR * (0.1 ** (epoch //200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

## 训练模块
def train(net, net_name, epoch, train_loader, group):
    print('\nEpoch: %d   ' % epoch + group[0] + " : " + group[1] + "------" + net_name)
    net.train()  # 指明训练网络，dropout
    train_loss = 0
    correct = 0
    total = 0
    ##  损失函数
    criterion = nn.CrossEntropyLoss()

    ## 优化方法
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                         weight_decay=5e-4)      # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adjust_learning_rate(optimizer, epoch)

        # 可视化
        # print(inputs.cpu().numpy().shape)
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

    return [1. * correct / total, train_loss / (batch_idx + 1)]

## 测试模块
def test(net, net_name, epoch, test_loader, group):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ## 优化方法
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(test_loader),'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    ## 保存模型
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc
        torch.save(net, "./model_pkl/" + group[0] + "_" + group[1] +"_"  + net_name + "_model.pkl")
    return [1. * correct / total, test_loss / (batch_idx + 1)]



######################################################  CVS-联合训练 ######################################################
transform_train = None
transform_test = None

def union_train(Class):

    # 记录所有的组别
    union_group = []
    for i in range(len(Class)):
        for j in range(len(Class)):
            if i == j:
                continue
            union_group.append((Class[i], Class[j]))
    print(union_group)
    # 为每个组都训练一个模型
    for group in union_group:
        #print(group)
        # if group != ("apple", "beetle"):
        #     continue
        train_Acc = [[]]
        train_loss = [[]]
        val_Acc = [[]]
        val_Loss = [[]]


        cls1 = group[0]
        cls2 = group[1]

        index_cls1 = Class.index(cls1)
        index_cls2 = Class.index(cls2)
        net = None
        ### 两种网络方案
        print('==> Building model..')
        net_name = None
        if index_cls1 < index_cls2:
            net = pnasnet.PNASNetB()
            net_name = "pnasnet"
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif index_cls1 > index_cls2:
            net = senet.SENet18()
            net_name = "SENet18"
            transform_train = transforms.Compose([

                # transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.3),
                # transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([

               # transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(p=0.3),
               # transforms.RandomVerticalFlip(p=0.3),
               # transforms.RandomRotation(20),
               # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise Exception("net error")

        net = net.to(device)
        if device == 'cuda':
            # global net
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        ## make label file
        src_val_dic = "./data/val"

        txt_val = "./data/label_txt/" + cls1 + "_" + cls2 + "_val.txt"

        src_train_dic = "./data/train"

        txt_train = "./data/label_txt/" + cls1 + "_" + cls2 + "_train.txt"

        makeLabelVal3(src_val_dic, txt_val, cls1, cls2)
        makeLabelTrain3(src_train_dic, txt_train, cls1, cls2)

        print('==> Preparing data..')
        train_data = MyDataset(txt=txt_train, transform=transform_train)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # 返回的是迭代器
        test_data = MyDataset(txt=txt_val, transform=transform_test)
        test_loader = DataLoader(test_data, batch_size=32)
        global best_acc   # best test accuracy
        best_acc = 0
        model_list = os.listdir("./model_pkl")

        # 断点继续
        if group[0] + "_" + group[1] +"_"  + net_name + "_model.pkl" in model_list:
            print("----"*5)
            continue

        ## 开始训练和测试
        global LR
        LR = 0.001
        for epoch in range(0, 300):
            train_data = train(net, net_name,  epoch, train_loader, group)
            test_data = test(net, net_name, epoch, test_loader, group)

        df = pd.read_csv(df_name, header=0, index_col=0)
        df.loc[cls1, cls2] = "0." + str(int(best_acc))
        df.to_csv(df_name)

        # 删除网络
        del net



#################################### 开始训练 ##############################################
union_train(Class)

#################################### 开始第 i 轮测试 #########################################

unlabel_img = []
with open("./data/first_unlabel_img_name.txt",'r') as f:
    for img in f:
        img = img.strip().split(",")[0]
        if "png" in img:
            unlabel_img.append(img)

# 记录所有的组别
union_group = []
for i in range(len(Class)):
    for j in range(len(Class)):
        if i == j:
            continue
        union_group.append((Class[i] + "_" + Class[j]))

# 为每个组都训练一个模型
df_2_path ="./result/result_predict_unlabel.csv" #"/home/elgong/GEL/one_shot/torch3/result_predict_unlabel.csv"
df_2 = pd.DataFrame(columns=union_group, index=unlabel_img)
df_2.to_csv(df_2_path)

model_list = os.listdir("./model_pkl")

# 加载单样本的方法
transform_test = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

unlabel_data = MyTest(txt="./data/first_unlabel_img_name.txt", transform=transform_test)
unlabel_loader = DataLoader(unlabel_data, batch_size=500)




df_2 = pd.read_csv(df_2_path, header=0, index_col=0)

from tqdm import tqdm

# 加载每个网络测试
n = 0
for model in model_list:

    the_model = torch.load("./model_pkl/" + model)
    the_model.eval()
    n+=1
    # 预测每张图片
    cls1 = model.split("_")[0]
    cls2 = model.split("_")[1]
    with torch.no_grad():
        for batch in tqdm(unlabel_loader):
            test_data, test_label, test_name = batch

            test_data = test_data.to(device)

            res = the_model(test_data)


            #score = F.sigmoid(res).cpu().detach().numpy()
            score = F.softmax(res,1).cpu().detach().numpy()
            score_index = np.argmax(score, 1)
            for i in range(len(score_index)):
                res = str(score[i][0]) + "/" + str(score[i][1])
                df_2.loc[test_name[i], cls1 + "_" + cls2] = res

    df_2.to_csv(df_2_path)
    df_2 = pd.read_csv(df_2_path, header=0, index_col=0)


