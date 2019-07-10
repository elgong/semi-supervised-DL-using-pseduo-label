#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 19-5-20 上午11:18
#@Author: elgong_hdu
#@File  : dayaLoder.py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random
from torchvision import transforms, utils


# # 数据预处理和加载
# def default_loader(path):
#     im = Image.open(path).convert('RGB')
#     #im = np.asarray(im.resize((32, 32)))
#     im = im.resize((32, 32))
#     return im
#
#
# class MyDataset(Dataset):
#     def __init__(self, txt, transform=True, target_transform=None, loader=default_loader):
#         f = open(txt, 'r')
#         self.folder = txt.split('/')[-1].split('.')[0]
#         #print(txt.split('/')[-1])
#         imgs = []
#         for line in f.readlines():
#             img_name = line.split(",")[0]
#             label = line.split(",")[1]
#             print(img_name)
#             imgs.append((img_name, int(label)))
#
#         self.imgs = imgs
#         random.shuffle(self.imgs)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     def __getitem__(self, index):  # 类的特殊方法
#
#         img_name, label = self.imgs[index]
#         img_path = '/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/' + self.folder + '/' + img_name
#         #print(img_path)
#         img = self.loader(img_path)
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)






# 数据预处理和加载
def default_loader(path):
    im = Image.open(path).convert('RGB')
    im = im.resize((32, 32))
    return im

class MyDataset(Dataset):
    def __init__(self, txt, data_path = "/home/elgong/GEL/one_shot/paper/cifar10-classification/data/",
                 transform=True, target_transform=None, loader=default_loader, iter = 0):
        f = open(txt, 'r')
        self.folder = txt.split('/')[-1].split('.')[0]
        imgs = []
        for line in f.readlines():
            img_name = line.split(",")[0]
            label = line.split(",")[1]
            imgs.append((img_name, int(label)))

        self.imgs = imgs
        random.shuffle(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.orgin = {}
        self.data_path = data_path
        self.iter = iter  # 判断是否需要RLB
        with open("./train.txt", "r") as f:
            for line in f:
                img_name = line.strip().split(",")[0]
                cls = img_name.split("_")[0]
                if cls not in self.orgin:
                    self.orgin[cls] = [img_name]
                else:
                    self.orgin[cls].append(img_name)

    def __getitem__(self, index):  # 类的特殊方法

        img_name, label = self.imgs[index]
        img_path = self.data_path + self.folder + '/' + img_name
        #print(img_path)
        img = self.loader(img_path)
        randint = random.randint(0, 19)

        if self.folder == "train" and self.iter == 1:
            if 3 == len(img_name.split("_")):

                label1 = img_name.split("_")[0]

                img_name1 = self.orgin[label1][randint]

                img_path1 = self.data_path + self.folder + '/' + img_name1

                pro = np.random.random()
                if pro < 0.4:
                    img1 = self.loader(img_path1)
                    img1 = np.array(img1)
                    img = np.array(img)
                    img2 = np.floor(0.7 * img + 0.3 * img1).astype("uint8")
                    img = Image.fromarray(img2)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

