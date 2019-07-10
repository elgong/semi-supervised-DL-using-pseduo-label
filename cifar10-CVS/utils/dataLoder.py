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
import cv2

# 数据预处理和加载
def default_loader(path):
    im = Image.open(path).convert('RGB')
    #im = np.asarray(im.resize((32, 32)))
    im = im.resize((32, 32))
    return im


class MyDataset(Dataset):
    def __init__(self, txt, transform=True, target_transform=None, loader=default_loader):
        f = open(txt, 'r')
        self.folder = txt.split('/')[-1].split('.')[0].split("_")[-1]
        #print(txt.split('/')[-1])
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
        self.len = len(self.imgs)

    def __getitem__(self, index):  # 类的特殊方法

        img_name, label = self.imgs[index]
        img_path = '/home/elgong/GEL/one_shot/paper/cifar10-CVS/data/' + self.folder + '/' + img_name
        img = self.loader(img_path)
        # label1 = 0
        # if 2 == label:
        #     while label1 != 2:
        #         img_name1, label1 = self.imgs[np.random.randint(0, self.len - 1)]
        #
        #     img_path1 = '/home/elgong/GEL/one_shot/paper/cifar10-CVS/data/' + self.folder + '/' + img_name1
        #     pro = np.random.random()
        #     if pro < 0.4:
        #         img1 = self.loader(img_path1)
        #         img1 = np.array(img1)
        #         img = np.array(img)
        #         img2 = np.floor(0.7 * img + 0.3 * img1).astype("uint8")
        #         img = Image.fromarray(img2)
        if self.transform is not None:
            img = self.transform(img)


        return img, label

    def __len__(self):
        return len(self.imgs)


class MyTestOneSample():
    def __init__(self, transform=True, loader=default_loader):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_flag = transform
        self.loader = loader

    def get_one_sample(self, img_path):

        img = self.loader(img_path)
        img = self.transform(img)
        img.unsqueeze_(0)
        return img


# 测试 为标注数据
class MyTest(Dataset):
    def __init__(self, txt, transform=True, target_transform=None, loader=default_loader):
        f = open(txt, 'r')
        # self.folder = txt.split('/')[-1].split('.')[0].split("_")[-1]
        #print(txt.split('/')[-1])
        imgs = []
        for line in f.readlines():
            img_name = line.strip()
            label = 1
            imgs.append((img_name, int(label)))

        self.imgs = imgs
        # random.shuffle(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_path = '/home/elgong/GEL/one_shot/paper/cifar10-CVS/data/unlabel/'

    def __getitem__(self, index):  # 类的特殊方法

        img_name, label = self.imgs[index]
        img_path = self.img_path + img_name
        #print(img_path)
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_name

    def __len__(self):
        return len(self.imgs)