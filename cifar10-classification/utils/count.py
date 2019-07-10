import os

train = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/train"

cls_dic = {}

train_list = os.listdir(train)

for img in train_list:
    label = img.split("_")[0]
    if label not in cls_dic:
        cls_dic[label] = 1
    else:
        cls_dic[label] += 1
print(cls_dic)


