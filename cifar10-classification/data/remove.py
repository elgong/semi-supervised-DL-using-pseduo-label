import os

img_list = os.listdir("/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/train")


img_ying = []

with open("/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/train.txt", "r") as f:
    
    for line in f:
        
        img_ying.append(line.strip().split(",")[0])

count = 0
for img in img_list:
    if img not in img_ying:
        os.remove("/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/train/" + img)


     
