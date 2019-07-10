import os

f = open("/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/1.txt","r")


dic = {}
with open("/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/2.txt","w") as f2:
    for line in f:
        l = line.split("_")
        if l[0] not in dic:
            dic[l[0]] = 1
        else:
            dic[l[0]]+=1
        if l[0] == l[1] and dic[l[0]] < 80:
            f2.write(line)
