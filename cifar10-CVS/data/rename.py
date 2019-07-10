import os
path = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/unlabel"
count = 1
for f in os.listdir(path):
    if "tree" in f:
        os.rename(os.path.join(path,f),os.path.join(path,"tree_"+ str(count) + ".png"))
        count+=1
print(count)
