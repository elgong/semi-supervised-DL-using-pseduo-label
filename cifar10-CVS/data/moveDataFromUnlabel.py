import os
import random
###  rename 具有剪切功能

src_path = "/home/elgong/GEL/one_shot/paper/cifar10-CVS/data/unlabel/"
dst_path = "/home/elgong/GEL/one_shot/paper/cifar10-CVS/data/train/"

num = 400

img_in_src = os.listdir(src_path)

dic = {}
for img in img_in_src:
    cls = img.split("_")[0]
    if cls not in dic:
        dic[cls] = [img]
    else:
        dic[cls].append([img])


# 

for cls in dic.keys():
    moved = []
    for i in range(len(dic[cls])):
       
        img = dic[cls][random.randint(0, len(dic[cls]))][0]
        if img not in moved:

            
        #print(img)
            os.rename(src_path + img,
                     dst_path + img
            
                   )
            moved.append(img)
        if len(moved) >=400:
            break
    
       # del dic[cls][dic[cls].index(img)]

