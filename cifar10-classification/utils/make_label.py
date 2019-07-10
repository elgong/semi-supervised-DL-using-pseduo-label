import os

def make_label(src, obj, cls1, cls2):
    
    
    class_name = [cls1, cls2]
    with open(obj, "w") as f:
        for root, dic, fList in os.walk(src):
            for img in fList:
                cls = img.split("_")[0]
               # if cls not in class_name:
                  #  class_name.append(cls)
                if cls in class_name:

                    f.write(img + "," + str(class_name.index(cls)) + "\n")
                

src1 = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/val"

obj1 = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/val.txt"


src2 = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/train"

obj2 = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/data/train.txt"

cls1 = input("inupt 1:")
cls2 = input("inupt 2:")
make_label(src1, obj1, cls1, cls2)
make_label(src2, obj2, cls1, cls2)
