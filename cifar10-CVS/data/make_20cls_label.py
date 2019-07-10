import os


train_path = "./train"
val_path = "./val"

train_txt = "../train.txt"
val_txt = "../val.txt"
class_name = []
with open(train_txt, "w") as f:
    
    for root, dic, fList in os.walk(train_path):
        for img in fList:
            cls = img.split("_")[0]
            if cls not in class_name:
                class_name.append(cls)
            if cls in class_name:

                f.write(img + "," + str(class_name.index(cls)) + "\n")



with open(val_txt, "w") as f:
    
    for root, dic, fList in os.walk(val_path):
        for img in fList:
            cls = img.split("_")[0]
            if cls in class_name:

                f.write(img + "," + str(class_name.index(cls)) + "\n")
