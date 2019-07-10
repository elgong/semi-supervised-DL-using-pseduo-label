import os

###  rename 具有剪切功能

src_path = "./unlabel/"
dst_path = "./train/"

img_file = "/home/elgong/GEL/one_shot/torch3/data/newLabelImage.txt"


img_in_dst = os.listdir(dst_path)
count = 0
with open(img_file, "r") as f:
    for line in f:
        img, cls = line.strip().split(",")
        print(img)
        if img not in img_in_dst:
            try:
                os.rename(src_path+img, dst_path + cls+"_" + img)
            except IOError:
                count += 1
print(count)
        
