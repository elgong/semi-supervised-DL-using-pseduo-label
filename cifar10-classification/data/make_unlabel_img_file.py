import os

with open("./first_unlabel_img_name.txt", "w") as f:
    # 未标注的图片名字
    img_list = os.listdir("./unlabel")

    # 拿去训练的400张图片名字
    img_list_train = os.listdir("./train")
    for img in img_list:
        if img not in img_list_train:
            f.write(img + "\n")