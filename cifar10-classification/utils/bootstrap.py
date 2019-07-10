import os
import numpy as np
import random


class Boostrap:

    def __init__(self, imgName = [], choosedNum=10):
        print("bootstrap ...")
    # bootstrap 不考虑类别均衡问题时
    def bootstrap(self, imgName = [], choosedNum = 10):
        """
        :param imgName:
        :param choosedNum:
        :return:
        """
        import random
        choosedImg = []
        for i in range(choosedNum):
            choosedImg.append(imgName[random.randint(0, len(imgName))])

        return choosedImg


    # 考虑类别均衡问题时
    def bootstrapBalance(self, imgName = [], choosedNumPerClass = 10):
        """
        :param imgName:
        :param choosedNum:
        :return:
        """
        import random
        choosedImg = []
        img_dic = {}
        for img_name in imgName:
            cls = img_name.split("_")[0]
            if cls not in img_dic:
                img_dic[cls] = img_name
            else:
                img_dic[cls].append(img_name)
        for cls in img_dic.keys():

            for i in range(choosedNumPerClass):
                choosedImg.append(imgName[random.randint(0, len(img_dic[cls]))])

        return choosedImg

bootstrap = Boostrap()
imgName = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
print(bootstrap.bootstrap(imgName,19))