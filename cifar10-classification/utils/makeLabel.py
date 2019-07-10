#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 19-5-24 下午2:29
#@Author: elgong_hdu
#@File  : makeLabel.py

import os

Class = ("apple", "orange", "bed", "bottle", "bus", "clock", "elephant", "keyboard", "motorcycle",
         "mushroom", "rocket", "rose", "tank", "television", "tiger", "train", "turtle",
         "tree", "woman", "worm")


# 投票阈值


# 380个组别
union_group = []
for i in range(len(Class)):
    for j in range(len(Class)):
        if i == j:
            continue
        union_group.append((Class[i] + "_" + Class[j]))



def acc(cls, img_list):
    true_num = 0.0
    error_list = []
    true_list = []
    for img in img_list:
        if cls in img:
            true_num += 1
            true_list.append(img)
        else:
            error_list.append(img)
    print("投票筛选的类别是： ", cls)
    print("参与的投票模型数量： 38")
    print("选出的图片个数：", len(img_list))
    print("准确率", true_num / max(len(img_list), 1))
    print("错误的图片是：",str(error_list))
    print("正确的图片是：", str(true_list))

    print("\n\n")

# 投票器 版本1
def vote_version1(cls = "apple", Threshold = 38, predict_path = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/result_predict_unlabel.csv"):
    """
    版本1: 预测是01的时候
    :param cls:   要筛选的类别
    :param Threshold:  投票的阈值 （0～38）
    :param predict_path:
    :return:
    """

    #  保存相关模型
    class1 = {"first": [], "second": []}
    for group in union_group:
        if cls + "_" in group:
            class1["first"].append(group)
        elif "_" + cls in group:
            class1["second"].append(group)

    # 记住投票出来的结果
    img_list = []
    with open(predict_path, "r") as df:

        for line in df:
            # 跳过第一行
            if "apple_orange" in line:
                continue

            vote_count = 0
            line = line.strip().split(",")
            img_name = line[0]
            label = line[1:]
            # print(len(label))
            # 统计set1 票数
            # print(type(class1["first"]))
            for g1 in class1["first"]:
                vote_count += (1 - float(label[union_group.index(g1)]))
            # 统计其他set模型中 票数
            for g2 in class1["second"]:
                vote_count += (float(label[union_group.index(g2)]))

            if vote_count >= Threshold:
                # pass
                img_list.append(img_name)

    # 上帝视角, 验证结果的准确性
    acc(cls, img_list)

# for cls in Class:
#     vote_version1(cls)


# 投票器 版本2
def vote_version2(cls = "apple", vote_Threshold = 0.9, cls_Threshold = 0.8, predict_path = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/result_predict_unlabel.csv"):
    """
    版本2: 预测是概率的时候
    :param cls:   要筛选的类别
    :param Threshold:  投票的阈值 （0～38）
    :param predict_path:
    :return:
    """

    #  保存相关模型
    class1 = {"first": [], "second": []}
    for group in union_group:
        if cls + "_" in group:
            class1["first"].append(group)
        elif "_" + cls in group:
            class1["second"].append(group)

    # 记住投票出来的结果
    img_list = []
    with open(predict_path, "r") as df:

        for line in df:
            # 跳过第一行
            if "apple_orange" in line:
                continue

            vote_count = 0
            line = line.strip().split(",")
            img_name = line[0]
            label = line[1:]

            # 概率投票
            for g1 in class1["first"]:
                score1, score2 = label[union_group.index(g1)].split("/")
                if max(float(score1), float(score2)) < cls_Threshold:
                    score1 = 0
                    score2 = 0

                vote_count += float(score1) if float(score1) > float(float(score2)) else 0
            # 统计其他set模型中 票数
            for g2 in class1["second"]:
                score3, score4 = label[union_group.index(g2)].split("/")
                if max(float(score3), float(score4)) < cls_Threshold:
                    score3 = 0
                    score4 = 0

                vote_count += float(score4) if float(score3) < float(float(score4)) else 0

            vote_count = vote_count/38.0
            if vote_count >= vote_Threshold:
                # pass
                img_list.append(img_name)

    # 上帝视角, 验证结果的准确性
    acc(cls, img_list)

for cls in Class:
    vote_version2(cls, 0.86, 0.8)