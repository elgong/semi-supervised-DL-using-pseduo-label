#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 19-5-24 下午2:29
#@Author: elgong_hdu
#@File  : makeLabel.py
import pandas as pd

####  20个类别
Class = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

####  380个组合
union_group = []
for i in range(len(Class)):
    for j in range(len(Class)):
        if i == j:
            continue
        union_group.append((Class[i] + "_" + Class[j]))

# del union_group[0]


####  打印准确度
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

####  投票器 版本1 -- 输出为标签时
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





# 投票器 版本2
def vote_version2(cls = "apple", vote_mean_Threshold = 0.9, cls_Threshold = 0.8, predict_path = "/home/elgong/GEL/one_shot/torch/pytorch-cifar-master/result_predict_unlabel.csv"):
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
            if vote_count >= vote_mean_Threshold:
                # pass
                img_list.append(img_name)

    # 上帝视角, 验证结果的准确性
    acc(cls, img_list)






# ############    投票器 版本3-------sigmoid 输出
#
predict_path = "/home/elgong/GEL/one_shot/torch3/result_predict_unlabel9600.csv"

#predict_path = "/home/elgong/GEL/one_shot/torch3/result_predict_unlabel222222222222.csv"

def vote_version3(Class,  vote_mean_Threshold = 0.9, _predict_path = predict_path):
    """
    投票器 版本3-------sigmoid 输出
    :param Class:
    :param vote_mean_Threshold:  set内投票阈值
    :param _predict_path:
    :return:
    """

    from tqdm import tqdm
    tmp_res = {}
    final_res = {}
    for cls in Class:
        final_res[cls] = []
    #df_2 = pd.read_csv(df_name, header=0, index_col=0)

    with open(predict_path, "r") as df:

        # 逐个图片操作
        for line in tqdm(df):
            # 跳过第一行
            if "airplane_automobile" in line:
                continue
            line = line.strip().split(",")
            img_name = line[0]
            tmp_res[img_name] = []
            label = line[1:]

            # 对每个类别组成的set都投票打分
            for cls in Class:

                #  保存相关模型
                class1 = {"first": [], "second": []}

                # 投票分数
                vote_count = 0.0

                # 投票器的数量
                vote_num = 18.0

                ##### 这里排除第一个模型
                # if cls == "apple" or cls == "bettle":
                #     vote_num = 37.0

                for group in union_group:
                    cls1, cls2 = group.split("_")
                    if cls + "_" in group:

                        #### 去掉效果差的模型
                       # model_score = df_2.loc[cls1, cls2]
                       #  if model_score < model_Threshold:
                       #      vote_num = vote_num-1
                       #
                       #      continue
                        class1["first"].append(group)
                    elif "_" + cls in group:
                        # model_score = df_2.loc[cls2, cls1]
                        # if model_score < model_Threshold:
                        #     vote_num = vote_num - 1
                        #     continue
                        class1["second"].append(group)


                # 概率投票
                for g1 in class1["first"]:
                    #if "apple_beetle" in g1:
                       #vote_num -= 1
                    score1, score2 = label[union_group.index(g1)].split("/")
                    # if max(float(score1), float(score2)) < cls_Threshold:
                    #     score1 = 0
                    #     score2 = 0

                    vote_count += float(score1)  #if float(score1) > float(float(score2)) else 0
                # 统计其他set模型中 票数
                for g2 in class1["second"]:
                    #if "apple_beetle" in g2:
                        #vote_num -= 1
                    score3, score4 = label[union_group.index(g2)].split("/")
                    # if max(float(score3), float(score4)) < cls_Threshold:
                    #     score3 = 0
                    #     score4 = 0
                    vote_count += float(score4) #if float(score3) < float(score4)else 0

                vote_count = vote_count/vote_num
                tmp_res[img_name].append(vote_count)
            score = tmp_res[img_name]

            flag = 0
            max_score = 0.0
            for i in score:
                if i >= vote_mean_Threshold:
                    flag += 1
                    max_score = max(i, max_score)
            if 0 <flag <= 1:
                final_res[Class[score.index(max_score)]].append(img_name)


    with open("./newLabelImage.txt", "w+") as f:
        for key in final_res.keys():
            print(key, len(final_res[key]))
            for img in final_res[key]:
                f.write(img +"," + key + "\n")
    return final_res










############    投票器 版本4 -------sigmoid 输出  --- 加网络权重

predict_path = "/home/elgong/GEL/one_shot/paper/cifar10-CVS/result/result_predict_unlabel.csv"

#predict_path = "/home/elgong/GEL/one_shot/torch3/result_predict_unlabel222222222222.csv"
df_name = "/home/elgong/GEL/one_shot/torch3/result_train_acc___2019-06-11 19:11:02.csv"
def vote_version31(Class,  vote_mean_Threshold = 0.9, _predict_path = predict_path):
    """
    投票器 版本3-------sigmoid 输出
    :param Class:
    :param vote_mean_Threshold:  set内投票阈值
    :param _predict_path:
    :return:
    """

    from tqdm import tqdm
    tmp_res = {}
    final_res = {}
    for cls in Class:
        final_res[cls] = []
    df_2 = pd.read_csv(df_name, header=0, index_col=0)

    with open(predict_path, "r") as df:

        # 逐个图片操作
        for line in tqdm(df):
            # 跳过第一行
            if "apple_beetle" in line:
                continue
            line = line.strip().split(",")
            img_name = line[0]
            tmp_res[img_name] = []
            label = line[1:]

            # 对每个类别组成的set都投票打分
            for cls in Class:

                #  保存相关模型
                class1 = {"first": [], "second": []}

                # 投票分数
                vote_count = 0.0

                # 投票器的数量
                vote_num = 38.0

                ## 相关模型的总和
                mean_model_acc = 0.0


                for group in union_group:
                    cls1, cls2 = group.split("_")

                    model_acc = df_2.loc[cls1, cls2]
                    if cls + "_" in group:
                        mean_model_acc = mean_model_acc + float(model_acc)
                        class1["first"].append((group, float(model_acc)))
                    elif "_" + cls in group:

                        class1["second"].append((group, float(model_acc)))
                        mean_model_acc = mean_model_acc + float(model_acc)



                # 概率投票
                for g1 in class1["first"]:
                    model_acc = g1[1]
                    model_name = g1[0]
                    score1, score2 = label[union_group.index(model_name)].split("/")

                    vote_count += float(score1)*(model_acc/mean_model_acc)  #if float(score1) > float(float(score2)) else 0
                # 统计其他set模型中 票数
                for g2 in class1["second"]:
                    model_acc = g2[1]
                    model_name = g2[0]
                    score3, score4 = label[union_group.index(model_name)].split("/")

                    vote_count += float(score4)*(model_acc/mean_model_acc) #if float(score3) < float(score4)else 0


                #vote_count = vote_count/vote_num
                tmp_res[img_name].append(vote_count)
            score = tmp_res[img_name]

            flag = 0
            max_score = 0.0
            for i in score:
                if i >= vote_mean_Threshold:
                    flag += 1
                    max_score = max(i, max_score)
            if 0 <flag <= 1:
                final_res[Class[score.index(max_score)]].append(img_name)

    with open("./newLabelImage.txt", "w+") as f:
        for key in final_res.keys():
            print(key, len(final_res[key]))
            for img in final_res[key]:
                f.write(img +"," + key + "\n")

    return final_res


#vote_version3(Class, 0.96)



# #### 画图
y1 = []
y2 = []

x_val= [    0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,
        0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,
        0.95,0.96,0.97,0.98,0.99]

for x in x_val:
    res = vote_version3(Class, x)  #0.95, 0.8

    sum_true = []
    sum_flase = []
    for cls in res.keys():
        error_list = []
        true_list = []
        print("投票筛选的类别是： ", cls)
        print("参与的set个数： ", len(Class))
        print("选出的图片个数：", len(res[cls]))

        for img_name in res[cls]:
            if cls in img_name:
                true_list.append(img_name)
                sum_true.append(img_name)
            else:
                error_list.append(img_name)
                sum_flase.append(img_name)

        print("准确率", len(true_list) / max(len(res[cls]), 1))  # 防止除零错误
        print("数量：", len(true_list), len(error_list))
        print("错误的图片是：", str(error_list))
        print("正确的图片是：", str(true_list))
        print("\n\n")
    print(len(sum_true)/(max((len(sum_true)+len(sum_flase)),1)))
    print(len(sum_true))
    num =len(sum_true)
   # if len(sum_true) +200 < (len(sum_true) + len(sum_flase)):
   #     num = len(sum_true) +200
    acc = (num) / (max((len(sum_true) + len(sum_flase)), 1))


    y1.append(acc)

    num = len(sum_true)+len(sum_flase)#+200

    
    y2.append(num)
print(y1, y2)

## 画图

import matplotlib.pyplot as plt


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()


p1, = host.plot(x_val, y1, "b-", label="Precision")
p2, = par1.plot(x_val, y2, "r-", label="Choosed Number")

host.set_xlim(x_val[0], 1)
host.set_ylim(0, 1)
par1.set_ylim(0, 40000)

host.set_xlabel("vote_Threshold")
host.set_ylabel("Precision")
par1.set_ylabel("Choosed Number")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2]

host.legend(lines, [l.get_label() for l in lines])

plt.show()
