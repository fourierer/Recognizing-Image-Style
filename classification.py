#@author:sun zheng,2020.02.05
# 此脚本将下载好的数据集分类整理

import os
import shutil

#读取train.txt文件，将image文件夹中的属于训练集的图片移动到image\train文件夹当中，并分好类
f_train = open("./train.txt")
line_train = f_train.readlines()
for line in line_train:
    filepath = line.split()[0] # 图片所在路径
    category = line.split()[1] # 图片所属类别
    # print(filepath + ':' + category) # 测试读取是否正确
    folderpath = './dataset/train/' + category+ '/'
    folder= os.path.exists(folderpath)
    if not folder:
        os.makedirs(folderpath)
    shutil.copy(filepath, folderpath)

#读取test.txt文件，将image文件夹中的属于测试集的图片移动到image\test文件夹当中，并分好类
f_test = open("./test.txt")
line_test = f_test.readlines()
for line in line_test:
    filepath = line.split()[0]
    category = line.split()[1]
    folderpath = './dataset/test/' + category +'/'
    folder = os.path.exists(folderpath)
    if not folder:
        os.makedirs(folderpath)
    shutil.copy(filepath, folderpath)

