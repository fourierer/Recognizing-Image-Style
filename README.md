# Recognizing-Image-Style
Recognizing Image Style with Resnet or other model in pytorch

识别图像风格本质上还是图像分类任务



1.下载数据集

数据集使用Flickr Style dataset，可以直接百度搜索下载即可。由于数据集网站(https://www.flickr.com/photos/51299916@N07/)每张图片对应于一个url，数据集不太好下载。这里采用caffe官网给出的例子来下载数据集(https://github.com/BVLC/caffe).

下载好整个repositories之后，开始下载数据集：

在文件夹caffe-master/examples/finetune_flickr_style下有assemble_data.py文件用于下载数据，在caffe-master下运行指令：python examples/finetune_flickr_style/assemble_data.py --workers=-1 --images=-1 --seed 831486即可下载。

使用assemble_data.py文件可以下载该数据集的任意子集，这里下载全部的数据集。(调节参数images即可，数据集共80k张照片)。需要注意的是，原assemble_data.py文件是使用python2以及添加多线程下载数据，运行的时候会报错无法下载数据，需要使用下面的代码替换掉。

assemble_data.py

```python
""" 
Form a subset of the Flickr Style data, download images to dirname, and write 
Caffe ImagesDataLayer training file. 
"""  
import os  
import urllib.request #修改，Python3使用import urllib.request，Python2使用import urllib
import hashlib  
import argparse  
import numpy as np  
import pandas as pd  
from skimage import io  
import multiprocessing  
import socket  
# Flickr returns a special image if the request is unavailable.  
MISSING_IMAGE_SHA1 = '6a92790b1c2a301c6e7ddef645dca1f53ea97ac2'  

example_dirname = os.path.abspath(os.path.dirname(__file__))  
caffe_dirname = os.path.abspath(os.path.join(example_dirname, '../..'))  
training_dirname = os.path.join(caffe_dirname, 'data/flickr_style')  

#修改，将原来的download_image函数修改为mydownload_image函数  
def mydownload_image(args_tuple):  
    try:  
        url, filename = args_tuple  
        if not os.path.exists(filename):  
            urllib.request.urlretrieve(url, filename) #修改，Python3 使用urllib.request，Python2 使用urllib
        return True  
    except KeyboardInterrupt:  
        raise Exception()   
    except:  
        return False  

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(  
        description='Download a subset of Flickr Style to a directory')  
    parser.add_argument(  
        '-s', '--seed', type=int, default=0,  
        help="random seed")  
    parser.add_argument(  
        '-i', '--images', type=int, default=-1,  
        help="number of images to use (-1 for all [default])",  
    )  
    parser.add_argument(  
        '-w', '--workers', type=int, default=-1,  
        help="num workers used to download images. -x uses (all - x) cores [-1 default]."  
    )  
    parser.add_argument(  
        '-l', '--labels', type=int, default=0,  
        help="if set to a positive value, only sample images from the first number of labels."  
    )  

    args = parser.parse_args()  
    np.random.seed(args.seed)  
    # Read data, shuffle order, and subsample.  
    csv_filename = os.path.join(example_dirname, 'flickr_style.csv.gz')  
    df = pd.read_csv(csv_filename, index_col=0, compression='gzip')  
    df = df.iloc[np.random.permutation(df.shape[0])]  
    if args.labels > 0:  
        df = df.loc[df['label'] < args.labels]  
    if args.images > 0 and args.images < df.shape[0]:  
        df = df.iloc[:args.images]  

    # Make directory for images and get local filenames.  
    if training_dirname is None:  
        training_dirname = os.path.join(caffe_dirname, 'data/flickr_style')  
    images_dirname = os.path.join(training_dirname, 'images')  
    if not os.path.exists(images_dirname):  
        os.makedirs(images_dirname)  
    df['image_filename'] = [  
        os.path.join(images_dirname, _.split('/')[-1]) for _ in df['image_url']  
    ]  

    # Download images.  
    num_workers = args.workers  
    if num_workers <= 0:  
        num_workers = multiprocessing.cpu_count() + num_workers  
    print('Downloading {} images with {} workers...'.format(  
        df.shape[0], num_workers))  
    #pool = multiprocessing.Pool(processes=num_workers)  #修改，注释掉原来的多线程、多进程使用
    map_args = zip(df['image_url'], df['image_filename'])  
    #results = pool.map(download_image, map_args) #修改，注释掉原来的多线程、多进程使用 
    socket.setdefaulttimeout(6)  
    results = []  
    for item in map_args:
        value = mydownload_image(item)  #调用mydownload_image函数一个一个下载图片
        results.append(value)  
        if value == False:  
                print('Flase')  
        else:  
                print('1')  
    # Only keep rows with valid images, and write out training file lists.  
    print(len(results))  
    df = df[results]  
    for split in ['train', 'test']:  
        split_df = df[df['_split'] == split]  
        filename = os.path.join(training_dirname, '{}.txt'.format(split))  
        split_df[['image_filename', 'label']].to_csv(  
            filename, sep=' ', header=None, index=None)  
    print('Writing train/val for {} successfully downloaded images.'.format(  
        df.shape[0]))  
```



2.整理分类

下载好的数据集里面所有类别是混在一起的，需要整理分类。数据集下载在文件夹caffe-master\data\flickr_style\images中，caffe-master中还有train.txt和test.txt标签文件。文件caffe-master\examples\finetune_flickr_style\style_name.txt是各个类别的名称。

脚本文件classification.py用于将下载的数据集分类整理成train和test的各个类别存放在文件夹dataset中，代码如下：

```python
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
```



3.训练

