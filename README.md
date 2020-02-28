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

下载好的数据集里面所有类别是混在一起的，需要整理分类。数据集下载在文件夹caffe-master\data\flickr_style\images中，caffe-master\data\flickr_style中还有train.txt和test.txt标签文件。文件caffe-master\examples\finetune_flickr_style\style_name.txt是各个类别的名称。

将脚本文件classification.py放到文件夹caffe-master\data\flickr_style当中运行，用于将下载的数据集分类整理成train和test的各个类别存放在文件夹dataset中，代码如下：

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

识别图像风格本质上还是图像分类任务，flickr style dataset数据集共20个类别，相较于ImageNet数据集类别数少了很多，训练方法准备采用下面两种。

(1)利用pytorch复现CaffeNet进行分类(使用CaffeNet进行微调，在测试集上的准确率最好结果为39%)；

(2)在pytorch框架下，使用ResNet等模型进行数据集的分类，可以使用ImageNet进行预训练更改所有参数，或者更改分类数之后只训练ResNet最后一层的权重。

采用第二种方法训练：

3.1 使用ResNet-50模型进行预训练

代码：Network_resnet-50.py，与ImageNet的代码大体相同

```python
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定一块gpu为可见
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定四块gpu为可见
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# #############创建数据加载器###################
print('data loaded begin!')
# 预处理，将各种预处理组合在一起
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])

train_dataset = torchvision.datasets.ImageFolder(root='/data/sz/raw/flickr_style/train', transform=data_transform)
# 使用ImageFolder需要数据集存储的形式：每个文件夹存储一类图像
# ImageFolder第一个参数root : 在指定的root路径下面寻找图片
# 第二个参数transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
train_data = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
# 第一个参数train_dataset是上面自定义的数据形式
# 最后一个参数是线程数，>=1即可多线程预读数据

test_dataset = torchvision.datasets.ImageFolder(root='/data/sz/raw/flickr_style/test', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

print(type(train_data))
print('data loaded done!')
# <class 'torch.utils.data.dataloader.DataLoader'>


# ##################创建网络模型###################
'''
这里选择从torch里面直接导入resnet，不搭建网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Torch.nn.Conv2d(in_channels，out_channels，kernel_size，stride = 1，
        # padding = 0，dilation = 1，groups = 1，bias = True)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3),  # 16,298,298
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3),  # 32,296,296
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))  # 32,148,148


'''
'''
print('resnet model loaded begin!')
# 使用resnet50,进行预训练
model = models.resnet50(pretrained=True)
print(model)
print('resnet model loaded done!')
# 对于模型的每个权重，使其进行反向传播，即不固定参数
for param in model.parameters():
    param.requires_grad = True

'''
print('resnet model loaded begin!')
model = models.resnet50(pretrained=True)
print(model)
print('resnet model loaded done!')
# 对于模型的每个权重，使其不进行反向传播，即固定参数
for param in model.parameters():
    param.requires_grad = False
# 修改最后一层的参数，使其不固定，即不固定全连接层fc
for param in model.fc.parameters():
    param.requires_grad = True



# 修改最后一层的分类数
class_num = 20  # flickr_style的类别数是20
channel_in = model.fc.in_features  # 获取fc层的输入通道数
model.fc = nn.Linear(channel_in, class_num)  # 最后一层替换


# ##############训练#################

# 在可见的gpu中，指定第一块卡训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), 1e-1)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-2)

nums_epoch = 10  # 训练10个epoch

print('training begin!')
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    model = model.train()
    print('Epoch ' + str(epoch+1) + ' begin!')
    for img, label in train_data:
        img = img.to(device)
        label = label.to(device)

        # 前向传播
        out = model(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        print('Train loss in current Epoch' + str(epoch+1) + ':' + str(loss))
        #print('BP begin!')
        # 反向传播
        loss.backward()
        #print('BP done!')
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        print('Train accuracy in current Epoch' + str(epoch+1) + ':' + str(acc))

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    print('Epoch' + str(epoch+1)  + ' Train  done!')
    print('Epoch' + str(epoch+1)  + ' Test  begin!')
    # 每个epoch测一次acc和loss
    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img1, label1 in test_data:
        img1 = img1.to(device)
        label1 = label1.to(device)
        out = model(img1)

        loss = criterion(out, label1)
        # print('Test loss in current Epoch:' + str(loss))

        # 记录误差
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label1).sum().item()
        acc = num_correct / img1.shape[0]
        eval_acc += acc

    print('Epoch' + str(epoch+1)  + ' Test  done!')
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('Epoch {} ,Train Loss: {} ,Train  Accuracy: {} ,Test Loss: {} ,Test Accuracy: {}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
            eval_acc / len(test_data)))
    torch.save(model, '/home/sz/Recognizing_Image_Style/model_F.pkl')
    print('model saved done!')
```

(1)训练过程只改变最后全连接层的权重，学习率为0.1，迭代方法为SGD，训练10个epoch，模型保存为model_f.pkl。训练结果如下：

......

Epoch 10 ,Train Loss: 1.9965185834140313 ,Train  Accuracy: 0.4061264718250631 ,Test Loss: 2.129829451149585 ,Test Accuracy: 0.36561553269376423



(2)训练过程改变所有层的权重，学习率为0.1，迭代方法为SGD，训练10个epoch，模型保存为model_F.pkl。训练结果如下：

......

Epoch 10 ,Train Loss: 0.6007640998637472 ,Train  Accuracy: 0.801908565888205 ,Test Loss: 2.8319117906046847 ,Test Accuracy: 0.38415397408963586

此时结果严重过拟合；



(3)在ResNet-50的基础上进行微调，调整学习率为0.01，训练10个epoch，结果如下：

Epoch 10 ,Train Loss: 1.8763645724154334 ,Train  Accuracy: 0.42409208751370037 ,Test Loss: 1.926366437883938 ,Test Accuracy: 0.40895337301587303

此时几乎没有过拟合现象，并且测试集上的训练结果已经超过CaffeNet，为40.1%。



(4)在ResNet-50的基础上，调整学习率为0.01，训练20个epoch，结果如下：

......
Epoch 14 ,Train Loss: 1.851296254358548 ,Train  Accuracy: 0.4299944408144339 ,Test Loss: 1.912512249806348 ,Test Accuracy: 0.4126152544351074
......
Epoch 15 ,Train Loss: 1.8466478076132702 ,Train  Accuracy: 0.4303059923277969 ,Test Loss: 1.9137008283652512 ,Test Accuracy: 0.4143148926237162

......
Epoch 16 ,Train Loss: 1.8424751091120004 ,Train  Accuracy: 0.4338068301576596 ,Test Loss: 1.9163187695484536 ,Test Accuracy: 0.40984112394957983

......
Epoch 17 ,Train Loss: 1.8378852556270608 ,Train  Accuracy: 0.4341032322316837 ,Test Loss: 1.9090858978383682 ,Test Accuracy: 0.41270643674136326

......
Epoch 18 ,Train Loss: 1.83375675025371 ,Train  Accuracy: 0.43598308005227215 ,Test Loss: 1.9098872273576026 ,Test Accuracy: 0.4104086426237162

......
Epoch 19 ,Train Loss: 1.8296146733836616 ,Train  Accuracy: 0.43560632008262373 ,Test Loss: 1.9032716599165225 ,Test Accuracy: 0.4161626108776844

......

Epoch 20 ,Train Loss: 1.8266588061829359 ,Train  Accuracy: 0.4376198781721608 ,Test Loss: 1.9032626444218206 ,Test Accuracy: 0.4139676704014939

......

测试集上测试结果约为41%，多训练10个epoch准确率没有太大改动。

后续的微调可以继续调整其他参数。

