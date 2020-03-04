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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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

train_dataset = torchvision.datasets.ImageFolder(root='/home/momo/data2/sun.zheng/flickr_style/train', transform=data_transform)
# 使用ImageFolder需要数据集存储的形式：每个文件夹存储一类图像
# ImageFolder第一个参数root : 在指定的root路径下面寻找图片
# 第二个参数transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
train_data = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
# 第一个参数train_dataset是上面自定义的数据形式
# 最后一个参数是线程数，>=1即可多线程预读数据

test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/data2/sun.zheng/flickr_style/test', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)

print(type(train_data))
print('data loaded done!')
# <class 'torch.utils.data.dataloader.DataLoader'>


# ##################创建网络模型###################
'''
自行创建网络模型模块，可以用于pytorch复现CaffeNet
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

# ImageNet预训练模块，更新所有层的参数
print('resnet model loaded begin!')
# 使用resnet50,进行预训练
model = models.resnet50(pretrained=True)
print(model)
print('resnet model loaded done!')
# 对于模型的每个权重，使其进行反向传播，即不固定参数
for param in model.parameters():
    param.requires_grad = True

'''
# ImageNet预训练模块，只更新最后一层的参数
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
'''


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

nums_epoch = 40  # 训练epoch的数量，动态调整

print('training begin!')
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
S = []

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
    s = 'Epoch {} ,Train Loss: {} ,Train  Accuracy: {} ,Test Loss: {} ,Test Accuracy: {}'.format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data))
    S.append(s);

    torch.save(model, '/home/momo/sun.zheng/Recognizing_Image_Style/model_F_l_0.01_SGD_epoch_40.pkl')
    print('model saved done!')
    print(losses)
    print(acces)
    print(eval_losses)
    print(eval_acces)
    print(S)
```



1.训练过程只更新最后一层的参数：

(1)学习率为0.01，迭代方法为SGD，训练30个epoch，模型保存为model_f_l_0.1_SGD_epoch_30.pkl。训练结果如下：

......
Epoch 10 ,Train Loss: 1.902140543921361 ,Train  Accuracy: 0.41654173868139277 ,Test Loss: 1.9437546963785208 ,Test Accuracy: 0.40381944444444445

......
Epoch 20 ,Train Loss: 1.8567710091840375 ,Train  Accuracy: 0.42908481578281765 ,Test Loss: 1.9234777677292918 ,Test Accuracy: 0.4086900385154062
......

Epoch 30 ,Train Loss: 1.828558410000976 ,Train  Accuracy: 0.4384050143326869 ,Test Loss: 1.912335047534868 ,Test Accuracy: 0.4108776844070962



(2)学习率为0.001，迭代方法为SGD，训练40个epoch，模型保存为model_f_l_0.01_SGD_epoch_40.pkl。训练结果如下：

......
Epoch 10 ,Train Loss: 2.281559232386147 ,Train  Accuracy: 0.3439155803195963 ,Test Loss: 2.27691650390625 ,Test Accuracy: 0.3451444553143146

......
Epoch 20 ,Train Loss: 2.113882165420346 ,Train  Accuracy: 0.37235334314550045 ,Test Loss: 2.1211741578345205 ,Test Accuracy: 0.3712380606749138

......
Epoch 30 ,Train Loss: 2.045651798713498 ,Train  Accuracy: 0.38550186606391923 ,Test Loss: 2.0619620225008797 ,Test Accuracy: 0.38386738144828747

......
Epoch 40 ,Train Loss: 2.0072273568409247 ,Train  Accuracy: 0.39226043418839357 ,Test Loss: 2.0252821679208792 ,Test Accuracy: 0.38980482885214174



将学习率定为0.01进行下面的训练：

2.训练过程更新所有的参数

(2)训练过程改变所有层的权重，学习率为0.01，迭代方法为SGD，训练40个epoch，模型保存为model_F_l_0.01_SGD_epoch_21.pkl。

（本来打算训练40个epoch，但中途发现训练结果严重过拟合，所以停止了训练了，保存了21个epoch的训练结果）

训练集loss：[2.350438571557766, 1.9172641951863358, 1.7942136380730607, 1.7071296529072086, 1.6328534864797826, 1.5606001795791997, 1.4783432902359381, 1.391603258760964, 1.2973330032534716, 1.1917561804375998, 1.0727995445088643, 0.943589751022618, 0.807379489991723, 0.667758800053015, 0.535146447071215, 0.4152965676493761, 0.3108352328219065, 0.22810486642325797, 0.16467725765414354, 0.12011081056623923, 0.08935694278376859];

训练集准确率：[0.3135118797308663, 0.41423333683767877, 0.4470793471404542, 0.47061277859545836, 0.49184385513036166, 0.5136820595037846, 0.5384448591253154, 0.5630801881833474, 0.5913687973086628, 0.625833158116064, 0.6681277596719933, 0.7085076745164003, 0.7557315759041211, 0.8076180088309504, 0.8545666000841042, 0.8959833631202692, 0.9312664266190075, 0.9560390822119429, 0.9737601187973087, 0.9848999947434819, 0.9910704899074853]

测试集loss：[2.028365217003168, 1.9077880943522734, 1.8536552564770568, 1.8174886563244987, 1.8170345974903481, 1.803846976336311, 1.8090978954352586, 1.8148543460696351, 1.8416614625968186, 1.8701898701050703, 1.916649343920689, 1.9919380346934001, 2.0653915755888996, 2.1646902981926415, 2.2747778331532196, 2.3746308672661876, 2.5098320269117167, 2.61287260055542, 2.6882708072662354, 2.770793914794922, 2.865333547779158]

测试集准确率：[0.38563000978288314, 0.41336428616510984, 0.42864313778086344, 0.43828729855676174, 0.44274383573171755, 0.44621419517377764, 0.44661228225195654, 0.45024240248674574, 0.447516278507111, 0.4425840749389885, 0.4399486793107801, 0.4341059049692838, 0.42953957334006565, 0.4237648452621392, 0.41898582796852646, 0.4136469904485399, 0.40987880533114535, 0.40190654453841623, 0.40949583964487085, 0.40947315755701424, 0.4050070873306404]



绘制曲线图观察：

![Figure_1](/Users/momo/Documents/momo学习笔记/工作汇报/2020.03.04/Figure_1.png)

蓝色为训练集曲线，红色为测试集曲线，横坐标为训练轮次epoch，纵坐标为准确率。随着训练轮次的增加，模型逐渐过拟合，从曲线和数据分布可以得出，当epoch=7或者8时，模型在测试集上可以达到最高的准确率（约45%）和最低的loss。

再次训练8个epoch，保存模型为model_F_l_0.01_SGD_epoch_8.pkl。



