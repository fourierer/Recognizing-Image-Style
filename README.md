# Recognizing-Image-Style
Recognizing Image Style with Resnet or other model in pytorch

识别图像风格本质上还是图像分类任务



1.下载数据集

数据集使用Flickr Style dataset，可以直接百度搜索下载即可。由于数据集网站(https://www.flickr.com/photos/51299916@N07/) 每张图片对应于一个url，数据集不太好下载。这里采用caffe官网给出的例子来下载数据集(https://github.com/BVLC/caffe).

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



4.结果

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



5.评估及分析

对测试集进行batch批量测试，Test.py：

```python
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
device = torch.device('cuda')
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])

print('Test data load begin!')
test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/data2/sun.zheng/flickr_style/test', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
print(type(test_data))
print('Test data load done!')

print('load model begin!')
model = torch.load('/home/momo/sun.zheng/Recognizing_Image_Style/model_F_l_0.01_SGD_epoch_8.pkl')
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
print('load model done!')


#测试单个图像属于哪个类别
'''
torch.no_grad()
img = Image.open('/home/momo/mnt/data2/datum/raw/val2/n01440764/ILSVRC2012_val_00026064.JPEG')
img = transform(img).unsqueeze(0)
img_= img.to(device)
outputs = net(img_)
_, predicted = torch.max(outputs,1)
print('this picture maybe:' + str(predicted))
'''
#批量测试准确率,并输出所有测试集的平均准确率
eval_acc = 0
torch.no_grad()
for img1, label1 in test_data:
    img1 = img1.to(device)
    label1 = label1.to(device)
    out = model(img1)

    _, pred = out.max(1)
    print(pred)
    print(label1)
    num_correct = (pred == label1).sum().item()
    acc = num_correct / img1.shape[0]
    print('Test acc in current batch:' + str(acc))
    eval_acc +=acc

print('final acc in Test data:' + str(eval_acc / len(test_data)))
```

测试集上的平均精度为45.37，其中发现有的batch测试结果大于45%(约60%)，有的batch测试结果小于45%(约30%)，模型的测试结果跟数据集中不同类别的样本有关。

接下来对每个类别进行测试，观察错分样本的个数，得到每个类样本的分类准确率，classify_test_data_result.py:

```python
# 此脚本用于识别测试数据集中每一类，将分类正确的和错误的分别置于每一类文件中两个文件夹里面
import shutil
import os
import torch
import torchvision
import D  # 自己写的D.py，为了方便后续分类
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
#定义数据转换格式transform
device = torch.device('cuda')
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std
])

#加载模型
print('load model begin!')
model = torch.load('/home/momo/sun.zheng/Recognizing_Image_Style/model_F_l_0.01_SGD_epoch_8.pkl')
model.eval()
model= model.to(device)
print('load model done!')


#从数据集中加载测试数据
test_dataset = D.ImageFolder(root='/home/momo/data2/sun.zheng/flickr_style/test', transform=data_transform)  # 这里使用自己写的data.py文件，ImageFolder不仅返回图片和标签，还返回图片的路径，方便后续方便保存
#test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

'''
路径/home/momo/data2/sun.zheng/flickr_style/test里面是原始测试集，
路径/home/momo/sun.zheng/Recognizing_Image_Style/test_result里面批量建立文件夹，每一类数据建立一个文件夹，
文件夹包括right和wrong两个文件夹，分别表示在保存的模型测试下，该类正确分类和错误分类的样本集合
'''


count = 0  # 当前类别测试图片个数



for img1, label1, path1 in test_data:
    count = count + 1

    #img11 = img1.squeeze()  # 此处的squeeze()去除size为1的维度，,将(1,3,224,224)的tensor转换为(3,224,224)的tensor
    #new_img1 = transforms.ToPILImage()(img11).convert('RGB')  # new_img1为PIL.Image类型
    img1 = img1.to(device)  # img1是tensor类型，规模是(1,3,224,224),gpu的tensor无法转换成PIL，所以在转换之后再放到gpu上
    label1 = label1.to(device)
    out = model(img1)
    _, pred = out.max(1)  # pred是类别数，tensor类型
    print(count)
    #print(path1[0])
    #print(type(path1[0]))

    if pred == label1:
        #将分对的图像放在right文件夹里面
        img_path = '/home/momo/sun.zheng/Recognizing_Image_Style/test_result/' + str(label1[0]) + '/right/'
        folder = os.path.exists(img_path)
        if not folder:
            os.makedirs(img_path)
        shutil.copy(path1[0], img_path)
    else:
        #将错的图像放在wrong文件夹里面
        img_path = '/home/momo/sun.zheng/Recognizing_Image_Style/test_result/' + str(label1[0]) + '/wrong/'
        folder = os.path.exists(img_path)
        if not folder:
            os.makedirs(img_path)
        shutil.copy(path1[0], img_path)
```



第1类：Detailed

Right:210,Wrong:359,正确率：36.9%



第2类：Pastel(柔和的)

Right:161,Wrong:417,正确率：27.9%



第3类：Melancholy(忧郁的)

Right:112,Wrong:458,正确率：19.6%



第4类：Noir(黑色的)

Right:440,Wrong:270,正确率：62%



第5类：HDR

Right:387,Wrong:275,正确率：58.5%



第6类：Vintage(古老的)

Right:254,Wrong:398,正确率：39%



第7类：Long Exposure(长时间曝光)

Right:408,Wrong:227,正确率：64.2%



第8类：Horror(恐怖的)

Right:406,Wrong:314,正确率：56.4%



第9类：Sunny(阳光的)

Right:435,Wrong:249,正确率：63.6%



第10类：Bright(明亮的)

Right:169,Wrong:479,正确率：26.1%



第11类：Hazy(朦胧的)

Right:417,Wrong:261,正确率：61.5%



第12类：Bokeh(散景的)

Right:289,Wrong:385,正确率：42.9%



第13类：Serene(宁静的)

Right:235,Wrong:496,正确率：32.1%



第14类：Texture(纹理的)

Right:222,Wrong:458,正确率：32.6%



第15类：Ethereal(飘渺的)

Right:340,Wrong:294,正确率：53.6%



第16类：Macro(微距摄影的)

Right:484,Wrong:170,正确率：74%



第17类：Depth of Field(景深的)

Right:90,Wrong:558,正确率：13.9%



第18类：Geometric Composition(几何组成的)

Right:306,Wrong:334,正确率：47.8%



第19类：Minimal

Right:386,Wrong:257,正确率：60.0%



第20类：Romantic(浪漫的)

Right:162,Wrong:461,正确率：26%



从上述结果中可以看出，人容易分辨的类别，ResNet模型的准确率也比较高，如Noir(黑色的)、Horror(恐怖的)、Hazy(朦胧的)以及Macro(微距摄影的)；相对的一些抽象的概念，比如Melancholy(忧郁的)、Depth of Field(景深的)以及Romantic(浪漫的)。要想提高测试集整体的准确率，就需要在这些抽象的类别上提高准确率。

改进：考虑到一个图片分风格不是由图片上某一个物体或者说由局部特征决定，而应该考虑图片上各部分之间的联系和整体构图。传统的卷积神经网络如ResNet，就是在提取一个图片上的局部特征，可以在ImageNet上表现很好，但是无法分辨一个图像的风格。后续改进考虑在ResNet(或者其他卷积神经网络模型)的基础之上引入Non-local机制，使得提取的特征是全局的，进而利用这些全局特征来进一步分类。





**进一步分析：**

暂时不添加Non-local机制，进一步分析不同类别之间的界限，观察各个类别误分的样本，主要误分为哪些类别。

1.classify_test_data_to_other.py，在classify_test_data.py的基础上略加修改

```python
# 此脚本用于识别测试数据集中每一类，将分类正确的和错误的分别置于每一类文件中两个文件夹里面，并且错分的类别要明确是错分为什么类别
import shutil
import os
import torch
import torchvision
import D  # 自己写的D.py，为了方便后续分类
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
#定义数据转换格式transform
device = torch.device('cuda')
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std
])

#加载模型
print('load model begin!')
model = torch.load('/home/momo/sun.zheng/Recognizing_Image_Style/model_F_l_0.01_SGD_epoch_8.pkl')
model.eval()
model= model.to(device)
print('load model done!')


#从数据集中加载测试数据
test_dataset = D.ImageFolder(root='/home/momo/data2/sun.zheng/flickr_style/test', transform=data_transform)  # 这里使用自己写的data.py文件，ImageFolder不仅返回图片和标签，还返回图片的路径，方便后续方便保存
#test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

'''
路径/home/momo/data2/sun.zheng/flickr_style/test里面是原始测试集，
路径/home/momo/sun.zheng/Recognizing_Image_Style/test_result_to_other里面批量建立文件夹，每一类数据建立一个文件夹，
文件夹包括right和wrong两个文件夹，分别表示在保存的模型测试下，该类正确分类和错误分类的样本集合，错分文件夹里面是具体错分的类别
'''


count = 0  # 当前类别测试图片个数



for img1, label1, path1 in test_data:
    count = count + 1

    #img11 = img1.squeeze()  # 此处的squeeze()去除size为1的维度，,将(1,3,224,224)的tensor转换为(3,224,224)的tensor
    #new_img1 = transforms.ToPILImage()(img11).convert('RGB')  # new_img1为PIL.Image类型
    img1 = img1.to(device)  # img1是tensor类型，规模是(1,3,224,224),gpu的tensor无法转换成PIL，所以在转换之后再放到gpu上
    label1 = label1.to(device)
    out = model(img1)
    _, pred = out.max(1)  # pred是类别数，tensor类型
    print(count)
    #print(path1[0])
    #print(type(path1[0]))

    if pred == label1:
        #将分对的图像放在right文件夹里面
        img_path = '/home/momo/sun.zheng/Recognizing_Image_Style/test_result_to_other/' + str(label1[0]) + '/right/'
        folder = os.path.exists(img_path)
        if not folder:
            os.makedirs(img_path)
        shutil.copy(path1[0], img_path)
    else:
        #将错的图像放在wrong文件夹里面，且精确到哪个类别
        img_path = '/home/momo/sun.zheng/Recognizing_Image_Style/test_result_to_other/' + str(label1[0]) + '/wrong/' + str(pred[0]) + '/'
        folder = os.path.exists(img_path)
        if not folder:
            os.makedirs(img_path)
        shutil.copy(path1[0], img_path)
```



2.批量更改文件夹名称，保存的类别文件夹名称都是tensor类型的，需要更改为风格化名称：(在每个类别文件夹里面运行下面的shell脚本)

Change_file_name.sh：

```shell
mv tensor\(0\,\ device\=\'cuda\:0\'\)/ Detailed;
mv tensor\(1\,\ device\=\'cuda\:0\'\)/ Pastel;
mv tensor\(2\,\ device\=\'cuda\:0\'\)/ Hazy;
mv tensor\(3\,\ device\=\'cuda\:0\'\)/ Bokeh;
mv tensor\(4\,\ device\=\'cuda\:0\'\)/ Serene;
mv tensor\(5\,\ device\=\'cuda\:0\'\)/ Texture;
mv tensor\(6\,\ device\=\'cuda\:0\'\)/ Ethereal;
mv tensor\(7\,\ device\=\'cuda\:0\'\)/ Macro;
mv tensor\(8\,\ device\=\'cuda\:0\'\)/ Depth_of_Field;
mv tensor\(9\,\ device\=\'cuda\:0\'\)/ Geometirc_Composition;
mv tensor\(10\,\ device\=\'cuda\:0\'\)/ Minimal;
mv tensor\(11\,\ device\=\'cuda\:0\'\)/ Romantic;
mv tensor\(12\,\ device\=\'cuda\:0\'\)/ Melancholy;
mv tensor\(13\,\ device\=\'cuda\:0\'\)/ Noir;
mv tensor\(14\,\ device\=\'cuda\:0\'\)/ HDR;
mv tensor\(15\,\ device\=\'cuda\:0\'\)/ Vintage;
mv tensor\(16\,\ device\=\'cuda\:0\'\)/ Long_Exposure;
mv tensor\(17\,\ device\=\'cuda\:0\'\)/ Horror;
mv tensor\(18\,\ device\=\'cuda\:0\'\)/ Sunny;
mv tensor\(19\,\ device\=\'cuda\:0\'\)/ Bright;
```



3.计算各个类别误分为其他类别的个数

```python
#此脚本用于分析测试集当中不同类别里面的错分样本，错分为哪些类别，且错分量为多少
#在划分好的测试数据集下（test_result_to_other）运行该脚本

import os

classlist = os.listdir('./test_result_to_other/')
num_class = len(classlist)
for i in range(num_class):
    category = classlist[i]
    print('large category:' + category)
    error_path = './test_result_to_other/' + category +'/wrong/'
    category_error_list = os.listdir(error_path)
    num_error_class = len(category_error_list)
    for j in range(num_error_class):
        category1 = category_error_list[j]
        error_category_path = './test_result_to_other/' + category + '/wrong/' + category1 + '/'
        image_list = os.listdir(error_category_path)
        error_num = len(image_list)
        print('category:' + category1 + '   num:' + str(error_num))
```



结果输出：

```python
large category:Bright
category:Bokeh   num:46
category:Noir   num:4
category:Melancholy   num:9
category:Sunny   num:48
category:Detailed   num:60
category:Ethereal   num:10
category:HDR   num:44
category:Vintage   num:13
category:Depth_of_Field   num:14
category:Minimal   num:24
category:Horror   num:17
category:Pastel   num:18
category:Geometirc_Composition   num:31
category:Macro   num:41
category:Hazy   num:7
category:Long_Exposure   num:25
category:Texture   num:20
category:Serene   num:35
category:Romantic   num:13
large category:Bokeh
category:Bright   num:24
category:Noir   num:22
category:Melancholy   num:21
category:Sunny   num:4
category:Detailed   num:27
category:Ethereal   num:10
category:HDR   num:8
category:Vintage   num:25
category:Depth_of_Field   num:70
category:Minimal   num:8
category:Horror   num:8
category:Pastel   num:34
category:Geometirc_Composition   num:4
category:Macro   num:78
category:Hazy   num:2
category:Long_Exposure   num:3
category:Texture   num:12
category:Serene   num:5
category:Romantic   num:20
large category:Noir
category:Bright   num:5
category:Bokeh   num:6
category:Melancholy   num:28
category:Detailed   num:5
category:Ethereal   num:23
category:HDR   num:6
category:Vintage   num:12
category:Depth_of_Field   num:13
category:Minimal   num:12
category:Horror   num:82
category:Pastel   num:1
category:Geometirc_Composition   num:23
category:Macro   num:4
category:Hazy   num:19
category:Long_Exposure   num:6
category:Texture   num:18
category:Serene   num:1
category:Romantic   num:6
large category:Melancholy
category:Bright   num:10
category:Bokeh   num:21
category:Noir   num:62
category:Sunny   num:10
category:Detailed   num:7
category:Ethereal   num:71
category:HDR   num:14
category:Vintage   num:41
category:Depth_of_Field   num:18
category:Minimal   num:19
category:Horror   num:35
category:Pastel   num:23
category:Geometirc_Composition   num:13
category:Hazy   num:34
category:Long_Exposure   num:10
category:Texture   num:36
category:Serene   num:15
category:Romantic   num:19
large category:Sunny
category:Bright   num:8
category:Bokeh   num:7
category:Noir   num:13
category:Melancholy   num:6
category:Detailed   num:5
category:Ethereal   num:8
category:HDR   num:27
category:Vintage   num:2
category:Depth_of_Field   num:7
category:Minimal   num:15
category:Horror   num:2
category:Pastel   num:4
category:Geometirc_Composition   num:11
category:Macro   num:2
category:Hazy   num:44
category:Long_Exposure   num:31
category:Texture   num:4
category:Serene   num:42
category:Romantic   num:11
large category:Detailed
category:Bright   num:36
category:Bokeh   num:36
category:Noir   num:5
category:Melancholy   num:1
category:Sunny   num:19
category:Ethereal   num:5
category:HDR   num:20
category:Vintage   num:15
category:Depth_of_Field   num:15
category:Minimal   num:12
category:Horror   num:21
category:Pastel   num:4
category:Geometirc_Composition   num:38
category:Macro   num:33
category:Hazy   num:9
category:Long_Exposure   num:17
category:Texture   num:28
category:Serene   num:31
category:Romantic   num:14
large category:Ethereal
category:Bright   num:8
category:Bokeh   num:13
category:Noir   num:21
category:Melancholy   num:40
category:Sunny   num:13
category:Detailed   num:3
category:HDR   num:2
category:Vintage   num:24
category:Depth_of_Field   num:2
category:Minimal   num:14
category:Horror   num:29
category:Pastel   num:23
category:Geometirc_Composition   num:3
category:Macro   num:7
category:Hazy   num:49
category:Long_Exposure   num:7
category:Texture   num:13
category:Serene   num:9
category:Romantic   num:14
large category:HDR
category:Bright   num:25
category:Bokeh   num:7
category:Noir   num:5
category:Melancholy   num:5
category:Sunny   num:20
category:Detailed   num:13
category:Ethereal   num:6
category:Vintage   num:7
category:Depth_of_Field   num:14
category:Minimal   num:4
category:Horror   num:18
category:Pastel   num:1
category:Geometirc_Composition   num:19
category:Macro   num:3
category:Hazy   num:12
category:Long_Exposure   num:42
category:Texture   num:15
category:Serene   num:52
category:Romantic   num:7
large category:Vintage
category:Bright   num:20
category:Bokeh   num:30
category:Noir   num:19
category:Melancholy   num:42
category:Sunny   num:3
category:Detailed   num:23
category:Ethereal   num:27
category:HDR   num:7
category:Depth_of_Field   num:21
category:Minimal   num:10
category:Horror   num:16
category:Pastel   num:71
category:Geometirc_Composition   num:17
category:Macro   num:4
category:Hazy   num:14
category:Long_Exposure   num:3
category:Texture   num:7
category:Serene   num:9
category:Romantic   num:55
large category:Depth_of_Field
category:Bright   num:30
category:Bokeh   num:155
category:Noir   num:22
category:Melancholy   num:26
category:Sunny   num:6
category:Detailed   num:24
category:Ethereal   num:13
category:HDR   num:18
category:Vintage   num:38
category:Minimal   num:17
category:Horror   num:20
category:Pastel   num:24
category:Geometirc_Composition   num:13
category:Macro   num:57
category:Hazy   num:12
category:Long_Exposure   num:11
category:Texture   num:25
category:Serene   num:26
category:Romantic   num:21
large category:Minimal
category:Bright   num:20
category:Bokeh   num:10
category:Noir   num:15
category:Melancholy   num:6
category:Sunny   num:18
category:Detailed   num:2
category:Ethereal   num:5
category:Vintage   num:1
category:Depth_of_Field   num:1
category:Horror   num:5
category:Pastel   num:2
category:Geometirc_Composition   num:60
category:Macro   num:26
category:Hazy   num:27
category:Long_Exposure   num:13
category:Texture   num:37
category:Serene   num:8
category:Romantic   num:1
large category:Horror
category:Bright   num:17
category:Bokeh   num:4
category:Noir   num:108
category:Melancholy   num:26
category:Sunny   num:2
category:Detailed   num:12
category:Ethereal   num:26
category:HDR   num:23
category:Vintage   num:13
category:Depth_of_Field   num:7
category:Minimal   num:6
category:Geometirc_Composition   num:9
category:Macro   num:7
category:Hazy   num:5
category:Long_Exposure   num:6
category:Texture   num:26
category:Serene   num:5
category:Romantic   num:12
large category:Pastel
category:Bright   num:16
category:Bokeh   num:60
category:Noir   num:6
category:Melancholy   num:28
category:Sunny   num:9
category:Detailed   num:10
category:Ethereal   num:35
category:HDR   num:6
category:Vintage   num:84
category:Depth_of_Field   num:14
category:Minimal   num:21
category:Horror   num:8
category:Geometirc_Composition   num:15
category:Macro   num:13
category:Hazy   num:18
category:Long_Exposure   num:6
category:Texture   num:3
category:Serene   num:10
category:Romantic   num:55
large category:Geometirc_Composition
category:Bright   num:37
category:Bokeh   num:5
category:Noir   num:35
category:Melancholy   num:7
category:Sunny   num:6
category:Detailed   num:17
category:Ethereal   num:7
category:HDR   num:31
category:Vintage   num:5
category:Depth_of_Field   num:10
category:Minimal   num:75
category:Horror   num:6
category:Pastel   num:1
category:Macro   num:11
category:Hazy   num:8
category:Long_Exposure   num:19
category:Texture   num:38
category:Serene   num:10
category:Romantic   num:6
large category:Macro
category:Bright   num:15
category:Bokeh   num:46
category:Noir   num:2
category:Sunny   num:1
category:Detailed   num:13
category:Ethereal   num:5
category:HDR   num:1
category:Vintage   num:9
category:Depth_of_Field   num:8
category:Minimal   num:19
category:Horror   num:5
category:Pastel   num:5
category:Geometirc_Composition   num:5
category:Hazy   num:1
category:Long_Exposure   num:3
category:Texture   num:31
category:Romantic   num:1
large category:Hazy
category:Bright   num:5
category:Bokeh   num:11
category:Noir   num:16
category:Melancholy   num:18
category:Sunny   num:35
category:Detailed   num:2
category:Ethereal   num:29
category:HDR   num:12
category:Vintage   num:12
category:Depth_of_Field   num:7
category:Minimal   num:18
category:Horror   num:3
category:Pastel   num:7
category:Geometirc_Composition   num:2
category:Macro   num:3
category:Long_Exposure   num:27
category:Texture   num:3
category:Serene   num:44
category:Romantic   num:7
large category:Long_Exposure
category:Bright   num:23
category:Bokeh   num:2
category:Noir   num:14
category:Melancholy   num:3
category:Sunny   num:28
category:Detailed   num:4
category:Ethereal   num:5
category:HDR   num:39
category:Vintage   num:4
category:Depth_of_Field   num:2
category:Minimal   num:11
category:Horror   num:15
category:Geometirc_Composition   num:23
category:Macro   num:4
category:Hazy   num:16
category:Texture   num:4
category:Serene   num:28
category:Romantic   num:2
large category:Texture
category:Bright   num:31
category:Bokeh   num:26
category:Noir   num:22
category:Melancholy   num:16
category:Sunny   num:12
category:Detailed   num:33
category:Ethereal   num:33
category:HDR   num:31
category:Vintage   num:18
category:Depth_of_Field   num:17
category:Minimal   num:64
category:Horror   num:21
category:Pastel   num:3
category:Geometirc_Composition   num:46
category:Macro   num:38
category:Hazy   num:6
category:Long_Exposure   num:5
category:Serene   num:24
category:Romantic   num:12
large category:Serene
category:Bright   num:22
category:Bokeh   num:36
category:Noir   num:13
category:Melancholy   num:8
category:Sunny   num:66
category:Detailed   num:33
category:Ethereal   num:13
category:HDR   num:49
category:Vintage   num:13
category:Depth_of_Field   num:11
category:Minimal   num:39
category:Horror   num:11
category:Pastel   num:13
category:Geometirc_Composition   num:14
category:Macro   num:21
category:Hazy   num:51
category:Long_Exposure   num:50
category:Texture   num:22
category:Romantic   num:11
large category:Romantic
category:Bright   num:18
category:Bokeh   num:31
category:Noir   num:17
category:Melancholy   num:30
category:Sunny   num:28
category:Detailed   num:27
category:Ethereal   num:26
category:HDR   num:20
category:Vintage   num:76
category:Depth_of_Field   num:12
category:Minimal   num:12
category:Horror   num:20
category:Pastel   num:60
category:Geometirc_Composition   num:13
category:Macro   num:6
category:Hazy   num:16
category:Long_Exposure   num:13
category:Texture   num:6
category:Serene   num:30
```



错分率从高到低的类别依次是：17（Depth of Field，wrong:558），3（Melancholy，wrong:458），20（Romantic，wrong:461），10（Bright，wrong:479），2（Pastel，wrong:417），13（Serene，wrong:496），14（Texture，wrong:458），1（Detailed，wrong:359），6（Vintage，wrong:398），12（Bokeh，wrong:385），18（Geometric Composition，wrong:334），15（Ethereal，wrong:294），8（Horror，wrong:314），5（HDR，wrong:275），19（Minimal，wrong:257），11（Hazy，wrong:261），4（Noir，wrong:270），9（Sunny，wrong:249），7（Long Exposure，wrong:227），16（Macro，wrong:170）

表格统计：

![统计分析结果表格](/Users/momo/Documents/Recognizing-Image-Style/统计分析结果表格.png)



将错分率高于平均值的类别分别进行分析：（红色是当前行类别错分率最高的类别，黄色是当前行类别错分率次高的类别）

1.Depth_of_Field,错分样本中，错分为Bokeh和Macro占比最多，分别为27.78%和10.22%；

2.Melancholy,错分样本中，错分为Ethereal和Noir占比最多，分别为15.5%和13.54%；

3.Romantic,错分样本中，错分为Vintage和Pastel占比最多，分别为16.49%和13.02%；

4.Bright,错分样本中，错分为Detailed和Sunny占比最多，分别为12.53%和10.02%；

5.Pastel,错分样本中，错分为Vintage和Bokeh占比最多，分别为20.14%和14.39%；再其次是Romantic，13.19%；

6.Serene,错分样本中，错分为Sunny和Hazy占比最多，分别为13.31%和10.28%；

7.Texsure,错分样本中，错分为Minimal和Geometric_Composition占比最多，分别为13.97%和10.04%；

8.Detailed,错分样本中，错分为Geometric_Comosition占比，Bright和Bokeh占比最多，分别为10.58%，10.03%，10.03%；

9.Vintage,错分样本中，错分为Pastel和Romantic占比最多，分别为17.84%和13.82%；

10.Bokeh,错分样本中，错分为Macro和Depth_of_Field占比最多，分别为20.26%和18.18%；

11.Geometric_Composition,错分样本中，错分为Minimal和Texture占比最多，分别为22.46%和11.38%；

12.Ethereal,错分样本中，错分为Hazy和Melancholy占比最多，分别为16.67%和13.61%；

13.Horror,错分样本中，错分为Horror最多，可以说远大于其他类，占比为34.39%；

14.HDR,错分样本中，错分为Senere和Long_exposure最多，分别为18.91%和15.27%；

15.Long_exposure，错分为HDR和Sunny以及Senere最多，分别为17.18%，12.33%和12.33%；



所以有以下结论：

1.Depth_of_Field和Bokeh类别风格相近；

2.Romantic,Vintage和Pastel类别风格相近；

3.Detailed和Bright类别风格相近；

4.Melancholy和Ethereal以及Noir类别风格相近，但是由于Noir和Horror非常类似，所以不考虑Noir；

5.Serene,Sunny和Hazy三个类别非常相似，每个类别都特别容易误分为其他两类；

6.Texsure,Minimal和Geometric_Composition三个类别非常相似，每个类别都特别容易误分为其他两类；

7.Noir和Horror都相互误分是最高的，看为非常相似；

8.Long_exposure和HDR相互错分较高，看为相似；

9.Macro是准确率是最高的一类，可以视为独立的；



**通过观察实际数据集的样本来对比相似类别：(这里只是找出不同类别中比较相似的，不代表所有样本都相似)**

1.Horror（恐怖的）和Noir（黑色的）

Noir:

![445639797_6323f7a39d](file:///Users/momo/Downloads/test_result_type_name/Noir/right/445639797_6323f7a39d.jpg?lastModify=1583669114)

Horror:

![3206946525_faf7dfae56](file:///Users/momo/Downloads/test_result_type_name/Horror/right/3206946525_faf7dfae56.jpg?lastModify=1583669114)



2.Texsure（纹理的）,Minimal和Geometric_Composition（几何构成的）

Texsure:

![img](file:///Users/momo/Downloads/test_result_type_name/Texture/right/11193491156_c5a521990e.jpg?lastModify=1583669114)

Minimal：

![9287599166_04825be17c](file:///Users/momo/Downloads/test_result_type_name/Minimal/right/9287599166_04825be17c.jpg?lastModify=1583669114)



Geometric_Composition:

![11652290563_5c55dd4e68](file:///Users/momo/Downloads/test_result_type_name/Geometric%20Composition/right/11652290563_5c55dd4e68.jpg?lastModify=1583669114)



3.Serene（宁静的）,Sunny（阳光的）和Hazy（舒适的）

Serene:

![9466469500_46fe48e783](file:///Users/momo/Downloads/test_result_type_name/Serene/right/9466469500_46fe48e783.jpg?lastModify=1583669114)



Sunny:

![10947913805_daac01a98e](file:///Users/momo/Downloads/test_result_type_name/Sunny/right/10947913805_daac01a98e.jpg?lastModify=1583669114)



Hazy:

![8609415175_183472a5a6](file:///Users/momo/Downloads/test_result_type_name/Hazy/right/8609415175_183472a5a6.jpg?lastModify=1583669114)



4.Melancholy（忧郁的）和Ethereal（飘渺的）

Melancholy:

![12118228314_3caf4fa52c](file:///Users/momo/Downloads/test_result_type_name/Melancholy/right/12118228314_3caf4fa52c.jpg?lastModify=1583669114)



Ethereal:

![8345739382_8a558f6cd8](file:///Users/momo/Downloads/test_result_type_name/Ethereal/right/8345739382_8a558f6cd8.jpg?lastModify=1583669114)



5.Detailed和Bright（明亮的）

Detailed:

![9205319480_7c0ba6c628](file:///Users/momo/Downloads/test_result_type_name/Detailed/right/9205319480_7c0ba6c628.jpg?lastModify=1583669114)



Bright:

![12514110653_9c2f02884a](file:///Users/momo/Downloads/test_result_type_name/Bright/right/12514110653_9c2f02884a.jpg?lastModify=1583669114)



6.Romantic（浪漫的）,Vintage（古老的）和Pastel（柔和的）

Romantic:

![8694528600_8be2bd4749](file:///Users/momo/Downloads/test_result_type_name/Romantic/right/8694528600_8be2bd4749.jpg?lastModify=1583669114)



Vintage:

![5085121061_34a448b9a6](file:///Users/momo/Downloads/test_result_type_name/Vintage/right/5085121061_34a448b9a6.jpg?lastModify=1583669114)



Pastel:

![8071705857_52355a11aa](file:///Users/momo/Downloads/test_result_type_name/Pastel/wrong/8071705857_52355a11aa.jpg?lastModify=1583669114)



7.Depth_of_Field（景深的）和Bokeh（散景的）

Depth_of_Field:

![12044705133_e0cca8498b](file:///Users/momo/Downloads/test_result_type_name/Depth%20of%20Field/right/12044705133_e0cca8498b.jpg?lastModify=1583669114)



Bokeh：

![12876968765_42b7000e63](file:///Users/momo/Downloads/test_result_type_name/Bokeh/right/12876968765_42b7000e63.jpg?lastModify=1583669114)



8.Long_exposure（长时间曝光的）和HDR

Long_exposure:

![12987943573_c3ebdcec07](file:///Users/momo/Downloads/test_result_type_name/Long%20Exposure/right/12987943573_c3ebdcec07.jpg?lastModify=1583669114)



HDR:

![12683548765_45329a5479](file:///Users/momo/Downloads/test_result_type_name/HDR/right/12683548765_45329a5479.jpg?lastModify=1583669114)



9.Macro（微距拍摄的）独自为一类：

![12398369464_13a9b360d6](file:///Users/momo/Downloads/test_result_type_name/Macro/right/12398369464_13a9b360d6.jpg?lastModify=1583669114)





综上，可以将某些类别归为一类来简化分类数，进而提高分类准确率，计划将类别划分为下：

1.Depth_of_Field和Bokeh归为一类；

2.Romantic,Vintage和Pastel归为一类；

3.Detailed和Bright归为一类；

4.Melancholy和Ethereal归为一类；

5.Serene,Sunny和Hazy归为一类；

6.Texsur,Minimal和Geometric_Composition归为一类；

7.Noir和Horror归为一类；

8.Long_exposure和HDR归为一类；

9.Macro独自为一类；

一共将20类简化为9类。



**其他解决方案**

**1.从数据集角度：除了上述分析方法中将相似的类别合并以外，对于某些类别界限不明显的，可以增加该类的数据量；**

**2.从模型角度：可以增加其它机制，比如attention机制，non-local机制等，也可以采用多模型集成，即多个模型做分类，最后采用一定的投票机制；**

**3.损失函数角度：更换不同的loss函数来满足模型，单个loss不适合可以使用多个loss函数进行评价训练。**



因为都很复杂，这个repository中就不再继续做了，只用ResNet-50模型来分类精度达到45%；然后对结果进行了简单分析，可以整合相似风格的类别。总体来说，这个repository只能用于图像风格识别的简单学习，离正常的业务使用还离得非常远！
