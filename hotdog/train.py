import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import time
import sys
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '../data'

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            # X是一个tensor，传到cuda上去
            X = X.to(device)
            y = y.to(device)
            # 网络的前向传播
            y_hat = net(X)
            # 输出的y_hat和原来导入的y作为loss函数的输入就可以得到损失
            l = loss(y_hat, y)
            # 先将网络中的所有梯度置0
            optimizer.zero_grad()
            # 计算得到loss后就要回传损失。要注意的是这是在训练的时候才会有的操作，测试时候只有forward过程。
            l.backward()
            # 训练开始的时候需要先更新下学习率
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            print("y", train_acc_sum, y.shape[0])
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# 指定RGB三个通道的均值和⽅差来将图像通道归⼀化
# 功能：transforms.Normalize对数据按通道进行标准化，即先减均值，再除以标准差，注意是 chw
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
# transforms.Compose()类,这个类的主要作用是串联多个图片变换的操作。
train_augs = transforms.Compose([
     # 随机长宽比裁剪 transforms.RandomResizedCrop
     # size- 输出的分辨率
     transforms.RandomResizedCrop(size=224),
     # 依概率p水平翻转transforms.RandomHorizontalFlip
     # class torchvision.transforms.RandomHorizontalFlip(p=0.5)
     # 功能：依据概率p对PIL图片进行水平翻转
     # p- 概率，默认值为0.5
     transforms.RandomHorizontalFlip(),
     # 功能：将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
     # 注意事项：归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
     transforms.ToTensor(),
     normalize
 ])
test_augs = transforms.Compose([
     transforms.Resize(size=256),
     transforms.CenterCrop(size=224),
     transforms.ToTensor(),
     normalize
 ])

pretrained_net = models.resnet18(pretrained=True)

# 因为预训练网络一般是在1000类的ImageNet数据集上进行的，所以要迁移到你自己数据集的2分类，需要替换最后的全连接层为你所需要的输出。
# 因此下面这三行代码进行的就是用models模块导入resnet18网络，然后获取全连接层的输入channel个数，用这个channel个数和你要做的分类类别数（这里是2）替换原来模型中的全连接层。
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)

pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 2)
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
# 定义优化函数,随机梯度下降法(SGD)
optimizer = optim.SGD([{'params': feature_params}, {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}], lr=lr, weight_decay=0.001)

def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
  train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs), batch_size, shuffle=True)
  test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs), batch_size)
  # 该损失函数结合了nn.LogSoftmax()和nn.NLLLoss() 两个函数。它在做分类（具体几类）训练的时候是非常有用的。
  # 在训练过程中，对于每个类分配权值，可选的参数权值应该是一个1D张量。当你有一个不平衡的训练集时，这是是非常有用的。
  loss = torch.nn.CrossEntropyLoss()
  train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


train_fine_tuning(pretrained_net, optimizer)