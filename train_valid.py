import math
from multiprocessing import freeze_support

import EarlyStop
import torch
from torch import nn,optim
import numpy as np
from torchvision.models import resnet50
import dataloader,weather_model
import matplotlib
import  time as tm
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

batch_size=64
learning_rate=0.04
basepath='../data/train_dataset/'
epoch=1

#训练和验证放一起，每个epoch，验证一次。
#每个epoch记录一次loss，绘图，
#尝试使用早停法
def train(batch_size,lr,basepath,epoch,valid=True):
    #定义模型、参数、优化器、loss函数
    # 定义权重为可学习的权重
    w1 = torch.tensor(1.0, requires_grad=True)
    w2 = torch.tensor(1.0, requires_grad=True)
    model = weather_model.WeatherModelRes18DeepFc()
    criterion = nn.CrossEntropyLoss()
    # 将模型参数和权重参数放入优化器
    optimizer = optim.SGD([
        {'params': model.parameters(), },
        {'params': [w1, w2], }
    ], lr=lr)

    # 获取数据
    train_loader, \
    valid_loader,\
    train_set,\
    valid_set = dataloader.dataset_load(basepath=basepath, batch_size=batch_size)

    #初始化loss记录
    train_losses={
        'total':[],
        'weather':[],
        'time':[]
    }
    valid_losses={
        'total':[],
        'weather':[],
        'time':[]
    }
    #初始化早停工具类
    early_stop=EarlyStop.EarlyStopping(patience=10)
    #训练
    starttime = tm.time()
    for i in range(epoch):
        model.train()
        iteration=0
        for img, time, weather in train_loader:
            if torch.cuda.is_available():
                img = img.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
                weather = weather.cuda(non_blocking=True)
                model = model.cuda()
                criterion = criterion.cuda()
            pre_wea, pre_time = model(img)
            weather_loss = criterion(pre_wea, weather)
            time_loss = criterion(pre_time, time)

            optimizer.zero_grad()  # 清空梯度

            # 通过指数函数，保证w1和w2一直为正数。
            w1_pos = torch.exp(w1)
            w2_pos = torch.exp(w2)

            # loss加权和，权值为可学习参数，且加入倒数，防止权值太小。
            loss = w1_pos * weather_loss + w2_pos * time_loss + (1 / w1_pos) + (1 / w2_pos)  # 取两者loss之和，作为损失函数
            loss.backward()  # 损失函数对参数求偏导（反向传播
            optimizer.step()  # 更新参数
            if iteration%20==0:
                print("weather_loss:{0},time_loss:{1}\n".format(weather_loss, time_loss))
            iteration+=1
            #记录本次迭代的loss
            train_losses['total'].append(loss.item())
            train_losses['weather'].append(weather_loss.item())
            train_losses['time'].append(time_loss.item())

        if valid or (i+1)==epoch:
            #模型验证
            model.eval()
            wea_acc = 0
            time_acc = 0
            for img, time, weather in valid_loader:
                if torch.cuda.is_available():
                    img = img.cuda(non_blocking=True)
                    time = time.cuda(non_blocking=True)
                    weather = weather.cuda(non_blocking=True)
                    model = model.cuda()
                    criterion = criterion.cuda()
                pre_wea, pre_time = model(img)

                weather_loss = criterion(pre_wea, weather)
                time_loss = criterion(pre_time, time)
                # #取w1，w2数值（并非tensor）计算总loss
                # w1_pos = np.exp(w1.item())
                # w2_pos = np.exp(w2.item())
                # loss加权和，权值为可学习参数，且加入倒数，防止权值太小。
                loss =weather_loss +time_loss

                _, wea_idx = torch.max(pre_wea, 1)  # 统计每行最大值，获得下标index
                _, time_idx = torch.max(pre_time, 1)
                _, weather = torch.max(weather, 1)
                _, time = torch.max(time, 1)
                wea_acc += sum(weather == wea_idx)
                time_acc += sum(time == time_idx)
                # 记录本次迭代的loss
                valid_losses['total'].append(loss.item())
                valid_losses['weather'].append(weather_loss.item())
                valid_losses['time'].append(time_loss.item())

            # 注：len(dataLoader) dataloader的长度，是指，当前dataset，在指定的batchsize下，可被分成多少个batch，这里的长度的batch的数量。
            print("wea_acc={:6f},time_acc={:6f}".format(wea_acc / len(valid_set), time_acc / len(valid_set)))


        # print(train_losses)
        # print(valid_losses)
        train_epoch_loss=np.average(train_losses['total'][-(math.floor(len(train_set)/batch_size)+1):])
        valid_epoch_loss = np.average(valid_losses['total'][-(math.floor(len(valid_set) / batch_size)+1):])
        print(
            "epoch[{0}/{1}]----train_loss:[{2}]----valid_loss:[{3}]"
            .format(i,epoch,train_epoch_loss,valid_epoch_loss)
        )
        early_stop(valid_epoch_loss,model)
        if early_stop.early_stop:
            print('early stop in epoch:{}'.format(i))
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model.state_dict(),'model_'+str(valid_epoch_loss)+'.pth')
    endtime = tm.time()
    print('time elapse{0}'.format(endtime-starttime))
    # 绘制train loss 曲线
    plt.plot(train_losses['total'], label='weather_loss')
    plt.plot(train_losses['weather'], label='time_loss')
    plt.plot(train_losses['time'], label='loss')
    plt.xlabel('iteration')
    plt.ylabel('train_loss')
    plt.legend()
    plt.show()
    # 绘制train loss 曲线
    plt.plot(valid_losses['total'], label='weather_loss')
    plt.plot(valid_losses['weather'], label='time_loss')
    plt.plot(valid_losses['time'], label='loss')
    plt.xlabel('iteration')
    plt.ylabel('valid_loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    freeze_support()
    train(batch_size,learning_rate,basepath,epoch,valid=False)


