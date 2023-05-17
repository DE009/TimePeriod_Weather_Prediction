import torch
from torch import nn,optim
from torchvision.models import resnet18
from torchvision.models import resnet50
import dataloader

batch_size=64
learning_rate=0.04
basepath='../data/train_dataset/'
epoch=20

#加载预训练模型resnet18
#可通过.layer1[0].conv1的方式来手动修改某一层，将自己定义的网络赋值给该层即可。
#修改其中fc层。
class WeatherModel(nn.Module):
    def __init__(self):
        super(WeatherModel,self).__init__()
        res=resnet18()
        # res=resnet50()
        res.fc=nn.Identity()    #恒等函数，即该层fc不做变化，（等会可以输出网络结构看看）
        self.res=res
        self.weather=nn.Linear(512,3)   #['Afternoon', 'Dawn', 'Dusk', 'Morning']
        self.time=nn.Linear(512,4)  #['Cloudy', 'Rainy', 'Sunny']

    def forward(self,x):
        out=self.res(x)
        weather=self.weather(out)
        time=self.time(out)
        return weather,time

# 定义权重为可学习的权重
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
model=WeatherModel()
criterion=nn.CrossEntropyLoss()
#将模型参数和权重参数放入优化器
optimizer=optim.SGD([
    {'params':model.parameters(),},
    {'params': [w1,w2],}
],lr=learning_rate)


#获取数据
train_loader,valid_loader,train_set,valid_set=dataloader.dataset_load(basepath=basepath,batch_size=batch_size)
#训练
for i in range(epoch):
    iteration=0
    for img,time,weather in train_loader:
        if torch.cuda.is_available():
            img=img.cuda()
            time=time.cuda()
            weather=weather.cuda()
            model=model.cuda()
            criterion=criterion.cuda()
        pre_wea,pre_time=model(img)
        weather_loss=criterion(pre_wea,weather)
        time_loss=criterion(pre_time,time)
        if iteration %20==0:
            print("weather_loss:{0},time_loss:{1}\n".format(weather_loss,time_loss))
            # print("weather_loss:{0}\n".format(weather_loss))
        optimizer.zero_grad()   #清空梯度

#通过指数函数，保证w1和w2一直为正数。
        w1_pos=torch.exp(w1)
        w2_pos=torch.exp(w2)

        #loss加权和，权值为可学习参数，且加入倒数，防止权值太小。
        loss=w1_pos*weather_loss+w2_pos*time_loss +(1/w1_pos)+(1/w2_pos)     #取两者loss之和，作为损失函数

        loss.backward() #损失函数对参数求偏导（反向传播
        optimizer.step()    #更新参数
        iteration+=1
model.eval()
wea_acc=0
time_acc=0
for img,time,weather in valid_loader:
    if torch.cuda.is_available():
        img = img.cuda()
        time = time.cuda()
        weather = weather.cuda()
        model = model.cuda()
        criterion = criterion.cuda()
    pre_wea, pre_time = model(img)
    # pre_wea = model(img)
    _,wea_idx=torch.max(pre_wea,1)  #统计每行最大值，获得下标index
    _,time_idx=torch.max(pre_time,1)
    _, weather = torch.max(weather, 1)
    _, time = torch.max(time, 1)
    wea_acc += sum(weather == wea_idx)
    time_acc += sum(time == time_idx)
    #注：len(dataLoader) dataloader的长度，是指，当前dataset，在指定的batchsize下，可被分成多少个batch，这里的长度的batch的数量。
    print("wea_acc={:6f},time_acc={:6f}".format(wea_acc/len(valid_set),time_acc/len(valid_set)))
torch.save(model.state_dict(),"../model.pth")


