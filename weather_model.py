from torch import nn
from torchvision.models import resnet18,resnet50
#加载预训练模型resnet18
#可通过.layer1[0].conv1的方式来手动修改某一层，将自己定义的网络赋值给该层即可。
#修改其中fc层。
class WeatherModelRes18(nn.Module):
    def __init__(self):
        super(WeatherModelRes18,self).__init__()
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
class WeatherModelRes50DeepFc(nn.Module):
    def __init__(self):
        super(WeatherModelRes50DeepFc,self).__init__()
        res=resnet50()
        res.fc=nn.Identity()    #恒等函数，即该层fc不做变化，（等会可以输出网络结构看看）
        self.res=res
        self.weather=nn.Sequential(
            nn.Linear(2048,512),
            nn.Linear(512, 128),   #['Cloudy', 'Rainy', 'Sunny']
            nn.Linear(128,32),
            nn.Linear(32,3)
        )

        self.time=nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512,128),
            nn.Linear(128, 32),  # ['Afternoon', 'Dawn', 'Dusk', 'Morning']
            nn.Linear(32,4)
        )

    def forward(self,x):
        out=self.res(x)
        weather=self.weather(out)
        time=self.time(out)
        return weather,time
class WeatherModelRes18DeepFc(nn.Module):
    def __init__(self):
        super(WeatherModelRes18DeepFc,self).__init__()
        res=resnet18()
        # res=resnet50()
        res.fc=nn.Identity()    #恒等函数，即该层fc不做变化，（等会可以输出网络结构看看）
        self.res=res
        self.weather=nn.Sequential(
            nn.Linear(512,128),
            nn.Linear(128, 32),   #['Cloudy', 'Rainy', 'Sunny']
            nn.Linear(32,3)
        )

        self.time=nn.Sequential(
            nn.Linear(512,128),
            nn.Linear(128, 32),  # ['Afternoon', 'Dawn', 'Dusk', 'Morning']
            nn.Linear(32,4)
        )
        # self.weather = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Linear(256, 3), # ['Cloudy', 'Rainy', 'Sunny']
        # )
        #
        # self.time = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, 4),# ['Afternoon', 'Dawn', 'Dusk', 'Morning']
        # )
    def forward(self,x):
        out=self.res(x)
        weather=self.weather(out)
        time=self.time(out)
        return weather,time

class WeatherModelRes50(nn.Module):
    def __init__(self):
        super(WeatherModelRes50,self).__init__()
        # res=resnet18()
        res=resnet50()
        res.fc=nn.Identity()    #恒等函数，即该层fc不做变化，（等会可以输出网络结构看看）
        self.res=res
        self.weather=nn.Linear(2048,3)   #['Afternoon', 'Dawn', 'Dusk', 'Morning']
        self.time=nn.Linear(2048,4)  #['Cloudy', 'Rainy', 'Sunny']

    def forward(self,x):
        out=self.res(x)
        weather=self.weather(out)
        time=self.time(out)
        return weather,time