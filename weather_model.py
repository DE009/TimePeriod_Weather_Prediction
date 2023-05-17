from torch import nn
from torchvision.models import resnet18
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