import dataloader,weather_model
from torchvision import transforms
from torch.utils.data import DataLoader
import torch,os,json
import pandas as pd
batch_size=4
basepath=r"..\data\test_dataset/"
#获取onehot还原对应表
train_loader,valid_loader,train_set,valid_set=dataloader.dataset_load(basepath='../data/train_dataset/',batch_size=batch_size)
weather_onehot=train_set.weather
period_onehot=train_set.period

#遍历test数据集，生成初始数据
test_data=[]
for root,dirs,files in os.walk(basepath+r"\test_images"):
    for file in ["test_images\\" + x for x in files]:
        tmp={
            "filename": file,
            "period": "",
            "weather": "",
        }
        test_data.append(tmp)
test_data_pd=pd.DataFrame(test_data)
test_dataset=dataloader.WeatherData(test_data_pd, basepath=basepath, train=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size)

model=weather_model.WeatherModelRes18DeepFc()

#读入训练参数
if torch.cuda.is_available():
    model.load_state_dict(torch.load("../model.pth"))
else:
    model.load_state_dict(torch.load("../model.pth", map_location=torch.device('cpu')))
model.eval()
#模型在test集上做预测
iteration=0
for img,time,weather in test_loader:
    if torch.cuda.is_available():
        img = img.cuda()
        time = time.cuda()
        weather = weather.cuda()
        model = model.cuda()
    pre_wea, pre_time = model(img)
    # pre_wea = model(img)
    _,wea_idx=torch.max(pre_wea,1)  #统计每行最大值，获得下标index
    _,time_idx=torch.max(pre_time,1)
    #获取类型str
    wea_str=[weather_onehot[int(x)] for x in wea_idx]
    time_str=[period_onehot[int(x)] for x in time_idx]
    #将类别保存到结果数据文件中
    for i in range(batch_size):
        print(iteration*batch_size+i,iteration)
        test_data[iteration * batch_size + i]['period']=time_str[i]
        test_data[iteration * batch_size + i]['weather']=wea_str[i]
    iteration+=1
print(test_data)
result={
    "annotations":test_data
}
print(result)
with open('./result.json', 'w',encoding='utf-8') as fp:
    json.dump(result,fp)


