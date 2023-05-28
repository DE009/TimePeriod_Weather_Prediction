import dataloader,weather_model
from torchvision import transforms
from torch.utils.data import DataLoader
import torch,os,json
import pandas as pd

import kmeans

batch_size=16
basepath= r"..\data\test_dataset/"

def model_deploy(test_loader,c="all"):
    # 获取onehot还原对应表
    train_loader, valid_loader = dataloader.dataset_load(basepath='../data/train_dataset/', batch_size=batch_size)
    weather_onehot = train_loader.dataset.weather
    period_onehot = train_loader.dataset.period
    #读取模型
    # model = weather_model.WeatherModelRes18DeepFc()
    model=weather_model.WeatherModelRes50DeepFc()
    #读取test数据
    test_dataset = test_loader.dataset
    test_data = test_dataset.labels.to_dict(orient='records')
    # 读入训练参数
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("model_"+str(c)+"_.pth"))
    else:
        model.load_state_dict(torch.load("model_"+str(c)+"_.pth", map_location=torch.device('cpu')))
    model.eval()

    # 模型在test集上做预测
    iteration = 0
    for img, time, weather in test_loader:
        if torch.cuda.is_available():
            img = img.cuda()
            model = model.cuda()
        pre_wea, pre_time = model(img)
        # pre_wea = model(img)
        _, wea_idx = torch.max(pre_wea, 1)  # 统计每行最大值，获得下标index
        _, time_idx = torch.max(pre_time, 1)
        # 获取类型str
        wea_str = [weather_onehot[int(x)] for x in wea_idx]
        time_str = [period_onehot[int(x)] for x in time_idx]
        # 将类别保存到结果数据文件中
        for i in range(batch_size):
            if (iteration*batch_size+i) >=len(test_dataset):
                break
            print(iteration * batch_size + i, iteration)
            test_data[iteration * batch_size + i]['period'] = time_str[i]
            test_data[iteration * batch_size + i]['weather'] = wea_str[i]
        iteration += 1
    print(test_data)
    return test_data

def deploy_kmeans():
    #调用kmeans类，读取测试集，并关闭训练。
    kmeans_cla = kmeans.kmeans(basepath=basepath, batch_size=batch_size, test=True,train=False)
    loaders = kmeans_cla.get_dataloader()
    del kmeans_cla
    result = {
        "annotations": []
    }
    for idx, loader in enumerate(loaders):
        label = model_deploy(test_loader=loader[0], c=idx)
        for i in label:
            result['annotations'].append(i)

    with open('./result.json', 'w', encoding='utf-8') as fp:
        json.dump(result, fp)
def deploy():
    result = {
        "annotations": []
    }

    # 遍历test数据集，生成初始数据
    test_labels=dataloader.test_labels_load(basepath)
    test_set=dataloader.WeatherData(labels=test_labels,basepath=basepath,)
    test_loader=DataLoader(test_set,batch_size=batch_size)
    label = model_deploy(test_loader=test_loader)
    for i in label:
        result['annotations'].append(i)
    with open('./result.json', 'w', encoding='utf-8') as fp:
        json.dump(result, fp)
if __name__ == '__main__':
    deploy()
    # deploy_kmeans()






