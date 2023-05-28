import torch
from torchvision import  transforms
from torch.utils.data import  DataLoader,Dataset
import  json
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
import os
#超参数定义
# basepath='./data/train_dataset/'
# batch_size=1

#数据准备
class WeatherData(Dataset):
    def __init__(self,labels,basepath,train=False):
        super(WeatherData, self).__init__()
        self.labels=labels.reset_index(drop=True)  #重新设置index，全都从0开始
        print(self.labels)
        self.is_train=train
        # 记录one-hot编码和类别的转换
        self.period = pd.get_dummies(labels['period']).columns
        self.weather = pd.get_dummies(labels['weather']).columns
        # self.
        #定义数据预处理
            #修改为随机裁切
        self.train_transform=transforms.Compose([
            transforms.Resize(size=(340,340)),
            transforms.RandomCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),

        ])
        self.valid_transform=transforms.Compose([
            transforms.Resize(size=(340,340)),
            transforms.RandomCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.basepath=basepath
    #返回数据，（img,period,weather）,按输入的index，返回一个数据以及对应的label
    def __getitem__(self, idx):
        filepath=self.basepath+self.labels['filename'][idx]
        imgs=Image.open(filepath.replace("\\","//"))
        if self.is_train:
            imgs = self.train_transform(imgs)
        else:
            imgs=self.valid_transform(imgs)

        # get_dummies将str类别转换为one-hot编码，用loc单取某一行
        # print(pd.get_dummies(self.labels['period']).head(1),"\n",
        #       pd.get_dummies(self.labels['weather']).head(1))
        #同时转为tensor
        return imgs, \
               torch.tensor(pd.get_dummies(self.labels['period']).loc[idx],dtype=torch.float32), \
               torch.tensor(pd.get_dummies(self.labels['weather']).loc[idx],dtype=torch.float32)
    #返回数据集长度
    def __len__(self):
        return len(self.labels)
#读取所有的图像labels
def trian_labels_load(train_basepath):
    #数据读入
    with open(train_basepath+"train.json",'r') as f:
        data=json.load(f)
    labels=pd.DataFrame(data['annotations'])
    return labels
def test_labels_load(test_basepath):
    test_data = []
    for root, dirs, files in os.walk(test_basepath + r"\test_images"):
        for file in ["test_images\\" + x for x in files]:
            tmp = {
                "filename": file,
                "period": "",
                "weather": "",
            }
            test_data.append(tmp)
    test_data_pd = pd.DataFrame(test_data)
    return test_data_pd

def dataset_load(basepath,batch_size,labels=None):
    #若自定义labels，则使用。否则就读取所有labels
    if labels is None:
        labels=trian_labels_load(basepath)
    #随机打乱labels，保证切分的数据是随机，但不重叠的
    # labels=labels.sample(frac=1.0)
    # train_labels=labels.iloc[:int(0.8*len(labels))]
    # valid_labels = labels.iloc[-int(0.2 * len(labels)):]
    train_labels, valid_labels= train_test_split(labels, test_size=0.2)
    train_set=WeatherData(train_labels,basepath,train=True)
    valid_set=WeatherData(valid_labels,basepath,train=False)
    #生成训练和验证集，用random_split函数
    # train_set,valid_set=torch.utils.data.random_split(dataset,[int(0.8*len(dataset)),int(0.2*len(dataset))])
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=8,persistent_workers=True)
    valid_loader=DataLoader(valid_set,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=8,persistent_workers=True)
    return train_loader,valid_loader
