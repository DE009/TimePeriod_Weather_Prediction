import joblib
import numpy as np
from sklearn.cluster import KMeans

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import dataloader
import pandas as pd

#输入数据集，
#train，训练模型，返回聚类结果
#save，保存训练后的结果。

#聚类一次，返回给每个类两个dataloader。然后用不同的网络在上面做fiting。
#验证：不同网络独立验证。
#tain_valid 返回训练的loss或者acc，最后做两个模型的总和

#测试：先过kmeans，然后不同的dataloader放到不同的模型里，预测结果保存到同一个文件中。
class kmeans():
    def __init__(self,basepath,batch_size,test=False,k=2):
        self.batch_size=batch_size
        self.basepath=basepath
        if not test:
            self.all_labels= dataloader.trian_labels_load(train_basepath=basepath)
        else:
            self.all_labels=dataloader.test_labels_load(test_basepath=basepath)
        self.transform=transforms.Compose([
            transforms.Resize(size=(224, 224)),
        ])
        self.k=k
        self.is_test=test
    def images_load(self):
        all_images = []
        for idx,label in self.all_labels.iterrows():
            filepath = self.basepath + label['filename']
            img = Image.open(filepath.replace("\\", "//"))
            img=self.transform(img)
            img=np.array(img)
            img =img.reshape(-1, )
            all_images.append(img)
        return all_images
    def train_save(self):
        all_images=self.images_load()
        clt=KMeans(n_clusters=self.k)
        k_labels=clt.fit_predict(all_images)
        #保存模型
        joblib.dump(clt,'kmeans.pkl')
        return k_labels
    def load_predict(self,):
        test_images = self.images_load()
        clt=joblib.load('kmeans.pkl')
        k_labels=clt.predict(test_images)
        return k_labels

    def labels_class(self):
        if not self.is_test:
            k_labels=self.train_save()
        else:
            k_labels=self.load_predict()
        data=[]
        #按类别将labels分类
        for i in range(self.k):
            idx = np.where((k_labels == i) == 1)
            # 统计同一类内的labels，放入list
            tmp = [self.all_labels.iloc[x] for x in idx]
            # 将list中多个datarame组合成新的dataframe
            pd.concat(tmp)
            data.append(tmp[0])
        return  data
    def get_dataloader(self):
        data=self.labels_class()
        loaders=[]
        for labels in data:
            if not self.is_test:
                train_loader,valid_loader=dataloader.dataset_load(self.basepath,batch_size=self.batch_size,labels=labels)
                loaders.append([train_loader,valid_loader])
            else:
                test_dataset = dataloader.WeatherData(labels, basepath=self.basepath, train=False)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
                loaders.append([test_loader])
        return loaders
# test =kmeans(basepath='../data/train_dataset/',batch_size=64,)
# loader=test.get_dataloader()
# print(loader)
#
# test1 =kmeans(basepath=r"..\data\test_dataset/",batch_size=64,test=True)
# loader1=test1.get_dataloader()
# print(loader1)