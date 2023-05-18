import dataloader

basepath='../data/train_dataset/'
train_loader,valid_loader,train_set,valid_set=dataloader.dataset_load(basepath=basepath,batch_size=2600)
data=next(iter(train_loader))[0]    #[0]只获取图像数据
#分别遍历每个图像不同通道的每个像素，计算不同通道的均值和方差
# #(batch_size, channels, height, width)
mean=data.mean(dim=[0,2,3])
std=data.std(dim=[0,2,3])
print(mean)
print(std)
data=next(iter(valid_loader))[0]    #[0]只获取图像数据
#分别遍历每个图像不同通道的每个像素，计算不同通道的均值和方差
# #(batch_size, channels, height, width)
mean=data.mean(dim=[0,2,3])
std=data.std(dim=[0,2,3])
print(mean)
print(std)