
import numpy as np
from sklearn.cluster import KMeans

from PIL import Image
from torchvision import transforms

#输入数据集，
#train，训练模型，返回聚类结果
#save，保存训练后的结果。

#聚类一次，返回给每个类两个dataloader。然后用不同的网络在上面做fiting。
#验证：不同网络独立验证。
#tain_valid 返回训练的loss或者acc，最后做两个模型的总和

#测试：先过kmeans，然后不同的dataloader放到不同的模型里，预测结果保存到同一个文件中。
class kmeans():
    def __init__(self,train_set,valid_set,k=2):

