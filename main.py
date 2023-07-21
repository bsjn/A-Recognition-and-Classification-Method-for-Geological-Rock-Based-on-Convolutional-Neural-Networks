import numpy as np
import tensorflow as tf

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data,self.train_label),(self.test_data,self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道RGB，如果没有这个维度就是灰度的图片，没有彩色。
        self.train_data = np.expand_dims(self.train_data.astype(np.float)/255.0,axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]   #60000,10000

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)  #可以重复取某条数据
        return self.train_data[index, :], self.train_label[index]

mnist = MNISTLoader()
batch_size = 1
train_data, train_label = mnist.get_batch(batch_size)
print(train_data*255)
print(train_label)
print(train_data[0, :, 1])

