import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取example.png图像并调整大小
image_path = "example.png"  # 图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, (100, 100))  # 调整为100x100像素

# 将图像数据转换为NumPy数组并归一化
image = image.astype('float32') / 255.0

# 添加batch维度并进行预处理
sample_image = image.reshape(1, 100, 100, 3)

# 定义一个卷积层
conv_layer = tf.keras.layers.Conv2D(
    filters=16,  # 卷积核数
    kernel_size=(3, 3),  # 卷积核大小
    strides=(1, 1),
    padding='valid',
    activation='sigmoid'
)

# 对图像进行卷积操作
convolved_image = conv_layer(sample_image)

# 获取卷积层的权重（卷积核）
conv_kernel = conv_layer.weights[0].numpy()

# 提取第一个卷积核的特征图
feature_map = convolved_image[0, :, :, 0]

# 绘制原始图像
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")

# 绘制第一个卷积核
plt.subplot(1, 3, 2)
plt.imshow(conv_kernel[:, :, 0, 0], cmap='gray')
plt.title("Convolution Kernel")

# 绘制第一个特征图
plt.subplot(1, 3, 3)
plt.imshow(feature_map, cmap='gray')
plt.title("Feature Map")

plt.tight_layout()
plt.show()
