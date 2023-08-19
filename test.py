import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 读取example.png图像并调整大小
image_path = "example.png"  # 图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, (100, 100))  # 调整为100x100像素

# 将图像数据转换为NumPy数组并归一化
image = image.astype('float32') / 255.0

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# 添加batch维度并进行预处理
sample_image = image.reshape(1, 100, 100, 3)
sample_processed_image = datagen.flow(sample_image, batch_size=1)[0][0]

# 显示原始图像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

# 显示预处理后的图像
plt.subplot(1, 2, 2)
plt.imshow(sample_processed_image)
plt.title("Processed Image")

plt.tight_layout()
plt.show()
