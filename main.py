import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 图像文件夹路径
data_dir = "F:\Rocks"

# 加载图像数据和标签
images = []
labels = []

# 遍历每个类别文件夹
for label, folder_name in enumerate(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 读取图像并调整大小
        image = cv2.imread(file_path)
        image = cv2.resize(image, (100, 100))  # 调整为100x100像素
        images.append(image)
        labels.append(label)

# 将图像数据和标签转换为NumPy数组
images = np.array(images)
labels = np.array(labels)

# 归一化像素值到[0, 1]范围
images = images.astype('float32') / 255.0

# 划分训练集和验证集
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 进行数据增强（示例，根据需要可以增加更多变换）
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,      # 随机旋转角度范围
    width_shift_range=0.1,   # 随机水平平移范围
    height_shift_range=0.1,  # 随机垂直平移范围
    horizontal_flip=True,    # 水平翻转
    zoom_range=0.1           # 随机缩放范围
)

# 对训练集数据进行增强
datagen.fit(train_images)

r_channel_image_1 = images[0, :, :, 0]  # 读取第一张图片R通道的像素值（100*100）
print(r_channel_image_1)

# 此时，train_images 和 val_images 用于训练和验证 CNN 模型的输入，train_labels 和 val_labels 是相应的标签。


