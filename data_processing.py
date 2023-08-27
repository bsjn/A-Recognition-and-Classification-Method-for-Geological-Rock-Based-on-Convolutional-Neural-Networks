import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, test_size=0.2, random_state=42):
    images = []
    labels = []

    for label, folder_name in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                image = cv2.imread(file_path)
                image = cv2.resize(image, (100, 100))
                images.append(image)
                labels.append(label)
            except cv2.error as e:
                print(f"处理图像 {file_path} 时出错：{str(e)}")

    images = np.array(images)
    labels = np.array(labels)
    images = images.astype('float32') / 255.0

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size, random_state=random_state)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(train_images)

    return train_images, val_images, train_labels, val_labels

def get_r_channel_image(images):
    r_channel_image = images[0, :, :, 0]
    return r_channel_image
