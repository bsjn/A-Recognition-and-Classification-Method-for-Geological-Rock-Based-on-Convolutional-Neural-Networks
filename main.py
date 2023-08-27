from data_processing import load_and_preprocess_data, get_r_channel_image
from model import ConvolutionLayer, FullyConnectedLayer, ActivationLayer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir = "F:\Rocks_0"  # 设置正确的路径

train_images, val_images, train_labels, val_labels = load_and_preprocess_data(data_dir)

r_channel_image = get_r_channel_image(train_images)

# Build the model
conv_layer = ConvolutionLayer(num_filters=32, kernel_size=3, input_channels=3)
fully_connected_layer = FullyConnectedLayer(num_units=128, num_classes=7)
activation_layer = ActivationLayer(activation_function='relu')

# Create a forward pass
inputs = tf.keras.Input(shape=(100, 100, 3))  # 定义输入形状
conv_output = conv_layer(inputs)  # 使用 inputs 作为输入
fc_output = fully_connected_layer(conv_output)
final_output = activation_layer(fc_output)

# Define the model
model = tf.keras.Model(inputs=inputs, outputs=final_output)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
epochs = 20  # 指定训练轮数
batch_size = 32  # 指定批次大小

#model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_labels))

# Evaluate the model
#test_loss, test_accuracy = model.evaluate(val_images, val_labels)
#print("Test accuracy:", test_accuracy)

###################################################
# Train the model and collect training history
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_labels))
test_loss, test_accuracy = model.evaluate(val_images, val_labels)
print("Test accuracy:", test_accuracy)

model.save('trained_model.h5')


# Extract training and validation metrics
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# ... 前面的代码 ...

# Define a function for calculating moving averages
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Apply moving average to the recorded metrics
window_size = 5  # You can adjust this window size
smooth_train_loss = moving_average(train_loss, window_size)
smooth_val_loss = moving_average(val_loss, window_size)
smooth_train_accuracy = moving_average(train_accuracy, window_size)
smooth_val_accuracy = moving_average(val_accuracy, window_size)

# Plot loss curves with smoothed data
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(window_size, epochs+1), smooth_train_loss, label='Smoothed Training Loss')
plt.plot(range(window_size, epochs+1), smooth_val_loss, label='Smoothed Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves with Smoothing')
plt.legend()

# Plot accuracy curves with smoothed data
plt.subplot(1, 2, 2)
plt.plot(range(window_size, epochs+1), smooth_train_accuracy, label='Smoothed Training Accuracy')
plt.plot(range(window_size, epochs+1), smooth_val_accuracy, label='Smoothed Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves with Smoothing')
plt.legend()

plt.tight_layout()
plt.show()
