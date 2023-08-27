import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D

class ConvolutionLayer(tf.keras.layers.Layer):      # 卷积层

    def __init__(self, num_filters, kernel_size, input_channels):   # 参数：卷积核数，卷积核大小，输入通道数
        super(ConvolutionLayer, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        self.conv_layer1 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            activation='relu',
            input_shape=(None, None, input_channels)
        )
        self.pooling_layer1 = MaxPooling2D(pool_size=(2, 2))  # 添加第一个池化层，池化窗口大小为2x2

        self.conv_layer2 = tf.keras.layers.Conv2D(
            filters=num_filters*2,  # 增加卷积核数
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            activation='relu'
        )
        self.pooling_layer2 = MaxPooling2D(pool_size=(2, 2))  # 添加第二个池化层，池化窗口大小为2x2

    def call(self, inputs):
        conv_output1 = self.conv_layer1(inputs)
        pooled_output1 = self.pooling_layer1(conv_output1)
        conv_output2 = self.conv_layer2(pooled_output1)
        pooled_output2 = self.pooling_layer2(conv_output2)
        return pooled_output2



class FullyConnectedLayer(tf.keras.layers.Layer):      # 全连接层

    def __init__(self, num_units, num_classes):   # 参数：神经元数量，类别数量
        super(FullyConnectedLayer, self).__init__()
        self.num_units = num_units
        self.num_classes = num_classes

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(units=num_units, activation='sigmoid')
        self.dense_layer2 = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        flattened_inputs = self.flatten_layer(inputs)
        dense_output1 = self.dense_layer1(flattened_inputs)
        output = self.dense_layer2(dense_output1)
        return output



class ActivationLayer(tf.keras.layers.Layer):      # 激活函数层

    def __init__(self, activation_function='relu'):   # 参数：激活函数名称（默认为ReLU）
        super(ActivationLayer, self).__init__()
        self.activation_function = activation_function

        self.activation_layer = tf.keras.layers.Activation(activation_function)

    def call(self, inputs):
        output = self.activation_layer(inputs)
        return output







