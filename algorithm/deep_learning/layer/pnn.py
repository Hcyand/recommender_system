# -*- coding: utf-8 -*-            
# @Time : 2022/8/3 10:51
# @Author : Hcyand
# @FileName: pnn.py
import tensorflow as tf
from keras.layers import Layer, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.regularizers import l2
import keras.backend as K


class DNN_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation='relu', dropout=0.2):
        super(DNN_layer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        self.dropout_layer = Dropout(dropout)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError('Unexpected inputs dimension %d, expect to be 2 dimensions' % (K.ndim(inputs)))

        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.dropout_layer(x)
        output = self.output_layer(x)
        return tf.nn.sigmoid(output)


class InnerProductLayer(Layer):
    def __init__(self):
        super(InnerProductLayer, self).__init__()

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError('Unexpected inputs dimension %d, expect to be 3 dimensions' % (K.ndim(inputs)))

        field_num = inputs.shape[1]
        row, col = [], []
        # 先将要相乘的emb找出来，存在两个矩阵中，然后进行点乘即可
        for i in range(field_num - 1):
            for j in range(i + 1, field_num):
                row.append(i)
                col.append(j)
        # tf.gather根据indices的参数值获取切片
        p = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), row), [1, 0, 2])
        q = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), col), [1, 0, 2])
        innerProduct = tf.reduce_sum(p * q, axis=-1)

        return innerProduct


class OuterProductLayer(Layer):
    def __init__(self):
        super(OuterProductLayer, self).__init__()

    def build(self, input_shape):
        self.field_num = input_shape[1]
        self.k = input_shape[2]
        self.pair_num = self.field_num * (self.field_num - 1) // 2

        # 每个外积矩阵对应一个w矩阵，共有pair个
        self.w = self.add_weight(name='W', shape=(self.k, self.pair_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(1e-4),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError('Unexpected inputs dimension %d, expect to be 3 dimensions' % (K.ndim(inputs)))

        row, col = [], []
        for i in range(self.field_num - 1):
            for j in range(i + 1, self.field_num):
                row.append(i)
                col.append(j)
        p = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), row), [1, 0, 2])  # [None, pair_num, k]
        q = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), col), [1, 0, 2])  # [None, pair_num, k]
        p = tf.expand_dims(p, axis=1)  # [None, 1, pair_num, k]

        tmp = tf.multiply(p, self.w)  # [None, 1, pair_num, k] * [k, pair, k] = [None, k, pair_num, k]
        tmp = tf.reduce_sum(tmp, axis=-1)  # [None, k, pair_num]
        tmp = tf.multiply(tf.transpose(tmp, [0, 2, 1]), q)  # [None, pair_num, k]
        outputProduct = tf.reduce_sum(tmp, axis=-1)  # [None, pair_num]
        return outputProduct


class FGCNN_layer(Layer):
    def __init__(self, filters=[14, 16], kernel_width=[7, 7], dnn_maps=[3, 3], pooling_width=[2, 2]):
        super(FGCNN_layer, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width
        self.dnn_maps = dnn_maps
        self.pooling_width = pooling_width

    def build(self, input_shape):
        n = input_shape[1]
        k = input_shape[-1]
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []
        for i in range(len(self.filters)):
            self.conv_layers.append(
                Conv2D(filters=self.filters[i],
                       kernel_size=(self.kernel_width[i], 1),
                       strides=(1, 1),
                       padding='same',
                       activation='tanh')
            )
            self.pool_layers.append(
                MaxPool2D(pool_size=(self.pooling_width[i], 1))
            )
        self.flatten_layer = Flatten()

    def call(self, inputs, **kwargs):
        k = inputs.shape[-1]
        dnn_output = []
        x = tf.expand_dims(inputs, axis=-1)
        for i in range(len(self.filters)):
            x = self.conv_layers[i](x)
            x = self.pool_layers[i](x)
            out = self.flatten_layer(x)
            out = Dense(self.dnn_maps[i] * x.shape[1] * x.shape[2], activation='relu')(out)
            out = tf.reshape(out, shape=(-1, out.shape[1] // k, k))
            dnn_output.append(out)

        output = tf.concat(dnn_output, axis=1)
        return output
