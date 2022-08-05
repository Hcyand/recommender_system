# -*- coding: utf-8 -*-            
# @Time : 2022/8/4 15:50
# @Author : Hcyand
# @FileName: dcn.py
import tensorflow as tf
from keras.layers import Layer, Input, Dense
from keras.regularizers import l2


class DenseLayer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super(DenseLayer, self).__init__()
        self.hidden_layer = [Dense(x, activation=activation) for x in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)

    def build(self, input_shape):
        pass

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


class CrossLayer(Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super(CrossLayer, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)
        ]

        self.cross_bias = [
            self.add_weight(name='b' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)
        ]

    def call(self, inputs, *args, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)
        x1 = x0
        for i in range(self.layer_num):
            x1_w = tf.matmul(tf.transpose(x1, [0, 2, 1]), self.cross_weight[i])
            x1 = tf.matmul(x0, x1_w) + self.cross_bias[i] + x1

        output = tf.squeeze(x1, axis=2)
        return output
