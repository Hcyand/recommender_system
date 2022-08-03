# -*- coding: utf-8 -*-            
# @Time : 2022/8/3 20:59
# @Author : Hcyand
# @FileName: wide_deep.py
import tensorflow as tf
from keras.layers import Layer, Dense
from keras.regularizers import l2


class WideLayer(Layer):
    def __init__(self):
        super(WideLayer, self).__init__()

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=l2(1e-4))

    def call(self, inputs, *args, **kwargs):
        x = self.w0 + tf.matmul(inputs, self.w)
        return x


class DeepLayer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super(DeepLayer, self).__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output
