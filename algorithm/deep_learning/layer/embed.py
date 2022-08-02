# -*- coding: utf-8 -*-            
# @Time : 2022/8/2 14:58
# @Author : Hcyand
# @FileName: embed.py
import tensorflow as tf
from keras.layers import Layer, Embedding, Dense
from keras import backend


class EmbedLayer(Layer):
    def __init__(self, k, sparse_feature_columns):
        super(EmbedLayer, self).__init__()
        # emb_layers，每个类别特征的embedding layers，以及转化为k维度embedding
        self.emb_layers = [Embedding(feat['feat_onehot_dim'], k) for feat in sparse_feature_columns]

    def call(self, inputs, **kwargs):
        if backend.ndim(inputs) != 2:
            raise ValueError('The dim of inputs should be 2, not %d' % (backend.ndim(inputs)))

        emb = tf.transpose(  # 置换a，根据perm重新排列尺寸，例如[1,2,3]维度经过下述转换后为[2,1,3]
            tf.convert_to_tensor(  # 转化为tensor格式
                [layer(inputs[:, i]) for i, layer in enumerate(self.emb_layers)]),
            perm=[1, 0, 2])
        emb = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))  # 转换为一维
        return emb


class ResLayer(Layer):
    """hidden layer"""
    def __init__(self, hidden_units):
        super(ResLayer, self).__init__()
        self.dense_layer = [Dense(i, activation='relu') for i in hidden_units]

    def build(self, input_shape):
        self.output_layer = Dense(input_shape[-1], activation=None)

    def call(self, inputs, **kwargs):
        if backend.ndim(inputs) != 2:
            raise ValueError('The dim of inputs shoulf be 2, not %d' % (backend.ndim(inputs)))

        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        x = self.output_layer(x)

        output = inputs + x
        return tf.nn.relu(output)
