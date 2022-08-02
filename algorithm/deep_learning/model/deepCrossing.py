# -*- coding: utf-8 -*-            
# @Time : 2022/8/2 15:14
# @Author : Hcyand
# @FileName: deepCrossing.py
from ..layer.embed import EmbedLayer, ResLayer
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense


class DeepCrossing(Model):
    def __init__(self, feature_columns, k, hidden_units, res_layer_num):
        super(DeepCrossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = EmbedLayer(k, self.sparse_feature_columns)
        self.res_layer = [ResLayer(hidden_units) for _ in range(res_layer_num)]
        self.output_layer = Dense(1, activation='sifmoid')

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        emb = self.embed_layer(sparse_inputs)
        x = tf.concat([dense_inputs, emb], axis=-1)
        for layer in self.res_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output
