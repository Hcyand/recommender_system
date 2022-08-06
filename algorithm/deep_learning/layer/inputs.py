# -*- coding: utf-8 -*-            
# @Time : 2022/8/5 17:11
# @Author : Hcyand
# @FileName: inputs.py
import tensorflow as tf
from keras.layers import Layer, Embedding


class EmbedLayer(Layer):
    def __init__(self, sparse_feature_columns, k=8):
        super(EmbedLayer, self).__init__()
        # emb_layers，每个类别特征的embedding layers，以及转化为k维度embedding
        self.emb_layers = [Embedding(feat['feat_onehot_dim'], k) for feat in sparse_feature_columns]

    def call(self, inputs, *args, **kwargs):
        # 置换a，根据perm重新排列尺寸，例如[1,2,3]维度经过下述转换后为[2,1,3]
        emb = tf.transpose(
            tf.convert_to_tensor(  # 转化为tensor格式
                [layer(inputs[:, i]) for i, layer in enumerate(self.emb_layers)]),
            perm=[1, 0, 2])
        emb = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))  # 转换为一维
        return emb
