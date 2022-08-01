# -*- coding: utf-8 -*-            
# @Time : 2022/7/29 15:37
# @Author :  Hcyand
# @FileName: model.py

from layer import FFM_layer

import tensorflow as tf
from tensorflow.python.keras import Model


class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output
