# -*- coding: utf-8 -*-            
# @Time : 2022/8/6 23:40
# @Author : Hcyand
# @FileName: afm.py
from layer.interaction import AFMLayer
from keras.models import Model
import tensorflow as tf


class AFM(Model):
    def __init__(self, feature_columns, mode):
        super(AFM, self).__init__()
        self.afm_layer = AFMLayer(feature_columns, mode)

    def call(self, inputs, training=None, mask=None):
        output = self.afm_layer(inputs)
        output = tf.nn.sigmoid(output)
        return output
