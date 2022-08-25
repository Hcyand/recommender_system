# -*- coding: utf-8 -*-            
# @Time : 2022/8/25 16:08
# @Author : Hcyand
# @FileName: deepCross.py
import pandas as pd

from layer.interaction import CrossLayer, DNNLayer
from layer.core import EmbedLayer
from utils.dataset import create_criteo_dataset, features_dict
from utils.compile_fit import compile_fit
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score


class DeepCross(Model):
    def __init__(self, feature_columns, hidden_units, layer_num, output_dim, activation):
        super(DeepCross, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = EmbedLayer(self.sparse_feature_columns)
        self.cross = CrossLayer(layer_num)
        self.deep = DNNLayer(hidden_units, output_dim, activation)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        embed = self.embed_layer(sparse_inputs)
        all_inputs = tf.concat([dense_inputs, embed], axis=1)

        # Cross
        cross_output = self.cross(all_inputs)
        # Deep
        deep_output = self.deep(all_inputs)

        output = tf.concat([cross_output, deep_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(output))
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    layer_num = 2

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('WideDeep', file)
    feature_dict = features_dict(file)

    model = DeepCross(feature_dict, hidden_units, output_dim, layer_num, activation)
    model = compile_fit(model, X_train, y_train)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("ACC: ", accuracy_score(y_test, pre))
