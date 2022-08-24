# -*- coding: utf-8 -*-            
# @Time : 2022/8/6 23:15
# @Author : Hcyand
# @FileName: deepFM.py
from layer.interaction import FMLayer, DNNLayer
from layer.inputs import EmbedLayer
from utils.dataset import create_criteo_dataset, features_dict
from utils.compile_fit import compile_fit

import tensorflow as tf
from keras import Model
from sklearn.metrics import accuracy_score


class DeepFM(Model):
    def __init__(self, feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super(DeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = EmbedLayer(self.sparse_feature_columns)
        self.fm = FMLayer(k, w_reg, v_reg)
        self.dnn = DNNLayer(hidden_units, output_dim, activation)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        sparse_embed = self.embed_layer(sparse_inputs)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        fm_output = self.fm(x)
        dnn_output = self.dnn(x)
        output = tf.nn.sigmoid(0.5 * (fm_output + dnn_output))
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    k = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('DeepFM', file, test_size=0.3)
    feature_dict = features_dict(file)

    model = DeepFM(feature_dict, k, w_reg, v_reg, hidden_units, output_dim, activation)
    model = compile_fit(model, X_train, y_train)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('AUC: ', accuracy_score(y_test, pre))
