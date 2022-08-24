# -*- coding: utf-8 -*-            
# @Time : 2022/8/3 21:00
# @Author : Hcyand
# @FileName: wideDeep.py
from layer.interaction import WideLayer, DNNLayer
from layer.inputs import EmbedLayer
from utils.dataset import create_criteo_dataset, features_dict
from utils.compile_fit import compile_fit
import tensorflow as tf
from keras import Model
from keras.layers import Embedding
from keras import optimizers, losses
from sklearn.metrics import accuracy_score


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation):
        super(WideDeep, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # self.embedding_layer = {'embed_layer' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
        #                         for i, feat in enumerate(self.sparse_feature_columns)}
        self.embed_layer = EmbedLayer(self.sparse_feature_columns)
        self.wide = WideLayer()
        self.deep = DNNLayer(hidden_units, output_dim, activation)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs, onehot_inputs = inputs[:, :13], inputs[:, 13:39], inputs[:, 39:]

        # Wide
        wide_input = tf.concat([dense_inputs, onehot_inputs], axis=1)
        wide_output = self.wide(wide_input)

        # Deep
        # sparse_embed = tf.concat([self.embedding_layer['embed_layer' + str(i)](sparse_inputs[:, i])
        #                           for i in range(sparse_inputs.shape[-1])], axis=-1)
        sparse_embed = self.embed_layer(sparse_inputs)
        deep_output = self.deep(sparse_embed)

        output = tf.nn.sigmoid(0.5 * (wide_output + deep_output))
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('WideDeep', file)
    feature_dict = features_dict(file)

    model = WideDeep(feature_dict, hidden_units, output_dim, activation)
    model = compile_fit(model, X_train, y_train)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("ACC: ", accuracy_score(y_test, pre))
