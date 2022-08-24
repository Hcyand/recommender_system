# -*- coding: utf-8 -*-            
# @Time : 2022/8/8 18:05
# @Author : Hcyand
# @FileName: din.py
import collections

import numpy as np

from layer.interaction import Attention, Dice
from utils.dataset import create_movies_dataset
from utils.compile_fit import compile_fit

import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, Dense, BatchNormalization, PReLU, Dropout
from sklearn.metrics import accuracy_score


class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40),
                 dnn_hidden_units=(256, 128, 64), att_attention='prelu',
                 dnn_activation='prelu', dnn_dropout=0.0):
        """
        :param feature_columns: feature columns dict
        :param behavior_feature_list: 行为特征的列表
        :param att_hidden_units: activation units 隐层单元数
        :param dnn_hidden_units: dnn层的隐层单元数
        :param att_attention: 隐层的激活函数
        :param dnn_activation: dnn的激活函数
        :param dnn_dropout: dropout参数
        """
        super(DIN, self).__init__()
        # dense feature & sparse feature
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.behavior_feature_list = behavior_feature_list

        # 除去behavior feature的数量
        self.other_sparse_num = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_num = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)

        # other sparse embedding
        self.embed_sparse_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        # activation layer
        self.att_layer = Attention(att_hidden_units, att_attention)
        self.bn_layer = BatchNormalization(trainable=True)
        self.dense_layer = [Dense(unit, activation=PReLU() if dnn_activation == 'prelu' else Dice())
                            for unit in dnn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        """
        dense_inputs: empty / (None,  dense_num)
        sparse_inputs: empty / (None, other_sparse_num)
        history_seq: (None, n, k); n=len(seq); k=embed_dim
        candidate_item: (None, k); k=embed_dim
        """
        dense_inputs = tf.transpose([inputs[feat['feat']] for feat in self.dense_feature_columns])
        sparse_inputs = tf.transpose([inputs[feat['feat']] for feat in self.sparse_feature_columns
                                      if feat['feat'] not in self.behavior_feature_list])
        history_seq = tf.transpose([inputs[feat['feat']] for feat in self.sparse_feature_columns
                                    if feat['feat'] in self.behavior_feature_list], [1, 2, 0])
        candidate_item = tf.transpose([inputs['movie_id']])

        # dense & sparse inputs embedding
        other_feat = tf.concat([layer(sparse_inputs[:, i]) for i, layer in enumerate(self.embed_sparse_layers)],
                               axis=-1)
        other_feat = tf.concat([other_feat, dense_inputs], axis=-1)

        # history_seq & candidate_item embedding
        # (None, n, k)
        seq_embed = tf.concat([layer(history_seq[:, :, i]) for i, layer in enumerate(self.embed_seq_layers)], axis=-1)
        # (None, k)
        item_embed = tf.concat([layer(candidate_item[:, i]) for i, layer in enumerate(self.embed_seq_layers)], axis=-1)

        # one_hot之后第一维是1的token，为填充的0
        mask = tf.cast(tf.not_equal(history_seq[:, :, 0], 0), dtype=tf.float32)
        att_emb = self.att_layer([item_embed, seq_embed, seq_embed, mask])

        # 若其它特征不为empty
        if self.dense_num > 0 or self.other_sparse_num > 0:
            emb = tf.concat([att_emb, item_embed, other_feat], axis=-1)
        else:
            emb = tf.concat([att_emb, item_embed], axis=-1)

        emb = self.bn_layer(emb)
        for layer in self.dense_layer:
            emb = layer(emb)

        emb = self.dropout(emb)
        output = self.out_layer(emb)
        return tf.nn.sigmoid(output)


if __name__ == '__main__':
    behavior_features = ['movies_seq']
    feature_dict, (X_train, y_train), (X_test, y_test) = create_movies_dataset(0.3, 10)

    features = X_train.columns
    X = {feat: list(X_train[feat].values) for feat in features}

    model = DIN(feature_dict, behavior_features)
    model = compile_fit(model, X, y_train)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('AUC: ', accuracy_score(y_test, pre))
