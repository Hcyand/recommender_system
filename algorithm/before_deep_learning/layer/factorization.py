# -*- coding: utf-8 -*-            
# @Time : 2022/8/1 16:11
# @Author : Hcyand
# @FileName: factorization.py
import tensorflow as tf
import keras.backend as K
from keras.regularizers import l2
from tensorflow.python.keras.layers import Layer


class FM_layer(Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM_layer, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=l2(self.v_reg))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError('Unexpected inputs dimensions %d, except to be 2 dimensions' % (K.ndim(inputs)))

        linear_part = self.w0 + tf.matmul(inputs, self.w)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)

        return linear_part + inter_part


class FFM_layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM_layer, self).__init__()  # 找到FFM_layer的父类Layer，然后把类FFM_layer的对象转换为类layer的对象
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) + len(
            self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v', shape=(self.feature_num, self.field_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]

        # one_hot encoding
        x = tf.cast(dense_inputs, dtype=tf.float32)  # cast：转换数据格式
        for i in range(sparse_inputs.shape[1]):
            # tf.one_hot中depth表示one hot维度的深度
            x = tf.concat([x, tf.one_hot(tf.cast(sparse_inputs[:, i], dtype=tf.int32),
                                         depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)
        linear_part = self.w0 + tf.matmul(x, self.w)
        inter_part = 0
        # 每维特征先跟自己的[field_num, k]相乘得到Vij*X
        field_f = tf.tensordot(x, self.v, axes=1)
        # 域之间两两相乘
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                # 按一定方式计算张量中元素之和
                inter_part += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]),
                    axis=1, keepdims=True
                )
        return linear_part + inter_part
