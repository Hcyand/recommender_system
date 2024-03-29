# -*- coding: utf-8 -*-            
# @Time : 2022/8/13 17:13
# @Author : Hcyand
# @FileName: utils.py
import tensorflow as tf
import numpy as np
import heapq
from tensorflow.python.keras.layers import Flatten, Concatenate, Layer, Add
from tensorflow.python.ops.lookup_ops import TextFileInitializer, StaticHashTable
from tensorflow.python.keras.layers import GRUCell
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.backend import zeros_like, expand_dims


class Hash(Layer):
    def __init__(self, num_buckets, mask_zero=False, vocabulary_path=None, default_value=0, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        self.vocabulary_path = vocabulary_path
        self.default_value = default_value
        if self.vocabulary_path:
            initializer = TextFileInitializer(vocabulary_path, 'string', 1, 'int64', 0, delimiter=',')
            self.hash_table = StaticHashTable(initializer, default_value=self.default_value)
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        if self.vocabulary_path:
            hash_x = self.hash_table.lookup(x)
            return hash_x

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets, name=None)
        except AttributeError:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None)

        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, 'vocabulary_apth': self.vocabulary_path,
                  'default_value': self.default_value}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


def reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keep_dims,
                             name=name, reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keep_dims, name=name)


def reduce_max(input_tensor, axis=None, keep_dims=False,
               name=None, reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor, axis=axis, keep_dims=keep_dims,
                             name=name, reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor, axis=axis, keepdims=keep_dims, name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


class NoMask(Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask=None):
        return None


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)  # tf 1.x
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


# 欧氏距离
def euclidean(list_1, list_2):
    res = np.sqrt(((np.array(list_1) - np.array(list_2)) ** 2).sum())
    return res


# 皮尔逊相关系数
def pearson(list_1, list_2):
    avg_1 = np.mean(list_1)
    avg_2 = np.mean(list_2)
    tmp1, tmp2, tmp3 = 0, 0, 0
    for i in range(len(list_1)):
        tmp1 += (list_1[i] - avg_1) ** 2
        tmp2 += (list_2[i] - avg_2) ** 2
        tmp3 += (list_1[i] - avg_1) * (list_2[i] - avg_2)
    res = round(tmp3 / (np.sqrt(tmp1) * np.sqrt(tmp2) + 0.001), 4)
    return res


# 输出top k列表，利用最大/小堆实现
def top_k(candidate, k):
    """
    :param candidate: list 候选数据集列表，存储形式[[user,score],[...],...]
    :param k: int 选取top_k候选元素
    :return: list
    """
    q = []
    heapq.heapify(q)
    for i in range(len(candidate)):
        tmp = [candidate[i][1], candidate[i][0]]
        if len(q) < k:  # 长度不足时直接加入
            heapq.heappush(q, tmp)
        else:
            if q[0][0] < tmp[0]:  # 进行判断
                heapq.heappop(q)
                heapq.heappush(q, tmp)
    res = sorted(q, reverse=True)
    return res


# 存储相似度得分
def calculate_sim(arr, t):
    m = len(arr)
    dp = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if t == 'euc':
                dp[i][j] = euclidean(arr[i], arr[j])
            elif t == 'pea':
                dp[i][j] = pearson(arr[i], arr[j])
    return dp


def inbatch_softmax_cross_entropy_with_logits(logits, item_count, item_idx):
    # tf.squeeze删除尺度为1的维度，axis可以确定只删除具体位置
    Q = tf.gather(tf.constant(item_count / np.sum(item_count), 'float32'), tf.squeeze(item_idx, axis=1))
    logQ = tf.reshape(tf.math.log(Q), (1, -1))
    logits -= logQ  # subtract_log_q
    # 创建对角矩阵
    labels = tf.linalg.diag(tf.ones_like(logits[0]))
    # softmax_cross_entropy_with_logits：计算softmax分类问题的交叉熵损失
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return loss

