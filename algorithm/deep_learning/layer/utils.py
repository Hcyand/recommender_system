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
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


class AUGRUCell(GRUCell):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        super(AUGRUCell, self).__init__(units, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel'
        )

    def call(self, inputs, states, att_score=None, training=None):
        h_tm1 = states[0] if nest.is_nested(states) else states

        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

        x_z = backend.dot(inputs_z, self.kernel[:, :self.units])
        x_r = backend.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
        x_h = backend.dot(inputs_h, self.kernel[:, self.units * 2:])

        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

        recurrent_z = backend.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
        recurrent_r = backend.dot(h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])

        z = self.recurrent_activation(x_z + recurrent_z)
        z = att_score * z
        r = self.recurrent_activation(x_r + recurrent_r)

        recurrent_h = backend.dot(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:])

        hh = self.activation(x_h + recurrent_h)

        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if nest.is_nested(states) else h
        return h, new_state


def rnn_augru(step_function,
              inputs,
              initial_states,
              att_score=None,
              go_backwards=False,
              mask=None,
              constants=None,
              unroll=True,
              input_length=None,
              time_major=False,
              zero_output_for_mask=False):
    def swap_batch_timestep(input_t):
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return array_ops.transpose(input_t, axes)

    if not time_major:
        inputs = nest.map_structure(swap_batch_timestep, inputs)

    flatted_inputs = nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = array_ops.shape(flatted_inputs[0])[0]

    if mask is not None:
        if mask.dtype != dtypes_module.bool:
            mask = math_ops.cast(mask, dtypes_module.bool)
        if len(mask.shape) == 2:
            mask = expand_dims(mask)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if nest.is_nested(mask_t):
            raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
        if nest.is_nested(input_t):
            raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = array_ops.expand_dims(mask_t, -1)
        multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
        return array_ops.tile(mask_t, multiples)

    # if roll
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []

    def _process_single_input_t(input_t):
        input_t = array_ops.unstack(input_t)
        if go_backwards:
            input_t.reverse()
        return input_t

    if nest.is_nested(inputs):
        processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
        processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
        inp = [t_[time] for t_ in processed_input]
        return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
        mask_list = array_ops.unstack(mask)
        if go_backwards:
            mask_list.reverse()
        for i in range(time_steps):
            inp = _get_input_tensor(i)
            mask_t = mask_list[i]
            output, new_states = step_function(inp, tuple(states) + tuple(constants), att_score[i])
            tiled_mask_t = _expand_mask(mask_t, output)

            if not successive_outputs:
                prev_output = zeros_like(output)
            else:
                prev_output = successive_outputs[-1]

            output = array_ops.where_v2(tiled_mask_t, output, prev_output)

            flat_states = nest.flatten(states)
            flat_new_states = nest.flatten(new_states)
            tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
            flat_final_states = tuple(array_ops.where_v2(m, s, ps)
                                      for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
            states = nest.pack_sequence_as(states, flat_final_states)

            successive_outputs.append(output)
            successive_states.append(states)
        last_output = successive_outputs[-1]
        new_states = successive_states[-1]
        outputs = array_ops.stack(successive_outputs)

    else:
        for i in range(time_steps):
            inp = _get_input_tensor(i)
            output, states = step_function(inp, tuple(states) + tuple(constants), att_score=att_score[:, i, :])
            successive_outputs.append(output)
            successive_states.append(states)
        last_output = successive_outputs[-1]
        new_states = successive_states[-1]
        outputs = array_ops.stack(successive_outputs)

    def set_shape(output_):
        if isinstance(output_, ops.Tensor):
            shape = output_.shape.as_list()
            shape[0] = time_steps
            shape[1] = batch
            output_.set_shape(shape)
        return output_

    outputs = nest.map_structure(set_shape, outputs)

    if not time_major:
        outputs = nest.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


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
