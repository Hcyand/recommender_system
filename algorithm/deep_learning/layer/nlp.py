# -*- coding: utf-8 -*-            
# @Time : 2022/9/14 17:46
# @Author : Hcyand
# @FileName: nlp.py
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer


class Embedding(Layer):
    """
    这里利用Transform训练每个单词的Embedding作为输出；
    也可以利用Word2Vec等算法进行计算得到单词Embedding
    """
    def __init__(self, vocab_size, model_dim, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        return embeddings


class ScaledDotProductAttention(Layer):
    """
    1. 将mask信息进行线性划分；
    2. 利用LinearOperatorLowerTriangular生成倒三角进行masked操作
    3. 输出相乘结果
    """
    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        # 将mask进行线性划分处理
        # input: [None*2, 128, 128]
        masks = K.cast(masks, 'float32')  # [None, 128]
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])  # [None*2, 128]
        masks = K.expand_dims(masks, 1)  # [None*2, 1, 128]
        outputs = inputs + masks * self._masking_num
        return outputs  # [None*2, 128, 128]

    def future_mask(self, inputs):
        # 生成tril进行masked操作
        diag_vals = tf.ones_like(inputs[0, :, :])
        """
        LinearOperatorLowerTriangular func
        input: [[1,2,3],[4,5,6],[7,8,9]]
        output: [[1,0,0],[4,5,0],[7,8,9]]
        """
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # [128, 128]
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # [None*2, 128, 128]
        paddings = tf.ones_like(future_masks) * self._masking_num
        # tf.where: if true paddings[i] else inputs[i]
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)  # [None*2, 128, 128]
        return outputs

    def call(self, inputs, masks=0, **kwargs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':
            queries = K.cast(queries, 'float32')  # [None*2, 128, 4]
        if K.dtype(keys) != 'float32':
            keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':
            values = K.cast(values, 'float32')

        # Q * K
        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul [None*2, 128, 128]
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks)  # Mask(opt.)

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        out = K.dropout(softmax_out, self._dropout_rate)  # Dropout

        outputs = K.batch_dot(out, values)

        return outputs


class MultiHeadAttention(Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')  # [8, 8]
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, masks=0, **kwargs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            # queries: [None, 128, 8]
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)  # [None, 128, 8]
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        # Q, K, V Linear split
        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)  # [None*2, 128, 4]
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)  # [None*2, 128, 4]

        # concat att_outs
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)  # [None, 128, 8]

        return outputs
