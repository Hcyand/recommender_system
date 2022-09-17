# -*- coding: utf-8 -*-            
# @Time : 2022/9/14 17:45
# @Author : Hcyand
# @FileName: transformer.py
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer

from layer.nlp import MultiHeadAttention


class PositionEncoding(Layer):
    """位置Embedding，通过固定公式计算"""

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings


class PositionWiseFeedForward(Layer):
    """Feed Forward，由两层全连接组成，一层激活函数为relu，后一层不使用激活函数"""

    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bias_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bias_inner")
        self.bias_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bias_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bias_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bias_out
        return outputs


class LayerNormalization(Layer):
    # 归一化层
    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs


class Transformer(Layer):
    """Transformer模型"""

    def __init__(self,
                 vocab_size,  # 序列大小
                 model_dim,  # embedding维度
                 n_heads=8,  # linear划分时数量，例如512维且h=8时，划分为512/8=64
                 encoder_stack=6,
                 decoder_stack=6,
                 feed_forward_size=2048,
                 dropout_rate=0.1,
                 **kwargs):

        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input embeddings
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        # encoder block
        self.EncoderPositionEncoding = PositionEncoding(self._model_dim)
        self.EncoderMultiHeadAttentions = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorms0 = [LayerNormalization() for _ in range(self._encoder_stack)]
        self.EncoderPositionWiseFeedForwards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorms1 = [LayerNormalization() for _ in range(self._encoder_stack)]

        # decoder block
        self.DecoderPositionEncoding = PositionEncoding(self._model_dim)
        # 需要使用masked，将参数future设置为True
        self.DecoderMultiHeadAttentions0 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms0 = [LayerNormalization() for _ in range(self._decoder_stack)]
        self.DecoderMultiHeadAttentions1 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms1 = [LayerNormalization() for _ in range(self._decoder_stack)]
        self.DecoderPositionWiseFeedForwards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms2 = [LayerNormalization() for _ in range(self._decoder_stack)]
        super(Transformer, self).build(input_shape)

    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        masks = K.equal(inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        # Position Encodings
        position_encodings = self.EncoderPositionEncoding(embeddings)
        # Embeddings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout encodings
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = self.EncoderMultiHeadAttentions[i]
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = self.EncoderLayerNorms0[i](attention_out)
            # Feed-Forward
            ff = self.EncoderPositionWiseFeedForwards[i]
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = self.EncoderLayerNorms1[i](ff_out)

        return encodings, masks

    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        # Position Encodings
        position_encodings = self.DecoderPositionEncoding(embeddings)
        # Embeddings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout encodings
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = self.DecoderMultiHeadAttentions0[i]
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = self.DecoderLayerNorms0[i](masked_attention_out)

            # Multi-head-Attention
            attention = self.DecoderMultiHeadAttentions1[i]
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = self.DecoderLayerNorms1[i](attention_out)

            # Feed-Forward
            ff = self.DecoderPositionWiseFeedForwards[i]
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = self.DecoderLayerNorms2[i](ff_out)

        # Pre-SoftMax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, inputs, **kwargs):
        encoder_inputs, decoder_inputs = inputs
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs


def label_smoothing(inputs, epsilon=0.1):
    """目标平滑"""
    output_dim = inputs.shape[-1]
    smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
    return smooth_label


###########################################################################################
# 测试模型效果
def load_dataset(vocab_size, max_len):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, x_train_masks, y_train), (x_test, x_test_masks, y_test)


def build_model(vocab_size, max_len, model_dim=8, n_heads=2, encoder_stack=2, decoder_stack=2, ff_size=50):
    encoder_inputs = tf.keras.Input(shape=(max_len,), name='encoder_inputs')
    decoder_inputs = tf.keras.Input(shape=(max_len,), name='decoder_inputs')
    outputs = Transformer(
        vocab_size,
        model_dim,
        n_heads=n_heads,
        encoder_stack=encoder_stack,
        decoder_stack=decoder_stack,
        feed_forward_size=ff_size
    )([encoder_inputs, decoder_inputs])
    outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(outputs)
    return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)


def train_model(vocab_size=5000, max_len=128, batch_size=128, epochs=10):
    train, test = load_dataset(vocab_size, max_len)

    x_train, x_train_masks, y_train = train
    x_test, x_test_masks, y_test = test

    model = build_model(vocab_size, max_len)

    model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit([x_train, x_train_masks], y_train,
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])

    test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    print("loss on Test: %.4f" % test_metrics[0])
    print("accu on Test: %.4f" % test_metrics[1])


if __name__ == '__main__':
    train_model()
