# -*- coding: utf-8 -*-            
# @Time : 2022/8/6 15:03
# @Author : Hcyand
# @FileName: interaction.py
import tensorflow as tf
from keras.layers import Layer, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Embedding, BatchNormalization, PReLU
from tensorflow.python.ops.init_ops import Zeros
from keras.regularizers import l2


class WideLayer(Layer):
    def __init__(self):
        super(WideLayer, self).__init__()

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=l2(1e-4))

    def call(self, inputs, *args, **kwargs):
        x = self.w0 + tf.matmul(inputs, self.w)
        return x


# 可见core DNN
class DNNLayer(Layer):
    def __init__(self, hidden_units, output_dim, activation='relu', dropout=0.2):
        super(DNNLayer, self).__init__()
        self.hidden_layer = [Dense(x, activation=activation) for x in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        self.dropout_layer = Dropout(dropout)

    def build(self, input_shape):
        pass

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
            x = self.dropout_layer(x)
        output = self.output_layer(x)
        return output


class CrossLayer(Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super(CrossLayer, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)
        ]

        self.cross_bias = [
            self.add_weight(name='b' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)
        ]

    def call(self, inputs, *args, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)
        x1 = x0
        for i in range(self.layer_num):
            x1_w = tf.matmul(tf.transpose(x1, [0, 2, 1]), self.cross_weight[i])
            x1 = tf.matmul(x0, x1_w) + self.cross_bias[i] + x1

        output = tf.squeeze(x1, axis=2)
        return output


class FMLayer(Layer):
    def __init__(self, k, reg_w=1e-4, reg_b=1e-4):
        super(FMLayer, self).__init__()
        self.k = k
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w1 = self.add_weight(name='w1', shape=(input_shape[-1], 1),
                                  initializer=tf.random_normal_initializer(),
                                  trainable=True,
                                  regularizer=l2(self.reg_w))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=l2(self.reg_b))

    def call(self, inputs, *args, **kwargs):
        linear_part = tf.matmul(inputs, self.w1) + self.w0

        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)

        output = linear_part + inter_part
        return output


class FFMLayer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFMLayer, self).__init__()  # 找到FFM_layer的父类Layer，然后把类FFM_layer的对象转换为类layer的对象
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) \
                           + len(self.dense_feature_columns)
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


class InnerProductLayer(Layer):
    def __init__(self):
        super(InnerProductLayer, self).__init__()

    def call(self, inputs, **kwargs):
        field_num = inputs.shape[1]
        row, col = [], []
        # 先将要相乘的emb找出来，存在两个矩阵中，然后进行点乘即可
        for i in range(field_num - 1):
            for j in range(i + 1, field_num):
                row.append(i)
                col.append(j)
        # tf.gather根据indices的参数值获取切片
        p = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), row), [1, 0, 2])
        q = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), col), [1, 0, 2])
        innerProduct = tf.reduce_sum(p * q, axis=-1)

        return innerProduct


class OuterProductLayer(Layer):
    def __init__(self):
        super(OuterProductLayer, self).__init__()

    def build(self, input_shape):
        self.field_num = input_shape[1]
        self.k = input_shape[2]
        self.pair_num = self.field_num * (self.field_num - 1) // 2

        # 每个外积矩阵对应一个w矩阵，共有pair个
        self.w = self.add_weight(name='W', shape=(self.k, self.pair_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(1e-4),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        row, col = [], []
        for i in range(self.field_num - 1):
            for j in range(i + 1, self.field_num):
                row.append(i)
                col.append(j)
        p = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), row), [1, 0, 2])  # [None, pair_num, k]
        q = tf.transpose(tf.gather(tf.transpose(inputs, [1, 0, 2]), col), [1, 0, 2])  # [None, pair_num, k]
        p = tf.expand_dims(p, axis=1)  # [None, 1, pair_num, k]

        tmp = tf.multiply(p, self.w)  # [None, 1, pair_num, k] * [k, pair, k] = [None, k, pair_num, k]
        tmp = tf.reduce_sum(tmp, axis=-1)  # [None, k, pair_num]
        tmp = tf.multiply(tf.transpose(tmp, [0, 2, 1]), q)  # [None, pair_num, k]
        outputProduct = tf.reduce_sum(tmp, axis=-1)  # [None, pair_num]
        return outputProduct


class FGCNNLayer(Layer):
    def __init__(self, filters=[14, 16], kernel_width=[7, 7], dnn_maps=[3, 3], pooling_width=[2, 2]):
        super(FGCNNLayer, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width
        self.dnn_maps = dnn_maps
        self.pooling_width = pooling_width

    def build(self, input_shape):
        n = input_shape[1]
        k = input_shape[-1]
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []
        for i in range(len(self.filters)):
            self.conv_layers.append(
                Conv2D(filters=self.filters[i],
                       kernel_size=(self.kernel_width[i], 1),
                       strides=(1, 1),
                       padding='same',
                       activation='tanh')
            )
            self.pool_layers.append(
                MaxPool2D(pool_size=(self.pooling_width[i], 1))
            )
        self.flatten_layer = Flatten()

    def call(self, inputs, **kwargs):
        k = inputs.shape[-1]
        dnn_output = []
        x = tf.expand_dims(inputs, axis=-1)
        for i in range(len(self.filters)):
            x = self.conv_layers[i](x)
            x = self.pool_layers[i](x)
            out = self.flatten_layer(x)
            out = Dense(self.dnn_maps[i] * x.shape[1] * x.shape[2], activation='relu')(out)
            out = tf.reshape(out, shape=(-1, out.shape[1] // k, k))
            dnn_output.append(out)

        output = tf.concat(dnn_output, axis=1)
        return output


class ResLayer(Layer):
    """hidden layer"""

    def __init__(self, hidden_units):
        super(ResLayer, self).__init__()
        self.dense_layer = [Dense(i, activation='relu') for i in hidden_units]

    def build(self, input_shape):
        self.output_layer = Dense(input_shape[-1], activation=None)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        x = self.output_layer(x)

        output = inputs + x
        return tf.nn.relu(output)


class InteractionLayer(Layer):
    """
    input shape: [None, field, k]
    output shape: [None, field * (field - 1) / 2, k]
    """

    def __init__(self):
        super(InteractionLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        element_wise_product_list = []
        # 两两特征embed的multiply记录
        for i in range(inputs.shape[1]):
            for j in range(i + 1, inputs.shape[1]):
                element_wise_product_list.append(tf.multiply(inputs[:, i], inputs[:, j]))
        element_wise_product = tf.transpose(tf.convert_to_tensor(element_wise_product_list), [1, 0, 2])
        return element_wise_product


class AttentionLayer(Layer):
    """
    input shape: [None, n, k]
    output shape: [None, k]
    """

    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.attention_w = Dense(input_shape[1], activation='relu')
        self.attention_h = Dense(1, activation=None)

    def call(self, inputs, *args, **kwargs):
        x = self.attention_w(inputs)
        x = self.attention_h(x)
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1])
        output = tf.reshape(tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))
        return output


class AFMLayer(Layer):
    def __init__(self, feature_columns, mode):
        super(AFMLayer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layer = {'emb_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                            for i, feat in enumerate(self.sparse_feature_columns)}
        self.interaction_layer = InteractionLayer()
        if self.mode == 'att':
            self.attention_layer = AttentionLayer()
        self.output_layer = Dense(1)

    def call(self, inputs, *args, **kwargs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        embed = [self.embed_layer['emb_' + str(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])

        # Pair-wise Interaction
        embed = self.interaction_layer(embed)

        if self.mode == 'avg':
            x = tf.reduce_mean(embed, axis=1)
        elif self.mode == 'max':
            x = tf.reduce_max(embed, axis=1)
        else:
            x = self.attention_layer(embed)

        output = tf.nn.sigmoid(self.output_layer(x))
        return output


class Attention(Layer):
    def __init__(self, hidden_units, activation='prelu'):
        """
        :param hidden_units: 隐层
        :param activation: 激活函数（prelu、dice）；Dice需要自主实现
        """
        super(Attention, self).__init__()
        self.hidden_units = hidden_units
        if activation == 'dice':
            self.activation_layer = [Dice() for _ in range(len(self.hidden_units))]
        elif activation == 'prelu':
            self.activation_layer = [Dense(i, activation=PReLU()) for i in hidden_units]
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, *args, **kwargs):
        """
        query: [None, k]; item embedding layer
        key: [None, n ,k]; seq embedding layer
        value: [None, n, k]
        mask: [None, n]
        """
        query, key, value, mask = inputs

        # expand_dims 用于增加维度
        query = tf.expand_dims(query, axis=1)  # [None, 1, k]
        # tile对应维度复制几次
        query = tf.tile(query, [1, key.shape[1], 1])  # [None, n, k]
        """
        e.g
        tf.tile([1,2], [2]) = [1,2,1,2]
        tf.tile([[1,2][3,4]], [2,3]) = [[1,2,1,2,1,2],
                                        [3,4,3,4,3,4],
                                        [1,2,1,2,1,2],
                                        [3,4,3,4,3,4]]
        """
        # 激活单元中连接层操作，包括原输入、元素乘、元素减操作进行concat
        emb = tf.concat([query, key, query - key, query * key], axis=-1)  # [None, n, 4*k]

        for i in range(len(self.hidden_units)):
            emb = self.activation_layer[i](emb)

        score = self.out_layer(emb)  # [None, n, 1]
        score = tf.squeeze(score, axis=-1)  # 移除大小为1的维度 [None, n]

        padding = tf.ones_like(score) * (-2 ** 32 + 1)  # [None, n]
        # tf.where: return if tf.equal(mask, 0) padding else score
        score = tf.where(tf.equal(mask, 0), padding, score)  # [None, n]

        score = tf.nn.softmax(score)  # [None, n]
        output = tf.matmul(tf.expand_dims(score, axis=1), value)  # [None, 1, k]
        output = tf.squeeze(output, axis=1)  # [None, k]
        return output


class Dice(Layer):
    def __init__(self, axis=-1, epsilon=1e-9):
        super(Dice, self).__init__()
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        self.bn = BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(), dtype=tf.float32,
                                      name='dice_alpha')
        self.uses_learning_phase = True

    def call(self, inputs, training=None, *args, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs


class mmoe_layer(Layer):
    def __init__(self, hidden_units, num_experts, num_tasks, use_expert_bias=True, use_gate_bias=True, **kwargs):
        super(mmoe_layer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias

    def build(self, input_shape):
        # expert网络， 形状为[input_shape[-1], hidden_units, num_experts]
        # 每个expert将输入维度从input_shape[-1]映射到hidden_units
        self.expert_matrix = self.add_weight(
            name='expert_matrix',
            shape=(input_shape[-1], self.hidden_units, self.num_experts),
            trainable=True,
            initializer=tf.random_normal_initializer(),
            regularizer=l2(1e-4)
        )

        # expert网络偏置项，形状为[hidden_units, num_experts]
        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.hidden_units, self.num_experts),
                trainable=True,
                initializer=tf.random_normal_initializer,
                regularizer=l2(1e-4)
            )

        # gate网络，每个gate形状为[input_shape[-1], num_experts]
        # 总共num_tasks个gate，每个对应一个任务
        self.gate_matrix = [self.add_weight(
            name='gate_matrix' + str(i),
            shape=(input_shape[-1], self.num_experts),
            trainable=True,
            initializer=tf.random_normal_initializer(),
            regularizer=l2(1e-4)
        ) for i in range(self.num_tasks)]

        # gate网络偏执项，形状为[num_experts]，总共num_tasks个
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias' + str(i),
                shape=(self.num_experts,),
                trainable=True,
                initializer=tf.random_normal_initializer(),
                regularizer=l2(1e-4)
            ) for i in range(self.num_tasks)]

    def call(self, inputs, *args, **kwargs):
        # inputs x expert = [None, input_shape[-1]] x [input_shape[-1], hidden_units, num_experts]
        # get [None, hidden_units, num_experts]
        expert_output = []
        for i in range(self.num_experts):
            expert_out = tf.matmul(inputs, self.expert_matrix[:, :, i])
            expert_output.append(expert_out)
        expert_output = tf.transpose(
            tf.convert_to_tensor(expert_output), [1, 2, 0])  # [None, hidden_units, num_experts]

        # 加偏执，形状保持不变
        if self.use_expert_bias:
            expert_output += self.expert_bias
        expert_output = tf.nn.relu(expert_output)

        # inputs x gate = [None, input_shape[-1]] x [input_shape[-1], num_experts]
        # num_tasks个gate得到输出列表 num_tasks x [None, num_experts]
        gate_outputs = []
        for i, gate in enumerate(self.gate_matrix):
            gate_out = tf.matmul(inputs, gate)
            if self.use_gate_bias:
                gate_out += self.gate_bias[i]
            gate_out = tf.nn.softmax(gate_out)

            gate_outputs.append(gate_out)  # list: num_tasks x [None, num_experts]

        # gate与expert的输出相乘
        outputs = []
        for gate_out in gate_outputs:
            gate_out = tf.expand_dims(gate_out, axis=1)  # 维度扩展 [None, 1, num_experts]
            gate_out = tf.tile(gate_out, [1, self.hidden_units, 1])  # 维度复制 [None, hidden_units, num_experts]

            out = tf.multiply(gate_out, expert_output)  # 元素乘 [None, hidden_units, num_experts]
            out = tf.reduce_sum(out, axis=-1)  # 取平均 [None, hidden_units]
            outputs.append(out)

        return outputs  # list: num_tasks x [None, hidden_units]


class tower_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation='relu'):
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        super(tower_layer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output
