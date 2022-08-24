# -*- coding: utf-8 -*-            
# @Time : 2022/8/15 17:17
# @Author : Hcyand
# @FileName: activation.py
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.layers import GRUCell
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.backend import zeros_like, expand_dims

try:
    from tensorflow.python.ops.init_ops import Zeros
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros
from tensorflow.python.keras.layers import Layer, Activation

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization

unicode = str


class Dice(Layer):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

      Input shape
        - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

      Output shape
        - Same shape as the input.

      Arguments
        - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).

        - **epsilon** : Small float added to variance to avoid dividing by zero.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, unicode)):
        act_layer = Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


# AUGRU Cell
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
        z = att_score * z  # 相乘注意力得分att_score
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
