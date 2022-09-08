# -*- coding: utf-8 -*-            
# @Time : 2022/9/8 14:08
# @Author : Hcyand
# @FileName: lstm.py

# copy from tensorflow.python.keras.layers.recurrent LSTM, LSTMCell

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _caching_device, RNN, InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')


class LSTMCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthoginal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for units, expected '
                             f'a positive integer, got {units}.')
        # By default use cached variable under v2 mode, see b/143699808.
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(LSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        # implementation 实现方式：1会将运算分解为若干小矩阵的乘法加法运算，2则相反
        # implementation==1的情况下，对于各个门的输入input_t/状态state_t都可以使用不同的mask值来进行dropout；
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return backend.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get('ones')((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(
            x_f, backend.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + backend.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]
        c_tm1 = states[1]

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)

        if self.implementation == 1:
            if 0 < self.dropout < 1:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)
            x_i = backend.dot(inputs_i, k_i)
            x_f = backend.dot(inputs_f, k_f)
            x_c = backend.dot(inputs_c, k_c)
            x_o = backend.dot(inputs_o, k_o)

            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(self.bias, num_or_size_splits=4, axis=1)
                x_i = backend.bias_add(x_i, b_i)
                x_f = backend.bias_add(x_f, b_f)
                x_c = backend.bias_add(x_c, b_c)
                x_o = backend.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1

            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
            z = backend.dot(inputs, self.kernel)
            z += backend.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = backend.bias_add(z, self.bias)
            z = array_ops.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]


# 需要关注的是RNN算法的代码实现
@keras_export(v1=['keras.layers.LSTM'])
class LSTM(RNN):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        implementation = kwargs.pop('implementation', 1)
        if implementation == 0:
            logging.warning('`implementation=0` has been deprecated, '
                            'and now defaults to `implementation=1`.'
                            'Please update your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                               kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = LSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(LSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(LSTM, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)
