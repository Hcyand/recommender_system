# -*- coding: utf-8 -*-            
# @Time : 2022/8/21 11:07
# @Author : Hcyand
# @FileName: mmoe.py
from layer.interaction import mmoe_layer, tower_layer
import tensorflow as tf
from tensorflow.python.keras import Model


class MMOE(Model):
    def __init__(self, mmoe_hidden_units, num_experts, num_tasks,
                 tower_hidden_units, output_dim, activation='relu',
                 use_expert_bias=True, use_gate_bias=True, **kwargs):
        super(MMOE, self).__init__()
        self.mmoe_layer = mmoe_layer(mmoe_hidden_units, num_experts, num_tasks,
                                     use_expert_bias, use_gate_bias)

        # 每个任务对应一个tower_laye
        self.tower_layer = [
            tower_layer(tower_hidden_units, output_dim, activation)
            for _ in range(num_tasks)
        ]

        def call(self, inputs, training=None, mask=None):
            mmoe_outputs = self.mmoe_layer(inputs)

            outputs = []
            for i, layer in enumerate(self.tower_layer):
                out = layer(mmoe_outputs[i])
                outputs.append(out)

            return outputs  # num_tasks * [None, output_dim]


if __name__ == '__main__':
    mmoe_hidden_units = 10
    num_experts = 3
    num_tasks = 3
    tower_hidden_units = [20, 10]
    tower_output_dim = 1

    model = MMOE(mmoe_hidden_units, num_experts, num_tasks,
                 tower_hidden_units, tower_output_dim)

    input = tf.constant(
        [[1., 1., 1., 3.],
         [2., 2., 1., 4.]]
    )

    output = model(input)
    print(output)
