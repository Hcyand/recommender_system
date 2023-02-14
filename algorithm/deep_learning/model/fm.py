# -*- coding: utf-8 -*-            
# @Time : 2022/8/1 16:04
# @Author : Hcyand
# @FileName: fm.py
from layer.interaction import FMLayer
from utils.dataset import create_criteo_dataset
from utils.compile_fit import compile_fit

import tensorflow as tf
from tensorflow.python.keras import Model
from sklearn.metrics import accuracy_score


class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FMLayer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        print(inputs.shape)
        output = self.fm(inputs)
        output = tf.nn.sigmoid(output)
        return output


# 测试数据
if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    test_size = 0.2
    k = 8
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('fm', file, test_size=test_size)

    model = FM(k)
    model = compile_fit(model, X_train, y_train)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('Accuracy: ', accuracy_score(y_test, pre))
