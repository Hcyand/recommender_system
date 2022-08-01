# -*- coding: utf-8 -*-            
# @Time : 2022/8/1 16:04
# @Author : Hcyand
# @FileName: fm.py
from layer.factorization import FM_layer
from utils.criteo_dataset import create_criteo_dataset

import tensorflow as tf
from tensorflow.python.keras import Model, losses
from sklearn.metrics import accuracy_score
from keras import optimizers


class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FM_layer(k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        output = self.fm(inputs)
        output = tf.nn.sigmoid(output)
        return output


# 测试数据
if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    test_size = 0.2
    k = 8
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('fm', file, test_size=test_size)

    model = FM(k=k)
    optimizer = optimizers.SGD(0.01)
    for i in range(10):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('Accuracy: ', accuracy_score(y_test, pre))
