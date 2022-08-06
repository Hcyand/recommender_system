# -*- coding: utf-8 -*-            
# @Time : 2022/8/1 16:04
# @Author : Hcyand
# @FileName: ffm.py
from layer.interaction import FFMLayer
from utils.criteo_dataset import create_criteo_dataset, features_dict
from utils.compile_fit import compile_fit

import tensorflow as tf
from keras import Model, losses
from sklearn.metrics import accuracy_score
from keras import optimizers


class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFMLayer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output


# 测试数据
if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    test_size = 0.2
    k = 8

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('ffm', file, test_size=test_size)
    feature_dict = features_dict(file)

    model = FFM(feature_dict, k=k)
    model = compile_fit(model, X_train, y_train)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('Accuracy: ', accuracy_score(y_test, pre))
