# -*- coding: utf-8 -*-            
# @Time : 2022/8/5 12:19
# @Author : Hcyand
# @FileName: fnn.py
from layer.interaction import FMLayer, DNNLayer
from keras.models import Model
from utils.dataset import create_criteo_dataset
import tensorflow as tf
from tensorflow.python.keras import optimizers
from sklearn.metrics import accuracy_score


class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FMLayer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        output = tf.nn.sigmoid(output)
        return output


class DNN(Model):
    def __init__(self, hidden_units, output_dim, activation='relu'):
        super(DNN, self).__init__()
        self.dnn = DNNLayer(hidden_units, output_dim, activation)

    def call(self, inputs, training=None, mask=None):
        output = self.dnn(inputs)
        output = tf.nn.sigmoid(output)
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('fnn', file, test_size=0.3)
    k = 8

    model = FM(k)
    optimizer = optimizers.gradient_descent_v2.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=20)

    fm_pre = model(X_test)
    fm_pre = [1 if x > 0.5 else 0 for x in fm_pre]

    v = model.variables[2]

    X_train = tf.cast(tf.expand_dims(X_train, -1), tf.float32)
    X_train = tf.reshape(tf.multiply(X_train, v), shape=(-1, v.shape[0] * v.shape[1]))

    hidden_units = [256, 128, 64]
    model = DNN(hidden_units, 1, 'relu')
    optimizer = optimizers.gradient_descent_v2.SGD(0.0001)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=50)

    X_test = tf.cast(tf.expand_dims(X_test, -1), tf.float32)
    X_test = tf.reshape(tf.multiply(X_test, v), shape=(-1, v.shape[0] * v.shape[1]))
    fnn_pre = model(X_test)
    fnn_pre = [1 if x > 0.5 else 0 for x in fnn_pre]

    print('FM Accuracy: ', accuracy_score(y_test, fm_pre))
    print('FNN Accuracy: ', accuracy_score(y_test, fnn_pre))
