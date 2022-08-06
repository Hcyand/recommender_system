# -*- coding: utf-8 -*-            
# @Time : 2022/8/6 17:15
# @Author : Hcyand
# @FileName: compile_fit.py
import tensorflow as tf
from keras import optimizers


def compile_fit(model, X, y, batch_size=32, epochs=10, sgd=0.01):
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    optimizer = optimizers.SGD(sgd)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=epochs)
    return model
