# -*- coding: utf-8 -*-            
# @Time : 2022/8/2 15:14
# @Author : Hcyand
# @FileName: deepCrossing.py
from layer.interaction import ResLayer
from layer.inputs import EmbedLayer
from utils.criteo_dataset import create_criteo_dataset, features_dict
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import accuracy_score


class DeepCrossing(Model):
    def __init__(self, feature_columns, k, hidden_units, res_layer_num):
        super(DeepCrossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = EmbedLayer(self.sparse_feature_columns)
        self.res_layer = [ResLayer(hidden_units) for _ in range(res_layer_num)]
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        emb = self.embed_layer(sparse_inputs)
        x = tf.concat([dense_inputs, emb], axis=-1)
        for layer in self.res_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    k = 32
    hidden_units = [256, 256]
    res_layer_num = 4

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('DeepCrossing', file)
    feature_dict = features_dict(file)

    model = DeepCrossing(feature_dict, k, hidden_units, res_layer_num)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=10)

    # 评估
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
