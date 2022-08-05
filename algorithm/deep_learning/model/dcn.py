# -*- coding: utf-8 -*-            
# @Time : 2022/8/4 15:49
# @Author : Hcyand
# @FileName: dcn.py
from layer.dcn import DenseLayer, CrossLayer
import tensorflow as tf
from keras.layers import Dense, Embedding
from keras import Model
from utils.criteo_dataset import create_criteo_dataset, features_dict
from keras import losses, optimizers
from sklearn.metrics import accuracy_score


class DCN(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, layer_num, reg_w=1e-4, reg_b=1e-4):
        super(DCN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dense_layer = DenseLayer(hidden_units, output_dim, activation)
        self.cross_layer = CrossLayer(layer_num, reg_w, reg_b)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=1)

        cross_output = self.cross_layer(x)
        dnn_output = self.dense_layer(x)

        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    test_size = 0.3
    hidden_units = [256, 128, 64]

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('dcn', file)
    feature_dict = features_dict(file)

    model = DCN(feature_dict, hidden_units, 1, activation='relu', layer_num=6)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    for i in range(10):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, y_pre))
            print(loss.numpy())
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('ACC: ', accuracy_score(y_test, pre))
