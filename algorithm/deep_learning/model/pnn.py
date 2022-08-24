# -*- coding: utf-8 -*-            
# @Time : 2022/8/3 15:20
# @Author : Hcyand
# @FileName: pnn.py
from layer.interaction import DNNLayer, InnerProductLayer, OuterProductLayer, FGCNNLayer
from utils.dataset import create_criteo_dataset, features_dict
import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding
from keras import optimizers, losses
from sklearn.metrics import accuracy_score


class PNN(Model):
    def __init__(self, feature_columns, mode, hidden_units, output_dim, activation='relu', dropout=0.2,
                 use_fgcnn=False):
        super(PNN, self).__init__()
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dnn_layer = DNNLayer(hidden_units, output_dim, activation, dropout)
        self.inner_product_layer = InnerProductLayer()
        self.outer_product_layer = OuterProductLayer()
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.use_fgcnn = use_fgcnn
        if use_fgcnn:
            self.fgcnn_layer = FGCNNLayer()

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])

        if self.use_fgcnn:
            fgcnn_out = self.fgcnn_layer(embed)
            embed = tf.concat([embed, fgcnn_out], axis=1)

        z = embed
        embed = tf.reshape(embed, shape=(-1, embed.shape[1] * embed.shape[2]))
        if self.mode == 'inner':
            inner_product = self.inner_product_layer(z)
            inputs = tf.concat([embed, inner_product], axis=1)
        elif self.mode == 'outer':
            outer_product = self.outer_product_layer(z)
            inputs = tf.concat([embed, outer_product], axis=1)
        elif self.mode == 'both':
            inner_product = self.inner_product_layer(z)
            outer_product = self.outer_product_layer(z)
            inputs = tf.concat([embed, inner_product, outer_product], axis=1)
        else:
            raise ValueError("Please choice mode's value in 'inner', 'outer', 'both'.")

        output = self.dnn_layer(inputs)
        return output


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    dropout = 0.3
    mode = 'both'
    use_fgcnn = True

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('pnn', file)
    feature_dict = features_dict(file)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    model = PNN(feature_dict, mode, hidden_units, output_dim, activation, dropout, use_fgcnn)
    optimizer = optimizers.SGD(0.01)

    for epoch in range(10):
        sum_loss = []
        for batch, data_batch in enumerate(train_dataset):
            X_train, y_train = data_batch[0], data_batch[1]
            with tf.GradientTape() as tape:
                pre = model(X_train)
                loss = tf.reduce_mean(losses.binary_crossentropy(y_train, pre))
                grad = tape.gradient(loss, model.variables)
                optimizer.apply_gradients((zip(grad, model.variables)))
            sum_loss.append(loss.numpy())
            if batch % 10 == 0:
                print("epoch: {} batch: {} loss: {}".format(epoch, batch, tf.reduce_mean(sum_loss)))

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('Accuracy: ', accuracy_score(y_test, pre))
