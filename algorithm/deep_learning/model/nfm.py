# -*- coding: utf-8 -*-            
# @Time : 2022/8/7 15:16
# @Author : Hcyand
# @FileName: nfm.py
from layer.interaction import DNNLayer
from layer.inputs import EmbedLayer
from utils.criteo_dataset import create_criteo_dataset, features_dict
from utils.compile_fit import compile_fit
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Embedding
from sklearn.metrics import accuracy_score


class NFM(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation='relu', dropout=0):
        super(NFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dnn_layers = DNNLayer(hidden_units, output_dim, activation, dropout)
        self.emb_layers = {'emb_'+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                           for i, feat in enumerate(self.sparse_feature_columns)}
        self.bn_layer = BatchNormalization()  # 批标准化
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        emb = [self.emb_layers['emb_'+str(i)](sparse_inputs[:, i])
               for i in range(sparse_inputs.shape[1])]
        emb = tf.transpose(tf.convert_to_tensor(emb), [1, 0, 2])  # [None, 26, embed_dim]
        # Bi-Interaction Layer
        emb = 0.5 * (tf.pow(tf.reduce_sum(emb, axis=1), 2) - tf.reduce_sum(tf.pow(emb, 2), axis=1))  # [None, embed_dim]
        x = tf.concat([dense_inputs, emb], axis=-1)
        x = self.bn_layer(x)
        x = self.dnn_layers(x)

        outputs = self.output_layer(x)
        return tf.nn.sigmoid(outputs)


if __name__ == '__main__':
    file = '../../data/criteo/train_1w.txt'
    test_size = 0.3
    hidden_units = [256, 128, 64]
    output_dim = 1
    dropout = 0.3

    (X_train, y_train), (X_test, y_test) = create_criteo_dataset('nfm', file, test_size=test_size)
    feature_dict = features_dict(file)

    model = NFM(feature_dict, hidden_units, output_dim)
    model = compile_fit(model, X_train, y_train)
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print('ACC: ', accuracy_score(y_test, pre))

