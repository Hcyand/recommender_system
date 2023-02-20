import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Concatenate, Dense, Flatten, Permute

from utils.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features, get_feature_names
from utils.inputs import get_varlen_pooling_list, create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, \
    get_dense_input
from utils.negative import NegativeSampler, sampledsoftmaxloss
from layer.core import DNN, PredictionLayer
from layer.sequence import AttentionSequencePoolingLayer
from layer.utils import concat_func, reduce_mean, combined_dnn_input
from itertools import chain
from collections import defaultdict
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Layer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from tensorflow.python.keras import backend as K


def inner_product(x, y, temperature=1.0):
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1])) / temperature)([x, y])


def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c


def inbatch_softmax_cross_entropy_with_logits(logits, item_count, item_idx):
    # tf.squeeze删除尺度为1的维度，axis可以确定只删除具体位置
    Q = tf.gather(tf.constant(item_count / np.sum(item_count), 'float32'), tf.squeeze(item_idx, axis=1))
    logQ = tf.reshape(tf.math.log(Q), (1, -1))
    logits -= logQ  # subtract_log_q
    # 创建对角矩阵
    labels = tf.linalg.diag(tf.ones_like(logits[0]))
    # softmax_cross_entropy_with_logits：计算softmax分类问题的交叉熵损失
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return loss


class InBatchSoftmaxLayer(Layer):
    def __init__(self, sampler_config, temperature=1.0, **kwargs):
        self.sampler_config = sampler_config
        self.temperature = temperature
        self.item_count = self.sampler_config['item_count']

        super(InBatchSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InBatchSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):
        user_vec, item_vec, item_idx = inputs_with_item_idx
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)
        user_vec /= self.temperature
        logits = tf.matmul(user_vec, item_vec, transpose_b=True)
        loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)
        return tf.expand_dims(loss, axis=1)


def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False, embedding_matrix_dict=None):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    if embedding_matrix_dict is None:
        embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                        seq_mask_zero=seq_mask_zero)

    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list


def gen_data_set(data, seq_max_len=50, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()
    item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values))
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len,
                                  genres_hist[::-1][:seq_len], genres_list[i], rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
                                      genres_hist[::-1][:seq_len], item_id_genres_map[neg_list[i * negsample + negi]]))
            else:
                test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i],
                                 rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    train_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_genres = np.array([line[5] for line in train_set])
    train_genres = np.array([line[6] for line in train_set])
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_seq_genres_pad = pad_sequences(train_seq_genres, maxlen=seq_max_len, padding='post', truncating='post',
                                         value=0)
    train_model_inputs = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                          "hist_genres": train_seq_genres_pad,
                          "hist_len": train_hist_len, "genres": train_genres}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_inputs[key] = user_profile.loc[train_model_inputs['user_id']][key].values

    return train_model_inputs, train_label


def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='relu', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, loss_type='softmax', temperature=0.05,
         sampler_config=None,
         seed=1024, ):
    """Instantiates the Deep Structured Semantic Model architecture.
    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param loss_type: string. Loss type.
    :param temperature: float. Scaling factor.
    :param sampler_config: negative sample config.
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.
    """

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    # user dnn
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation='linear', seed=seed)(user_dnn_input)
    user_dnn_out = tf.math.l2_normalize(user_dnn_out, axis=-1)

    # 判断item是否还需要经过DNN转换
    if len(item_dnn_hidden_units) > 0:
        # item dnn
        item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                           dnn_use_bn, output_activation='linear', seed=seed)(item_dnn_input)
    else:
        item_dnn_out = item_dnn_input
    item_dnn_out = tf.math.l2_normalize(item_dnn_out, axis=-1)

    if loss_type == "logistic":
        score = inner_product(user_dnn_out, item_dnn_out, temperature)
        output = PredictionLayer("binary", False)(score)

    elif loss_type == "softmax":
        output = InBatchSoftmaxLayer(sampler_config._asdict(), temperature)(
            [user_dnn_out, item_dnn_out, item_features[sampler_config.item_name]])
    else:
        raise ValueError(' `loss_type` must be `logistic` or `softmax` ')

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    # __setattr__用来设置对象的属性值
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model


if __name__ == '__main__':
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    mnames = ['movie_id', 'title', 'genres']
    user = pd.read_csv('../../data/ml-1m/users.dat', sep='::', header=None, names=unames)
    ratings = pd.read_csv('../../data/ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    movies = pd.read_csv('../../data/ml-1m/movies.dat', sep='::', header=None, names=mnames, encoding="unicode_escape")
    movies['genres'] = list(map(lambda x: x.split('|')[0], movies['genres'].values))

    data = pd.merge(pd.merge(ratings, movies), user)

    # 构建特征列，训练模型，导出embedding
    sparse_feature = ['movie_id', 'user_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    SEQ_LEN = 50
    negsample = 0

    feature_max_idx = {}
    for feature in sparse_feature:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates('user_id')
    item_profile = data[['movie_id', 'genres']].drop_duplicates('movie_id')
    user_profile.set_index('user_id', inplace=True)
    user_item_list = data.groupby('user_id')['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, SEQ_LEN, negsample)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    embedding_dim = 32
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            VarLenSparseFeat(SparseFeat('hist_genres', feature_max_idx['genres'], embedding_dim,
                                                        embedding_name="genres"), SEQ_LEN, 'mean', 'hist_len'),
                            ]
    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim),
                            SparseFeat('genres', feature_max_idx['genres'], embedding_dim)
                            ]

    train_counter = Counter(train_model_input['movie_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('inbatch', num_sampled=255, item_name="movie_id", item_count=item_count)

    import tensorflow as tf

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)

    model = DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(128, 64, embedding_dim),
                 item_dnn_hidden_units=(64, embedding_dim,), loss_type='softmax', sampler_config=sampler_config)

    model.compile(optimizer="adam", loss=sampledsoftmaxloss)

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=256, epochs=20, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile['movie_id'].values, "genres": item_profile['genres'].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)
