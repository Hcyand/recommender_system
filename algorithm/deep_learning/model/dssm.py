import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Model

from utils.feature_column import SparseFeat, VarLenSparseFeat, build_input_features
from utils.inputs import create_embedding_matrix, input_from_feature_columns, gen_data_set, gen_model_input
from utils.output import inner_product
from utils.negative import NegativeSampler, sampledsoftmaxloss

from layer.core import DNN, PredictionLayer
from layer.utils import combined_dnn_input
from layer.activation import InBatchSoftmaxLayer


def DSSM(user_feature_columns, item_feature_columns,
         user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32),
         dnn_activation='relu', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6,
         dnn_dropout=0, loss_type='softmax', temperature=0.05,
         sampler_config=None, seed=1024):
    """
    Instantiates the Deep Structured Semantic Model architecture.
    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, item tower
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

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns,
                                                    l2_reg_embedding, seed=seed, seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,  item_feature_columns,
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
    # read data
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

    tf.compat.v1.disable_eager_execution()

    model = DSSM(user_feature_columns, item_feature_columns,
                 user_dnn_hidden_units=(128, 64, embedding_dim), item_dnn_hidden_units=(64, embedding_dim,),
                 loss_type='softmax', sampler_config=sampler_config)
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
