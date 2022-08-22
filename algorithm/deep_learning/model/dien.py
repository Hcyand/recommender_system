# -*- coding: utf-8 -*-            
# @Time : 2022/8/12 17:13
# @Author : Hcyand
# @FileName: dien.py
import numpy as np
import tensorflow as tf
from layer.sequence import AUGRU
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Concatenate, Dense, Flatten, Permute

from utils.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features, get_feature_names
from utils.inputs import get_varlen_pooling_list, create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, \
    get_dense_input
from layer.core import DNN, PredictionLayer
from layer.sequence import AttentionSequencePoolingLayer
from layer.utils import concat_func, reduce_mean, combined_dnn_input


def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag=None):
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    """
    tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                    #  [True, True, True, False, False],
                                    #  [True, True, False, False, False]]
    """
    mask = tf.sequence_mask(mask, hist_len)
    mask = mask[:, 0, :]
    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)
    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    auxiliary_nn = DNN([100, 50, 1], activation='sigmoid')

    click_prop_ = auxiliary_nn(click_input_, stag=stag)[:, :, 0]
    noclick_prop_ = auxiliary_nn(noclick_input_, stag=stag)[:, :, 0]

    click_loss_ = -tf.reshape(tf.compat.v1.log(click_prop_),
                              [-1, tf.shape(click_seq)[1]]) * mask
    noclick_loss_ = - tf.reshape(tf.compat.v1.log(1.0 - noclick_prop_),
                                 [-1, tf.shape(noclick_seq)[1]]) * mask

    loss_ = reduce_mean(click_loss_ + noclick_loss_)

    return loss_


def interest_evolution(concat_behavior, deep_input_item, user_behavior_length, gru_type="AUGRU", use_neg=False,
                       neg_concat_behavior=None, att_hidden_size=(64, 16), att_activation='sigmoid',
                       att_weight_normalization=False, ):
    aux_loss_1 = None
    embedding_size = deep_input_item.shape[-1]
    # GRU func to h(1), h(2) ... h(T-1), h(T)
    rnn_outputs = GRU(embedding_size, return_sequences=True, name='gru1')(
        concat_behavior)

    if gru_type == "AUGRU" and use_neg:
        # get auxiliary loss
        aux_loss_1 = auxiliary_loss(rnn_outputs[:, :-1, :], concat_behavior[:, 1:, :],
                                    neg_concat_behavior[:, 1:, :],
                                    tf.subtract(user_behavior_length, 1), stag='gru')

    # get attention scores
    scores = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                           weight_normalization=att_weight_normalization, return_score=True)([
        deep_input_item, rnn_outputs, user_behavior_length])
    final_state = AUGRU(embedding_size, gru_type="AUGRU", return_sequences=False, name='gru2')(
        rnn_outputs, att_score=Permute([2, 1])(scores))
    hist = tf.expand_dims(final_state, axis=1)
    return hist, aux_loss_1


def DIEN(dnn_feature_columns, history_feature_list,
         gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(256, 128, 64),
         dnn_activation='relu',
         att_hidden_units=(64, 16), att_activation="dice", att_weight_normalization=True,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0., seed=1024, task='binary'):
    # get features columns dict
    features = build_input_features(dnn_feature_columns)
    # seq length
    user_behavior_length = features["seq_length"]
    # divide sparse/dense/varlen_sparse features
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    neg_history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))  # ['hist_item_id', 'hist_cate_id']
    neg_history_fc_names = list(map(lambda x: "neg_" + x, history_fc_names))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        elif feature_name in neg_history_fc_names:
            neg_history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="",
                                             seq_mask_zero=False)

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list, to_list=True)

    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)
    dnn_input_emb_list += sequence_embed_list
    keys_emb = concat_func(keys_emb_list)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list)

    if use_negsampling:

        neg_uiseq_embed_list = embedding_lookup(embedding_dict, features, neg_history_feature_columns,
                                                neg_history_fc_names, to_list=True)

        neg_concat_behavior = concat_func(neg_uiseq_embed_list)

    else:
        neg_concat_behavior = None
    # with GRU + Attention + AUGRU, aux_loss
    hist, aux_loss_1 = interest_evolution(keys_emb, query_emb, user_behavior_length, gru_type=gru_type,
                                          use_neg=use_negsampling, neg_concat_behavior=neg_concat_behavior,
                                          att_hidden_size=att_hidden_units,
                                          att_activation=att_activation,
                                          att_weight_normalization=att_weight_normalization, )

    deep_input_emb = Concatenate()([deep_input_emb, hist])

    deep_input_emb = Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)

    if use_negsampling:
        model.add_loss(alpha * aux_loss_1)

    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    tf.compat.v1.experimental.output_all_intermediates(True)

    return model


def get_xy_fd(use_neg=False, hash_flag=False):
    feature_columns = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
                         length_name="seq_length")]

    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])

    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': score, "seq_length": behavior_length}

    if use_neg:
        feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_cate_id'] = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])
        feature_columns += [
            VarLenSparseFeat(
                SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('neg_hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                             maxlen=4, length_name="seq_length")]

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    USE_NEG = True
    x, y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=USE_NEG)
    print(x)
    print(y)
    print('feature_columns: ', feature_columns)
    print('behavior_feature_list: ', behavior_feature_list)

    model = DIEN(feature_columns, behavior_feature_list,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.6, gru_type="AUGRU", use_negsampling=USE_NEG)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)

    # output X, y, feature_columns, behavior_feature_list
