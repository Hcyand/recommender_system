# -*- coding: utf-8 -*-            
# @Time : 2022/8/9 23:15
# @Author : Hcyand
# @FileName: movie_lens.py
# 对movie_lens数据进行处理
# 需要获取的数据：
# 用户-电影表（用户属性、电影属性、评分等）；
# 用户-电影序列表：用户最近好评的10部电影（评分大于等于3为好评）；
# 电影序列表：电影序列数据；
# rating > 3的为正，其余为负
# 将用户随机划分，分为训练集和测试集
# 用户的历史电影评分序列，取最近时间的结果为训练集，其余为历史序列
import numpy as np
import pandas as pd
import collections
import heapq
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

file_ratings = '../../data/ml-1m/ratings.dat'
file_users = '../../data/ml-1m/users.dat'
file_movies = '../../data/ml-1m/movies.dat'
dense_features = ['age', 'occupation']
sparse_features = ['user_id', 'movie_id', 'gender', 'zip_code', 'generes', 'title']
behavior_features = ['movies_seq']


def latest_top_k(seq, k, candidate):
    """依据时间timestamp，获取top k的序列"""
    heapq.heapify(seq)
    if len(seq) < k:
        heapq.heappush(seq, candidate)
    else:
        if seq[0][0] < candidate[0]:
            heapq.heappop(seq)
            heapq.heappush(seq, candidate)
    seq = sorted(seq, reverse=True)
    return seq


def sparseFeature(feat, feat_onehot_dim, embed_dim):
    # 特征、one_hot维度、embed维度
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}


def create_movies_dataset(test_size=0.3, n=10):
    # 读取数据
    ratings = pd.read_csv(file_ratings, sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
    users = pd.read_csv(file_users, sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    movies = pd.read_csv(file_movies, encoding='ISO-8859-1', sep='::', header=None,
                         names=['movie_id', 'title', 'generes'])

    # 将rating划分为历史数据和最新数据
    max_time_list = collections.defaultdict(int)
    s = []
    for userId, timestamp in np.array(ratings[['user_id', 'timestamp']]):
        max_time_list[userId] = max(max_time_list[userId], timestamp)  # 记录userId的最大时间timestamp
    for userId, timestamp in np.array(ratings[['user_id', 'timestamp']]):
        if max_time_list[userId] == timestamp:
            s.append(True)
        else:
            s.append(False)
    ratings_latest = ratings[s]  # 最近时间的用户的评价数据
    ratings_old = ratings[[not x for x in s]]  # 用户历史的评价数据

    # 输出用户历史的喜爱电影序列movies_seq
    d, u_seq = collections.defaultdict(list), collections.defaultdict(list)
    like_rate = np.array(ratings_old.query('rating >= 3')[['user_id', 'movie_id', 'timestamp']])  # 历史喜爱电影
    for user, movie, times in like_rate:
        d[user] = latest_top_k(d[user], 10, [times, movie])  # 获取top k最近的喜爱电影
    for u in users['user_id'].values:
        # 序列需要相同的长度，不足长度的序列赋值0
        if d[u]:
            tmp = [x[1] for x in d[u]]
            if len(d[u]) < n:
                tmp += [0] * (n - len(d[u]))
        else:
            tmp = [0] * n
        u_seq[u] = tmp
    users['movies_seq'] = users['user_id'].map(lambda x: u_seq[x])

    # 数据合并，数据类型转换，缺失值处理
    data = pd.merge(ratings_latest, users, on='user_id', how='left')
    data = pd.merge(data, movies, on='movie_id', how='left')
    data['label'] = data['rating'].map(lambda x: 1 if x > 3 else 0)
    data = data.drop(['timestamp', 'rating'], axis=1)
    data['user_id'] = data['user_id'].astype(str)
    data['movie_id'] = data['movie_id'].astype(str)
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    # 特征处理
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    for col in sparse_features:
        data[col] = LabelEncoder().fit_transform(data[col]).astype(int)

    # 划分训练集和测试集
    X = data.drop(['label'], axis=1)
    y = data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    embed_dim = 8
    # feature_onehot_dim的处理方式有待优化
    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, data[feat].nunique() + 1, embed_dim)
                        for feat in sparse_features] +
                       [sparseFeature(feat, 6400 + 1, embed_dim)
                        for feat in behavior_features]]
    # len(np.unique(list(data[feat].values))) 不能这么处理，因为有些item之前可能没有出现过，所以需要记录的是所有item的量级

    return feature_columns, (X_train, y_train), (X_test, y_test)
