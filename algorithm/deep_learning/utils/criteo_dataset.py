# -*- coding: utf-8 -*-            
# @Time : 2022/8/1 15:09
# @Author : Hcyand
# @FileName: criteo_dataset.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

dense_features = ['I' + str(i) for i in range(1, 14)]
sparse_features = ['C' + str(i) for i in range(1, 27)]


# FFM model 对类别特征的记录
def sparseFeature(feat, feat_onehot_dim, embed_dim):
    # 特征、one_hot维度、embed维度
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}


# FFM model 对数据值特征记录
def denseFeature(feat):
    return {'feat': feat}


def create_criteo_dataset(t, file_path, test_size=0.3):
    # 数值特征和类别特征名
    data = pd.read_csv(file_path, sep='\t', header=None, names=['label'] + dense_features + sparse_features)

    # 缺失值填充
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    # 归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    if t == 'fm':
        data = pd.get_dummies(data)
    elif t in ['ffm', 'DeepCrossing']:
        # LabelEncoding编码
        for col in sparse_features:
            data[col] = LabelEncoder().fit_transform(data[col]).astype(int)

    # 数据集划分
    X = data.drop(['label'], axis=1).values
    y = data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return (X_train, y_train), (X_test, y_test)


def features_dict(file_path, embed_dim=8):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['label'] + dense_features + sparse_features)

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, data[feat].nunique()+1, embed_dim) for feat in sparse_features]]
    return feature_columns
