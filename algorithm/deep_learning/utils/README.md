# utils

## movielens data
[movielens数据获取](http://files.grouplens.org/datasets/movielens/)
下载ml-1m.zip至data目录中进行解压；

## criteo_dataset
criteo数据的获取和处理

## feature_column
### SparseFeat
* name: 特征名
* vocabulary_size: 特征类型数
* embedding_dim: embedding dim
* use_hash: 是否使用hash
* vocabulary_path: 
* dtype: 数据类型
* embedding_initializer: embedding的初始值
* embedding_name: embedding的名字
* group_name: group的name
* trainable: 是否可训练

### VarLenSparseFeat
原SparseFeat的参数 + 
* maxlen: 序列最大长度
* combiner: 合成器
* length_name: user_behavior_seq_length name
* weight_name: 
* weight_norm:

### DenseFeat
* name: 特征名
* dimension: 维度
* dtype: 数据类型
* transform_fn: use transform values of the feature

## inputs
### create_embedding_dict
遍历sparse_feature_columns和varlen_sparse_feature_columns进行Embedding，输出字典

### embedding_lookup & varlen_embedding_lookup
输出每个特征的lookup_idx的dict

### get_dense_input
get dense feature list

# 代码结构优化方案
* feature_column 单独作为特征处理；
* inputs 主要存储数据输入时的数据；
* dataset 获取不同数据源的数据并进行初步处理的函数集；
  * 对原始数据进行初步处理后，输出训练数据(X, y)和测试数据；（初步处理包括：dense feats / sparse feats / seq feats等）
  * 输出feature_dict（特征的处理方式？是不是需要单独记录呢？）
  
* seq_dataset 生成数据集中的序列数据；
* comp_fit 编译和训练（考虑删除）因为参数输入可能会存在很多种不同的选择；