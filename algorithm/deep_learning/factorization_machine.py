# FM算法代码实现
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def sigmoid(z):
    """二分类输出非线性映射"""
    return 1 / (1 + np.exp(-z))


def logit(y, y_hat):
    """计算logit损失函数"""
    if y_hat == 'nan':
        return 0
    return np.log(1 + np.exp(-y * y_hat))


def df_logit(y, y_hat):
    """计算logit损失函数的外层偏导（不含y_hat的一阶偏导）"""
    return sigmoid(-y * y_hat) * (-y)


class FactorizationMachine:
    def __init__(self, k=5, learning_rate=0.01, iter_num=2):
        self.w0 = None
        self.W = None
        self.V = None
        self.k = k
        self.alpha = learning_rate
        self.iter = iter_num

    def _fm(self, Xi):
        """FM模型方程：LR线性组合 + 交叉项组合"""
        interaction = np.sum((Xi.dot(self.V)) ** 2 - (Xi ** 2).dot(self.V ** 2))
        y_hat = self.w0 + Xi.dot(self.W) + interaction / 2
        return y_hat[0]

    def _fm_sgd(self, X, y):
        """SGD更新FM模型的参数，w0, W, V"""
        m, n = np.shape(X)
        # 初始化参数
        self.w0 = 0
        self.W = np.random.uniform(0, 1, size=(n, 1))
        self.V = np.random.uniform(0, 1, size=(n, self.k))  # 初始化隐权重向量V=(n,k)~N(0,1)

        for it in range(self.iter):
            total_loss = 0  # 当前迭代模型的损失值
            for i in range(m):  # 遍历训练集
                y_hat = self._fm(X[i])

                total_loss += logit(y[i], y_hat)
                d_loss = df_logit(y[i], y_hat)  # 计算logit损失函数的外层偏导

                d_loss_w0 = d_loss * 1  # 公式中w0求导，计算复杂度为O(1)
                self.w0 = self.w0 - self.alpha * d_loss_w0  # 更新w0

                for j in range(n):
                    if X[i, j] != 0:
                        d_loss_wj = d_loss * X[i, j]  # wi求导，复杂度为O(n)
                        self.W[j] = self.W[j] - self.alpha * d_loss_wj  # 更新W[j]
                        for f in range(self.k):  # vjf求导，复杂度为O(kn)
                            d_loss_vjf = d_loss * (X[i, j] * (X[i].dot(self.V[:, f])) - self.V[j, f] * X[i, j] ** 2)
                            self.V[j, f] = self.V[j, f] - self.alpha * d_loss_vjf  # 更新V[j]
            print('iter={}, loss={:.4f}'.format(it + 1, total_loss / m))

        return self

    def _fm_predict(self, X):
        pre, threshold = [], 0.5
        for i in range(X.shape[0]):
            y_hat = self._fm(X[i])
            pre.append(-1 if sigmoid(y_hat) < threshold else 1)  # 判断预测结果
        return np.array(pre)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
            y = np.array(y)
        return self._fm_sgd(X, y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        return self._fm_predict(X)


if __name__ == '__main__':
    r_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv('../data/ml-100k/u.data', sep='\t', header=None, names=r_names, engine='python')
    print(df.shape)
    print(df.head())
    df['rating'] = df['rating'].map(lambda x: -1 if x >= 3 else 1)
    # one-hot encoder
    columns = ['user_id', 'movie_id']
    for i in columns:
        get_dummy_feature = pd.get_dummies(df[i])
        df = pd.concat([df, get_dummy_feature], axis=1)
        df = df.drop(i, axis=1)
    df = df.drop(['timestamp'], axis=1)
    X = df.drop('rating', axis=1)
    y = df['rating']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    # fm model
    print(X_train.shape)
    t1 = time.time()
    model = FactorizationMachine(k=10, learning_rate=0.001, iter_num=2)
    model.fit(X_train, y_train)
    t2 = time.time()
    print('cost time: ', t2 - t1)

    y_pred = model.predict(X_train)
    print('train data roc: {:.2f}'.format(roc_auc_score(y_train.values, y_pred)))
    print('train data confusion matrix: \n', confusion_matrix(y_train.values, y_pred))

    y_val_pred = model.predict(X_val)
    print('val data roc: {:.2f}'.format(roc_auc_score(y_val.values, y_val_pred)))
    print('val data confusion matrix: \n', confusion_matrix(y_val.values, y_val_pred))
