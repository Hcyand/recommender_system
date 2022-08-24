# -*- coding: utf-8 -*-            
# @Time : 2022/8/22 22:42
# @Author : Hcyand
# @FileName: lr.py

# 逻辑回归算法 Logistic Regression
import numpy as np
import time
import pandas as pd


class LogisticRegression:
    def __init__(self, n):
        self.STOP_ITER = 0
        self.STOP_COST = 1
        self.STOP_GRAD = 2
        self.n = n

    @staticmethod
    def sigmoid(z):
        """sigmoid函数"""
        return 1 / (1 + np.exp(-z))

    def model(self, X, theta):
        """
        预测函数
        :param X: 特征值
        :param theta: 参数
        :return: 预测值
        """
        return self.sigmoid(np.dot(X, theta.T))

    def cost(self, X, y, theta):
        """损失函数/目标函数"""
        left = np.multiply(y, np.log(self.model(X, theta)))
        right = np.multiply(1 - y, np.log(1 - self.model(X, theta)))
        return -np.sum(left + right) / len(X)

    def gradient(self, X, y, theta):
        """计算梯度，根据梯度公式"""
        grad = np.zeros(theta.shape)
        error = (self.model(X, theta) - y).ravel()  # ravel将数组拉成一维数组
        for j in range(len(theta.ravel())):
            term = np.multiply(error, X[:, j])
            grad[0, j] = np.sum(term) / len(X)
        return grad

    def stop_criterion(self, t, value, threshold):
        """停止迭代的条件：迭代次数、损失值变化、梯度变化"""
        if t == self.STOP_ITER:
            return value > threshold
        elif t == self.STOP_COST:
            return abs(value[-1] - value[-2]) < threshold
        elif t == self.STOP_GRAD:
            return np.linalg.norm(value) < threshold

    @staticmethod
    def shuffle_data(data):
        np.random.shuffle(data)
        cols = data.shape[1]
        train = data[:, 0:cols - 1]
        label = data[:, cols - 1:]
        return train, label

    def descent(self, data, theta, batch_size, stop_type, thresh, alpha):
        """梯度下降求解"""
        init_time = time.time()
        i = 0
        k = 0
        train, label = self.shuffle_data(data)  # 打乱数据
        # grad = np.zeros(theta.shape)  # 梯度初始化
        costs = [self.cost(train, label, theta)]  # 初始损失
        value = 0

        while True:
            # 批量计算梯度
            grad = self.gradient(train[k:k + batch_size], label[k:k + batch_size], theta)
            k += batch_size
            # 当k>n时，打乱数据，k赋值0
            if k >= self.n:
                k = 0
                train, label = self.shuffle_data(data)
            theta = theta - alpha * grad  # 模型参数更新公式
            costs.append(self.cost(train, label, theta))  # 记录当前损失
            i += 1

            # 停止迭代的条件判断
            if stop_type == self.STOP_ITER:
                value = i
            elif stop_type == self.STOP_COST:
                value = costs
            elif stop_type == self.STOP_GRAD:
                value = grad
            if self.stop_criterion(stop_type, value, thresh):
                break
        # 输出参数、迭代次数、损失列表、当前梯度、消耗时间
        return theta, i - 1, costs, grad, time.time() - init_time

    def predict(self, X, theta):
        return [1 if x >= 0.5 else 0 for x in self.model(X, theta)]


if __name__ == '__main__':
    path = '../data/LogiReg_data.txt'
    pdData = pd.read_csv(path, header=None, names=['exam_1', 'exam_2', 'admitted'])
    # 处理数据
    pdData.insert(0, 'Ones', 1)
    orig_data = pdData.values  # 转化为正确的输入形式，array
    theta = np.zeros([1, 3])
    scaled_X = orig_data[:, :3]
    y = orig_data[:, 3]
    print(pdData.head())
    print('data shape: ', pdData.shape)
    print('theta: ', theta)

    # 预测准确率
    lr = LogisticRegression(100)
    # 输出参数、迭代次数、损失列表、当前梯度、消耗时间
    theta, times, loss_list, grad_now, cost_time = lr.descent(orig_data, theta, 100, 1, 0.000001, 0.001)
    predictions = lr.predict(scaled_X, theta)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))
