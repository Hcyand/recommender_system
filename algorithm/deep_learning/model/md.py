# -*- coding: utf-8 -*-            
# @Time : 2022/8/22 22:43
# @Author : Hcyand
# @FileName: md.py

# 矩阵分解算法 Matrix Decomposition
import numpy as np
from math import pow
from layer.utils import top_k


class MatrixFactorization:
    def __init__(self, users, movies, users_movies, steps=5000, alpha=0.0002, beta=0.02):
        """
        :param users: 用户名列表
        :param movies: 电影名列表
        :param users_movies: 用户-电影共现矩阵
        :param steps: 步数
        :param alpha: 步长
        :param beta: 系数
        """
        self.users = users
        self.movies = movies
        self.users_movies = users_movies
        self.steps = steps
        self.alpha = alpha
        self.beta = beta

    def matrix_factorization(self, r, p, q, hide):
        """
        :param r: 评分矩阵
        :param p: 用户矩阵
        :param q: 物品矩阵
        :param hide: 隐向量维度
        :return:
        """
        q = q.T
        loss_log = []
        err_old = 0  # 记录前一个的loss
        for step in range(self.steps):
            for i in range(len(r)):
                for j in range(len(r[i])):
                    if r[i][j] > 0:
                        eij = r[i][j] - np.dot(p[i, :], q[:, j])
                        for k in range(hide):
                            p[i][k] = p[i][k] + self.alpha * (2 * eij * q[k][j] - self.beta * p[i][k])
                            q[k][j] = q[k][j] + self.alpha * (2 * eij * p[i][k] - self.beta * q[k][j])
            err = 0
            for i in range(len(r)):
                for j in range(len(r[i])):
                    if r[i][j] > 0:
                        err = err + pow(r[i][j] - np.dot(p[i, :], q[:, j]), 2)
                        for k in range(hide):
                            err = err + (self.beta / 2) * (pow(p[i][k], 2) + pow(q[k][j], 2))
            loss_log.append(err)
            if step == 0:
                err_old = err
                continue
            if abs(err_old - err) < 1e-10:
                break
            else:
                err_old = err
            if err < 0.001:
                break

        return p, q.T, loss_log

    def rec_movies(self, user, k, p, q):
        """
        :param user: 用户名
        :param k: top k推荐
        :param p: 用户隐向量矩阵，n*k
        :param q: 物品隐向量矩阵,m*k
        :return:
        """
        m = self.users.index(user)
        looked = self.users_movies[m]
        u_list = p[m]
        scores = np.dot(u_list, q.T)
        candidate = [[self.movies[i], scores[i]] for i in range(len(looked)) if looked[i] == 0]
        rec = top_k(candidate, k)
        return rec


if __name__ == '__main__':
    users_1 = ["User1", "User2", "User3", "User4", "User5"]
    movies_1 = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    # 电影存在评分的情况，评分在1~5之间，没有看过的电影为0
    users_movies_1 = [
        [3, 4, 5, 0, 3, 0, 0],  # User1
        [0, 4, 2, 0, 0, 5, 0],
        [1, 0, 3, 5, 3, 3, 2],
        [3, 3, 5, 1, 2, 0, 0],
        [5, 5, 0, 2, 0, 4, 5]]
    R = np.array(users_movies_1)
    K = 2
    # 随机初始化隐向量
    P = np.random.rand(len(R), K)
    Q = np.random.rand(len(R[0]), K)
    d = MatrixFactorization(users_1, movies_1, users_movies_1)
    u_matrix, i_matrix, loss = d.matrix_factorization(R, P, Q, K)
    res = d.rec_movies('User1', 2, u_matrix, i_matrix)
    print(res)
