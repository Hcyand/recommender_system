# 协同过滤算法python实现
import numpy as np
import heapq


##########################################################
# 计算物品之间相似度
# 欧氏距离
def euclidean(list_1, list_2):
    res = np.sqrt(((np.array(list_1) - np.array(list_2)) ** 2).sum())
    return res


# 皮尔逊相关系数
def pearson(list_1, list_2):
    res = np.sqrt(((np.array(list_1) - np.array(list_2)) ** 2).sum())
    return res


# 输出top k列表，利用最大/小堆实现
def top_k(candidate, k):
    """
    :param candidate: list 候选数据集列表，存储形式[[user,score],[...],...]
    :param k: int 选取top_k候选元素
    :return: list
    """
    q = []
    heapq.heapify(q)
    for i in range(len(candidate)):
        if len(q) < k:  # 长度不足时直接添加
            heapq.heappush(q, [-candidate[i][1], candidate[i][0]])
        else:
            tmp = heapq.heappop(q)
            if tmp[0] < -candidate[i][1]:  # 判断是否添加
                heapq.heappush(q, [-candidate[i][1], candidate[i][0]])
            else:
                heapq.heappush(q, tmp)
    res = sorted(q, key=lambda x: -x[0])
    return res


def calculate_sim(arr, t):
    m = len(arr)
    dp = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if t == 'euc':
                dp[i][j] = euclidean(arr[i], arr[j])
            elif t == 'pea':
                dp[i][j] = pearson(arr[i], arr[j])
    return dp


class ItemCF:
    def __init__(self, users, movies, users_movies):
        self.users = users
        self.movies = movies
        self.users_movies = users_movies
        self.scores = self.all_movies_sim('euc')

    # 统计所有电影之间相似度，存入数组中
    def all_movies_sim(self, t):
        arr_movies = np.array(self.users_movies).transpose().tolist()
        dp = calculate_sim(arr_movies, t)
        return dp

    # 相似电影推荐
    def rec_movies(self, user, k):
        m = self.users.index(user)  # 用户user下标
        looked = self.users_movies[m]  # 用户浏览过的电影队列
        assert len(looked) == len(self.scores[m]), 'Unequal lengths'
        candidate = [[self.movies[i], round(self.scores[m][i], 3)] for i in range(len(looked)) if looked[i] == 0]
        res = top_k(candidate, k)
        return res  # 输出电影列表


# 计算用户之间相似度
class UserCF:
    def __init__(self, users, movies, users_movies):
        self.users = users
        self.movies = movies
        self.users_movies = users_movies
        self.scores = self.all_users_sim('euc')

    def all_users_sim(self, t):
        dp = calculate_sim(self.users_movies, t)
        return dp

    # 先输出最相似的k1个用户，再利用权重输出没看过电影的得分，输出前k2个电影作为推荐
    def rec_movies(self, user, k1, k2):
        m = self.users.index(user)
        # 输出top k1个用户
        candidate_user = [[self.users[i], round(self.scores[m][i], 3)]
                          for i in range(len(self.users)) if self.users[i] != user]
        res_user = top_k(candidate_user, k1)
        # 输出top k2个电影
        # [[用户下标，权重（相似度倒数）],...]
        tmp = [[self.users.index(res_user[i][1]), 1 / -res_user[i][0]] for i in range(len(res_user))]
        score = [0] * len(self.movies)  # 初始分数为0
        for x in tmp:
            score = [score[i] + self.users_movies[x[0]][i] * x[1] for i in range(len(score))]  # 利用权重计算总分数
        looked = self.users_movies[m]  # 用户浏览过的电影队列
        candidate_score = [[self.movies[i], -score[i]] for i in range(len(looked)) if looked[i] == 0]
        res = top_k(candidate_score, k2)
        return res


if __name__ == '__main__':
    users_1 = ["User1", "User2", "User3", "User4", "User5"]
    movies_1 = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    # 对用户和电影进行编号
    users_movies_1 = [
        [1, 1, 1, 0, 1, 0, 0],  # User1
        [0, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]]
    d1 = UserCF(users_1, movies_1, users_movies_1)
    print(d1.rec_movies('User1', 2, 2))
    d2 = ItemCF(users_1, movies_1, users_movies_1)
    print(d2.rec_movies('User2', 2))
