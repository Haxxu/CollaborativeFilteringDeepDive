from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class uuCF(object):
    def __init__(self, Y_data, k, sim_func=cosine_similarity):
        self.Y_data = Y_data
        self.k = k
        self.sim_func = sim_func
        self.Ybar = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        print(self.n_users)

    def fit(self):
        users = self.Y_data[:, 0]
        self.Ybar = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        print(self.Y_data[:, 0])
        for n in range(5):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]
            print(ids, item_ids, ratings, '\n\n')
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0
            self.Ybar[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar[:, 2],
                                       (self.Ybar[:, 1], self.Ybar[:, 0])),
                                      (self.n_items, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)

    def pred(self, u, i):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]

        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns]
        r = self.Ybar[i, users_rated_i[nns]]
        eps = 1e-8
        return (r*nearest_s).sum() / (np.abs(nearest_s).sum()+eps) + self.mu[u]


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('../data/ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('../data/ml-100k/ua.test', sep='\t', names=r_cols)

rate_train = ratings_base.values
rate_test = ratings_test.values


# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1
# rs = uuCF(rate_train, k = 40)
# rs.fit()

rate_train = rate_train[:, [1, 0, 2]]
rate_test = rate_test[:, [1, 0, 2]]
rs = uuCF(rate_train, k = 40)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1])
    SE += (pred - rate_test[n, 2])**2

RMSE = np.sqrt(SE/n_tests)
print('User-user CF, RMSE = ', RMSE)


