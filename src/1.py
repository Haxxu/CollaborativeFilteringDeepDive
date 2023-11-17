from fastai.collab import *
from fastai.tabular.all import *
set_seed(42)

# import pandas as pd


# import chardet
# with open('data/ml-100k/u.item', 'rb') as f:
#     result = chardet.detect(f.read())
# print(result)

ratings = pd.read_csv('../data/ml-100k/u.data', delimiter='\t', header=None,
                      names=['user', 'movie', 'rating', 'timestamp'])
# print(ratings.head())

movies = pd.read_csv('../data/ml-100k/u.item', delimiter='|', encoding='utf-16',
                     header=None, usecols=(0, 1),
                     names=('movie', 'title'))
# print(movies.head())

ratings = ratings.merge(movies)
# print(ratings.head())

dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()

n_users = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factor = torch.randn(n_users, n_factors)
movie_factor = torch.randn(n_movies, n_factors)


