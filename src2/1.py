# Ignore  the warnings
import warnings

from sklearn.utils import column_or_1d


# Data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import os
from surprise import Dataset, Reader
from collections import defaultdict

from reco_utils.common.general_utils import invert_dictionary

# Accuracy
import itertools
from surprise import accuracy

# Method to measure the accuracy of recommendation model
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline
from operator import itemgetter
import heapq

# Predict
from surprise import KNNBasic
from surprise import SVD

from surprise import KNNWithMeans

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

rating_path = os.path.expanduser('../data/ml-100k/u.data')
# rating_path = os.path.expanduser('/home/nnminh1/Desktop/Python/CollaborativeFilteringDeepDive/data/ml-100k/u.data')

item_path = os.path.expanduser('../data/ml-100k/u.item')
# item_path = os.path.expanduser('/home/nnminh1/Desktop/Python/CollaborativeFilteringDeepDive/data/ml-100k/u.item')

df_rating = pd.read_csv(rating_path, sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
# print(df_rating.head())

reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(rating_path, reader=reader)

ratings_list = [(x, y, z) for x, y, z in zip(df_rating['userId'],
                                             df_rating['movieId'],
                                             df_rating['rating'])]

ratings = defaultdict(int)
rankings = defaultdict(int)
for row in ratings_list:
    movieID = int(row[1])
    ratings[movieID] += 1


rank = 1
for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
    rankings[movieID] = rank
    rank += 1

df_item = pd.read_csv(item_path, sep='|', encoding='utf-16', header=None)
df_item.rename(columns={0: 'movieId'}, inplace=True)
# print(df_item.head())


def getMovieName(movieID):
    return df_item.loc[movieID, 1]


df = pd.merge(df_rating, df_item[['movieId', 1]], on='movieId')
# print(df.head())

util_df = pd.pivot_table(data=df, values='rating', index='userId', columns='movieId', aggfunc='mean', fill_value=0)
# print(util_df)


def surprise_trainset_to_df(trainset, col_user='uid', col_item='iid', col_rating='rating'):
    df = pd.DataFrame(trainset.all_ratings(), columns=[col_user, col_item, col_rating])
    map_user = trainset._inner2raw_id_users if trainset._inner2raw_id_users is not None else invert_dictionary(trainset._raw2inner_id_users)
    map_item = trainset._inner2raw_id_items if trainset._inner2raw_id_items is not None else invert_dictionary(trainset._raw2inner_id_items)
    df[col_user] = df[col_user].map(map_user)
    df[col_item] = df[col_item].map(map_item)
    return df


# df.info()
# print(df.nunique())

# print(df.head())
temp = pd.DataFrame(df.groupby(1).mean()['rating'])
temp['count'] = pd.DataFrame(df.groupby(1).count()['rating'])
# print(temp.head())
# print('Min: \n', temp.min(), '\nMax: \n', temp.max())


# Checking the distribution of number of rating vs appearances
plt.figure(figsize=(10, 6))

# plt.hist(temp['count'], bins=70)
# plt.xlabel('No. of user ratings')
# plt.ylabel('The appearances for every rating')
# plt.title('Distribution of no. of ratings')
# plt.show()

# Distribution of ratings
# plt.hist(temp['rating'], bins=70)
# plt.xlabel('Avg. rating')
# plt.ylabel('No. of rating')
# plt.show()

# sns.jointplot(x=temp['rating'], y=temp['count'], data=temp, alpha=0.5)


class RecommenderMetrics:
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimunRating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimunRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break

            if (hit):
                hits += 1

            total += 1

        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit):
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0):
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1 - S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n


fullTrainSet = data.build_full_trainset()
fullAntiTestSet = fullTrainSet.build_anti_testset()

testUnwatched = list()
items = util_df.columns
for item in items:
    users = util_df.loc[util_df[item] == 0].index
    for user in users:
        testUnwatched.append((str(user), str(item), 0))


temp305 = util_df[305]
print(temp305)

# 75 train , 25 test
trainSet, testSet = train_test_split(data, test_size=.25, random_state=1)

# leave one out train/test
LOOCV = LeaveOneOut(n_splits=1, random_state=1)
for train, test in LOOCV.split(data):
    LOOCVTrain = train
    LOOCVTest = test

# anti-test-set
LOOCVAntiTestSet = LOOCVTrain.build_anti_testset()

# Compute similarty matrix between items so we can measure diversity
sim_options = {'name': 'cosine', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)

# shows the Top10 most similar movies are recommended to
#the active User 305


def GetAntiTestSetForUser(testSubject='305'):
    trainset = fullTrainSet
    fill = trainset.global_mean
    anti_testset = []
    u = trainset.to_inner_uid(testSubject)
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset


# alg = SVD()
# alg.fit(trainSet)
# predictions = alg.test(testSet)
# print('RMSE: ', RecommenderMetrics.RMSE(predictions))
# print('MAE:', RecommenderMetrics.MAE(predictions))


# # Evaluate top-10 with Leave One Out testing
# alg = SVD()
# alg.fit(LOOCVTrain)
# leftOutPredictions = alg.test(LOOCVTest)
# # Build predictions for all ratings not in the training set
# allPredictions = alg.test(LOOCVAntiTestSet)
# # Compute top 10 recs for each user
# topNPredicted = RecommenderMetrics.GetTopN(allPredictions, 10)
#
# # See how often we recommended a movie the user actually rated
# print("HR: ", RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   )
# # See how often we recommended a movie the user actually liked
# print("cHR: ", RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions))
# # Compute ARHR
# print("ARHR: ", RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions))


# # Computing recommendations with full data set
# alg = SVD()
# alg.fit(fullTrainSet)
# predictions = alg.test(testUnwatched)
# predictions = pd.DataFrame(predictions)
# allPredictions = alg.test(fullAntiTestSet)
# topNPredicted = RecommenderMetrics.GetTopN(allPredictions, 10)
#
# # Analyzing coverage, diversity, and novelty
# # user coverage with a minimum predicted rating of 4.0:
# print("Coverage: ", RecommenderMetrics.UserCoverage(topNPredicted, fullTrainSet.n_users, ratingThreshold=4.0))
# # Measure diversity of recommendations:
# print("Diversity: ", RecommenderMetrics.Diversity(topNPredicted, simsAlgo))
#
# # Measure novelty (average popularity rank of recommendations):
# print("Novelty: ", RecommenderMetrics.Novelty(topNPredicted,  rankings))


# temp0 = pd.DataFrame(predictions)
# temp0 = temp0[['uid', 'iid', 'est']]
# temp0.rename(columns={'uid': 'userId', 'iid': 'movieId', 'est': 'rating'}, inplace=True)
# temp0 = temp0.astype('int32')



# Building recommendation model
alg = SVD()
alg.fit(fullTrainSet)
# Computing recommendations
print('get anti test set for user 2: ', GetAntiTestSetForUser('2'))
predictions = alg.test(GetAntiTestSetForUser('1'))
print('predictions: ', predictions)
recommendations = []

print("\nWe recommend:")
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

recommendations.sort(key=lambda x: x[1], reverse=True)

for ratings in recommendations[:10]:
    print(getMovieName(ratings[0]), ratings[0], ratings[1])
