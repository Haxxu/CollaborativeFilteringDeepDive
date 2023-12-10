import random

from flask import Flask, jsonify

app = Flask(__name__)

import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import os
from surprise import Dataset, Reader
from collections import defaultdict
import matplotlib.pyplot as plt

from reco_utils.common.general_utils import invert_dictionary

import itertools
from surprise import accuracy

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline, KNNBasic, KNNWithMeans

from surprise import SVD

emotion_path = os.path.expanduser(
    'D:\\Programming\\Web\projects\\journey-connect-backend\\recommendations\\emotions.csv')

df_emotion = pd.read_csv(emotion_path, sep=',', names=['userID', 'postID', 'emotionType'])
# print("head", df_emotion.head())

reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file(emotion_path, reader=reader)

# print("data", data)

emotions_list = [(x, y, z) for x, y, z in zip(df_emotion['userID'], df_emotion['postID'], df_emotion['emotionType'])]

emotions = defaultdict(int)
rankings = defaultdict(int)
for row in emotions_list:
    postID = str(row[1])
    emotions[postID] += 1

# print("emotions items\n", emotions.items())

rank = 1
for postID, emotionCount in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
    rankings[postID] = rank
    rank += 1

# print("rankings", rankings)

util_df = pd.pivot_table(data=df_emotion, values='emotionType', index='userID', columns='postID', aggfunc='mean',
                         fill_value=0)

# print(util_df.head())

# df_temp = pd.DataFrame(df_emotion.groupby('postID').mean()['emotionType'])
# df_temp['count'] = pd.DataFrame(df_emotion.groupby('postID').count()['emotionType'])

fullTrainSet = data.build_full_trainset()
fullAntiTestSet = fullTrainSet.build_anti_testset()


def GetAntiTestSetForUser1(testSubject='b044bae7e38fcfdddbed2b33', all_items=None):
    trainset = fullTrainSet
    fill = trainset.global_mean
    anti_test_set = []

    if all_items is None:
        all_items = trainset.all_items()

    u = trainset.to_inner_uid(testSubject)
    user_items = set([j for (j, _) in trainset.ur[u]])
    # print('user_items: ', trainset.ur[u])
    anti_test_set += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for i in all_items if i not in user_items]

    print('anti test set: ', anti_test_set)
    return anti_test_set


alg = KNNWithMeans()
alg.fit(fullTrainSet)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        global alg, fullTrainSet

        # Load the updated emotions CSV file
        updated_emotion_path = 'D:\\Programming\\Web\projects\\journey-connect-backend\\recommendations\\emotions.csv'
        df_updated_emotion = pd.read_csv(updated_emotion_path, sep=',', names=['userID', 'postID', 'emotionType', 'timestamp'])

        # Update the training set with the new data
        new_data = Dataset.load_from_file(updated_emotion_path, reader=reader)
        fullTrainSet = new_data.build_full_trainset()

        # Retrain the model with the updated training set
        alg.fit(fullTrainSet)

        return jsonify({'message': 'Model successfully retrained with updated data'})

    except Exception as e:
        print(e)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

@app.route('/recommend/<string:user_id>')
def recommend(user_id):
    try:
        all_items = fullTrainSet.all_items()
        anti_test_set = GetAntiTestSetForUser1(user_id, all_items)
        # print('all items: ', all_items)
        # print('full train set: ', fullTrainSet)
        # print('anti test set: ', anti_test_set)

        if not anti_test_set:
            return jsonify({'recommendations': []})  # Return empty recommendations for invalid user_id

        # print(anti_test_set)
        predictions = alg.test(anti_test_set)
        # predictions = alg.predict(anti_test_set)
        recommendations = []
        print('predict: ', predictions)
        for userID, postID, actualEmotion, estimatedEmotion, _ in predictions:
            strPostID = str(postID)
            recommendations.append({'post_id': strPostID, 'score': estimatedEmotion})

        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        print(e)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


@app.route('/train-test1', methods=['POST'])
def train_test_model1():
    global alg, fullTrainSet, testset, mse, rmse

    try:
        # Load the updated emotions CSV file
        updated_emotion_path = 'D:\\Programming\\Web\projects\\journey-connect-backend\\recommendations\\emotions.csv'
        df_updated_emotion = pd.read_csv(updated_emotion_path, sep=',', names=['userID', 'postID', 'emotionType', 'timestamp'])

        # Update the training set with the new data
        new_data = Dataset.load_from_file(updated_emotion_path, reader=reader)
        trainset, testset = train_test_split(new_data, test_size=0.2, random_state=42)
        fullTrainSet = new_data.build_full_trainset()  # Use build_full_trainset on new_data instead of trainset

        # Retrain the model with the updated training set
        alg.fit(fullTrainSet)

        # Make predictions on the test set
        predictions = alg.test(testset)

        # Calculate MSE and RMSE
        mse = accuracy.mse(predictions)
        rmse = accuracy.rmse(predictions)

        return jsonify({'message': 'Model successfully retrained with updated data', 'mse': mse, 'rmse': rmse})

    except Exception as e:
        print(e)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


# @app.route('/train-test', methods=['POST'])
# def train_test_model():
#     global alg, fullTrainSet, testset, mse, rmse
#
#     try:
#         # Load the updated emotions CSV file
#         updated_emotion_path = 'D:\\Programming\\Web\projects\\journey-connect-backend\\recommendations\\emotions.csv'
#         df_updated_emotion = pd.read_csv(updated_emotion_path, sep=',', names=['userID', 'postID', 'emotionType', 'timestamp'])
#
#         # Update the training set with the new data
#         new_data = Dataset.load_from_file(updated_emotion_path, reader=reader)
#         trainset, testset = train_test_split(new_data, test_size=0.2, random_state=42)
#         fullTrainSet = new_data.build_full_trainset()  # Use build_full_trainset on new_data instead of trainset
#
#         # Retrain the model with the updated training set
#         alg.fit(trainset)
#
#         # Make predictions on the test set
#         predictions = alg.test(testset)
#
#         # Calculate MSE and RMSE
#         mse = accuracy.mse(predictions)
#         rmse = accuracy.rmse(predictions)
#
#         # Calculate the percentage accuracy
#         rating_range = fullTrainSet.rating_scale[1] - fullTrainSet.rating_scale[0]
#         print(fullTrainSet.rating_scale)
#         percentage_accuracy = 100 * (1 - (rmse / rating_range))
#
#         return jsonify({
#             'message': 'Model successfully retrained with updated data',
#             'percentage_accuracy': percentage_accuracy
#         })
#
#     except Exception as e:
#         print(e)
#         return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


@app.route('/train-test', methods=['POST'])
def evaluate():
    global alg, fullTrainSet, testset, mse, rmse

    try:
        updated_emotion_path = 'D:\\Programming\\Web\\projects\\journey-connect-backend\\recommendations\\emotions.csv'
        new_data = Dataset.load_from_file(updated_emotion_path, reader=reader)
        rmse_values = []
        for _ in range(10):
            random_state = random.randint(1, 1000)
            trainset, testset = train_test_split(new_data, test_size=0.2, random_state=random_state)
            fullTrainSet = new_data.build_full_trainset()
            alg.fit(fullTrainSet)
            predictions = alg.test(testset)
            rmse = accuracy.rmse(predictions)
            rmse_values.append(rmse)

        average_rmse = sum(rmse_values) / len(rmse_values)
        rating_range = 6
        percentage_accuracy = 100 * (1 - (average_rmse / rating_range))

        return jsonify({
            'average_rmse': average_rmse,
            'percentage_accuracy': percentage_accuracy
        })

    except Exception as e:
        print(e)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
