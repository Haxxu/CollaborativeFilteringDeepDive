from flask import Flask, jsonify

app = Flask(__name__)

import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import os
from surprise import Dataset, Reader
from collections import defaultdict

from reco_utils.common.general_utils import invert_dictionary

import itertools
from surprise import accuracy

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

from surprise import SVD

print('hello')

emotion_path = os.path.expanduser(
    'D:\\Programming\\Web\projects\\journey-connect-backend\\recommendations\\emotions.csv')

df_emotion = pd.read_csv(emotion_path, sep=',', names=['userID', 'postID', 'emotionType', 'timestamp'])
# print(df_emotion.head())

reader = Reader(line_format='user item rating timestamp', sep=',')
data = Dataset.load_from_file(emotion_path, reader=reader)

emotions_list = [(x, y, z) for x, y, z in zip(df_emotion['userID'], df_emotion['postID'], df_emotion['emotionType'])]

emotions = defaultdict(int)
rankings = defaultdict(int)
for row in emotions_list:
    postID = str(row[1])
    emotions[postID] += 1

# print(emotions)

rank = 1
for postID, emotionCount in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
    rankings[postID] = rank
    rank += 1

# print(rankings)

util_df = pd.pivot_table(data=df_emotion, values='emotionType', index='userID', columns='postID', aggfunc='mean',
                         fill_value=0)


# print(util_df)

# df_temp = pd.DataFrame(df_emotion.groupby('postID').mean()['emotionType'])
# df_temp['count'] = pd.DataFrame(df_emotion.groupby('postID').count()['emotionType'])

fullTrainSet = data.build_full_trainset()
fullAntiTestSet = fullTrainSet.build_anti_testset()


def GetAntiTestSetForUser(testSubject='b044bae7e38fcfdddbed2b33'):
    trainset = fullTrainSet
    fill = trainset.global_mean
    anti_test_set = []
    u = trainset.to_inner_uid(testSubject)
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_test_set += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for i in trainset.all_items() if
                      i not in user_items]

    return anti_test_set

alg = SVD()
alg.fit(fullTrainSet)
print('hellodfs')

@app.route('/recommend/<string:user_id>')
def recommend(user_id):
    try:
        print(GetAntiTestSetForUser(user_id))
        predictions = alg.test(GetAntiTestSetForUser(user_id))
        recommendations = []
        for userID, postID, actualEmotion, estimatedEmotion, _ in predictions:
            strPostID = str(postID)
            recommendations.append((strPostID, estimatedEmotion))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        print(e)
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
