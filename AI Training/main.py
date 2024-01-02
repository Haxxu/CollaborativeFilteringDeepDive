import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval


dtypes = {'release_date': str, 'id': str, 'movieId': str, 'title': str, }

credit_df = pd.read_csv('./data/credits.csv', dtype=dtypes)
movie_df = pd.read_csv('./data/movies_metadata.csv', dtype=dtypes, low_memory=False)
rating_df = pd.read_csv('./data/ratings_small.csv', dtype=dtypes, low_memory=False)
keyword_df = pd.read_csv('./data/keywords.csv', dtype=dtypes, low_memory=False)

movie_df = movie_df.merge(credit_df, on='id')
# movie_df = movie_df.merge(rating_df, left_on='id', right_on='movieId', how='inner')
movie_df = movie_df.merge(keyword_df, left_on='id', right_on='id', how='inner')
# movie_df['rating_count'] = movie_df.groupby('id')['rating'].transform('count')
# movie_df['rating_average'] = movie_df.groupby('id')['rating'].transform('mean')
# movie_df = movie_df.drop_duplicates(subset='id')

C = movie_df['vote_average'].mean()
m = movie_df['vote_count'].quantile(0.9)

q_movies_df = movie_df.copy().loc[movie_df['vote_count'] >= m]


# print(movie_df[['title', 'rating_count', 'rating_average']])

# print('C = ', C, ', m = ', m)
# print(q_movies_df.shape)


##########################################################################
# Demographic Filtering
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula]
    return (v/(v+m) * R) + (m/(m+v) * C)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies_df['score'] = q_movies_df.apply(weighted_rating, axis=1)

# Sort movies based on score calculated above
q_movies_df = q_movies_df.sort_values('score', ascending=False)

# Print the top 15 movies
# print(q_movies_df[['title', 'vote_count', 'vote_average', 'score']].head(10))

# popular_movie_df = movie_df.copy()
# popular_movie_df['popularity'] = pd.to_numeric(popular_movie_df['popularity'], errors='coerce')
# popular_movie_df['popularity'] = popular_movie_df['popularity'].fillna(0.0)
# popular_movie_df = movie_df.sort_values('popularity', ascending=False)
# plt.figure(figsize=(12, 4))
# plt.barh(popular_movie_df['title'].astype(str).head(6), popular_movie_df['popularity'].head(6), align='center', color='skyblue')
# # plt.gca().invert_yaxis()
# plt.xlabel("Popularity")
# plt.title("Popular Movies")
#
# plt.show()

# print(popular_movie_df[['title', 'popularity']].head(5))


##########################################################################
# Content Based Filtering
# print(movie_df['overview'].head(5))

tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
movie_df['overview'] = movie_df['overview'].fillna('')


# CONTENT BASED TITLE RECOMMENDATION
# # Construct the required TF-IDF matrix by fitting and transforming the data
# tfidf_matrix = tfidf.fit_transform(movie_df['overview'])
# print(tfidf_matrix.shape)
#
# # Compute the cosine similarity matrix
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#
# # Construct a reverse map of indices and movie titles
# indices = pd.Series(movie_df.index, index=movie_df['title']).drop_duplicates()


def get_recommendations(title, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movie_df['title'].iloc[movie_indices]


# print(get_recommendations('The Dark Knight Rises'))


# Credits, Genres and Keywords Based Recommender
# Parse the stringifies features into their corresponding python objects
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    movie_df[feature] = movie_df[feature].apply(literal_eval)


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


# Define new director, cast, genres and keywords features that are in a suitable form.
movie_df['director'] = movie_df['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    movie_df[feature] = movie_df[feature].apply(get_list)

# Print the new features of the first 3 films
print(movie_df[['title', 'cast']].head(3))


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    movie_df[feature] = movie_df[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


movie_df['soup'] = movie_df.apply(create_soup, axis=1)


countModel = CountVectorizer(stop_words='english')
count_matrix = countModel.fit_transform(movie_df['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
movie_df = movie_df.reset_index()
indices = pd.Series(movie_df.index, index=movie_df['title'])

# get_recommendations('The Dark Knight Rises', cosine_sim2)


##########################################################################
# Collaborative Filtering
