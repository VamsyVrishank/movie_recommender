import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#helper functions
def get_title_from_index(index):
    return df[df['index'] == index]['title'].values[0]

def get_index_from_title(title):
    try:
        return df[df['title'] == title]['index'].values[0]
    except:
        print("Capitalize the first letter of movie or movie may not exist in the database  XD!!")
#Reading the csv file
df = pd.read_csv('/home/vamc/Documents/Projects/rec_engine/data.csv')

#Selecting the features
features = ['keywords', 'cast','genres','director']

#filling up the nan values
for feature in features:
    df[feature] = df[feature].fillna('')

#combining features
def combine_features(row):
    return (row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row['director'])
df['combined_features'] = df.apply(combine_features , axis =1)

#Creaating the count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

#computing the cosine similarity
cs = cosine_similarity(count_matrix)

movie_user_likes = 'Ghost Rider'


movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cs[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x: x[1], reverse=True)

#printing the movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i+=1
    if(i>50):
        break


