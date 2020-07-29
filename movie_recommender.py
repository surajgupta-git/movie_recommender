import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df=pd.read_csv("movie_dataset.csv")
#print (df.columns)
##Step 2: Select Features
features=['keywords','cast','genres','director']
for f in features:
	df[f]=df[f].fillna('')
##Step 3: Create a column in DF which combines all selected features
def combine_features(row):
	try:
		return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
	except:
		print('error')

df['combined_features']=df.apply(combine_features,axis=1)
print(df['combined_features'].head())

##Step 4: Create count matrix from this new combined column
cv=CountVectorizer()

count_matrix=cv.fit_transform(df['combined_features'])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosinesim=cosine_similarity(count_matrix)
print(cosinesim)
movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movtitle=get_index_from_title(movie_user_likes)
simmovies=list(enumerate(cosinesim[movtitle]))

## Step 7: Get a list of similar movies in descending order of similarity score

simmovies.sort(key= lambda x:x[1],reverse=True)

for mv in simmovies:
	print(get_title_from_index(mv[0]))

## Step 8: Print titles of first 50 movies