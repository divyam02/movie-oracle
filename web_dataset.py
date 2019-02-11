import pandas as pd
import numpy as np


if __name__ == '__main__':
	df_ratings= pd.read_csv("rate.csv")
	df_movies = pd.read_csv("./ml-latest-small/movies.csv")

	n_users = df_ratings['userId'].unique().shape[0]
	n_items = df_ratings['movieId'].unique().shape[0]

	user_ids = df_ratings['userId'].unique()
	movie_ids = df_ratings['movieId'].unique()





	ratings_matrix = df_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating').values

	ratings_matrix[ratings_matrix>0] = 1

	print(ratings_matrix)
	print(ratings_matrix[0, 0])


	#print(df_ratings)

	a = df_ratings['movieId'].value_counts()


	movie_ids_list = sorted(a.keys())

	#print(a)
	#print(a.keys())

	b = a.keys()[:200]
	
	b = sorted(b)

	c = np.zeros((ratings_matrix.shape[0], 1))
	print(c)

	print(b)

	for i in b:
		print(i)
		movie_index = movie_ids_list.index(i)
		c = np.column_stack((c, ratings_matrix[:, movie_index]))

	most_rating_users = dict()
	total_users = 0
	min_rates = np.nansum(c[0])
	min_people = 25


	for i in range(len(c)):
		most_rating_users[i] = np.nansum(c[i])
		min_rate = min(most_rating_users.values())

	print(most_rating_users)		
	most_rating_users = sorted(most_rating_users.items(), key=lambda kv: kv[1], reverse=True)
	print(most_rating_users)

	print(c, c.shape)

	c = c[:, 1:]

	print(c, c.shape)
