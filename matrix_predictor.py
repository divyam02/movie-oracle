import numpy as np
import pandas as pd
import random
import math

def optimize(user_matrix, movie_matrix, ratings_matrix, K=3):
	"""
	@Input
		user_matrix = U x K
		movie_matrix = M x K
		ratings_matrix = U x M: NaN value for unrated movies

	@Output
		optimized user_matrix and movie_matrix
	"""
	alpha = 5e-4
	beta = 5e-2

	prediction_matrix = user_matrix.dot(movie_matrix.T)
	prediction_matrix[prediction_matrix > 5] = 5
	prediction_matrix[prediction_matrix < 0] = 0

	not_nan = prediction_matrix.size - np.count_nonzero(np.isnan(ratings_matrix))
	prev_error = float('inf')

	for step in range(10*not_nan):
		for i in range(len(ratings_matrix)):
			for j in range(len(ratings_matrix[i])):
				if not np.isnan(ratings_matrix[i, j]):
					e = user_matrix[i, :].dot(movie_matrix[j, :].T) - ratings_matrix[i, j]
					for k in range(K):
						user_matrix[i, k] -= alpha * ((2 * e * movie_matrix[j, k]) + (beta * user_matrix[i, k]))
						movie_matrix[j, k] -= alpha * ((2 * e * user_matrix[i, k]) + (beta * movie_matrix[j, k]))
		
		prediction_matrix = user_matrix.dot(movie_matrix.T)
		diff_matrix = prediction_matrix - ratings_matrix
		e = np.nansum(np.square(diff_matrix)) + (beta/2)*(np.nansum(np.diagonal(user_matrix.dot(user_matrix.T)))) + (beta/2)*(np.nansum(np.diagonal(movie_matrix.dot(movie_matrix.T))))

		RMSE = np.sqrt(e/not_nan)
		print('error:', RMSE, 'lr:', alpha, 'step:',step)

		if np.absolute(RMSE-prev_error) < 0.0001:
			print('convergence!')
			print('error:', RMSE, 'lr:', alpha, 'step:',step)
			break
		else:
			if (step+1)%not_nan==0:
				alpha-=alpha/10
				prev_error = RMSE
				print('EPOCH:', step/not_nan)
				print('error:', RMSE, 'lr:', alpha, 'step:',step)

	return user_matrix, movie_matrix

def predict(user_vector, movie_matrix, ratings_vector, K=3):
	"""
	@Input
		user_vector = 1 x K: new user latent features, list
		movie_matrix = M x K: optimized. No changes here
		ratings_vector = 1 x M: Has some values (T), rest are NaN

	@Output
		ratings_vector: with all indices filled with movie ratings. Pick top (T) movies!
	"""
	alpha = 1e-3
	beta = 1e-1

	prediction_vector = user_vector.dot(movie_matrix.T)
	not_nan = prediction_vector.size - np.count_nonzero(np.isnan(ratings_vector))
	prev_error = float('inf')

	for step in range(5000):
		for j in range(len(ratings_vector)):
			if not np.isnan(ratings_vector[j]):
				e = user_vector.dot(movie_matrix[j, :].T) - ratings_vector[j]
				user_vector -= alpha * ((2 * e * movie_matrix[j, :]) + (beta * user_vector))

		prediction_vector = user_vector.dot(movie_matrix.T)
		diff_vector = prediction_vector - ratings_vector
		e = np.nansum(np.sqaure(diff_vector)) + (beta/2)*(np.nansum(user_vector.dot(user_matrix.T)))

		RMSE = np.sqrt(e/not_nan)
		print('error:', RMSE, 'lr:', alpha, 'step:',step)

		if np.absolute(RMSE-prev_error) < 0.001:
			print('convergence!')
			print('error:', RMSE, 'lr:', alpha, 'step:', step)
			break
		else:
			if step%5000==0:
				alpha-=alpha/10
				prev_error = RMSE
				print('!epoch:', step/5000)
				print('error:', RMSE, 'lr:', alpha, 'step:',step)

	
	prediction_vector = user_vector.dot(movie_matrix.T)
	prediction_vector[prediction_vector > 5] = 5
	prediction_vector[prediction_vector < 0] = 0
	ratings_vector[np.isnan(ratings_vector)] = 0
	recommendations = prediction_vector - ratings_vector

	return recommendations


def util_get_train_test_split(ratings_matrix):
	index = sorted(random.sample(range(ratings_matrix.shape[0]), ratings_matrix.shape[0]//10), reverse = True)
	test = np.array([ratings_matrix[i] for i in index])
	for i in index:
		train = np.delete(ratings_matrix, i, axis = 0)

	return train, test

def save_optimized_matrix(user_matrix, movie_matrix):
	np.save('opt_user_matrix', user_matrix)
	np.save('opt_movie_matrix', movie_matrix)

if __name__=="__main__":
	df_ratings= pd.read_csv("rate.csv")
	df_movies = pd.read_csv("./ml-latest-small/movies.csv")

	n_users = df_ratings['userId'].unique().shape[0]
	n_items = df_ratings['movieId'].unique().shape[0]

	user_ids = df_ratings['userId'].unique()
	movie_ids = df_ratings['movieId'].unique()

	table = df_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating')
	ratings_matrix = df_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating').values
	print(ratings_matrix)


	train, test = util_get_train_test_split(ratings_matrix)
	print(test, 'test')

	x = test[np.random.randint(len(test))]
	flag = 0
	true_index = []
	true_dict = dict()
	for i in range(len(x)):
		if not np.isnan(x[i]):
			true_index.append(i)

	for i in random.sample(true_index, 5):
		true_dict[i] = x[i]
		x[i] = np.NaN
		flag+=1

	print(x)
	print(true_dict)


	user_opt, movie_opt = optimize(np.random.rand(n_users, 3), np.random.rand(n_items, 3), ratings_matrix)
	save_optimized_matrix(user_opt, movie_opt)
	recommendations = predict(np.random.rand(1, 3), movie_opt, x)

	for i in true_dict.keys():
		print('predict:', recommendations[i], 'true:', true_dict[i])



	"""
	print(df_ratings.head())
	print(table.head())
	print(user_ids)
	print(sorted(movie_ids))
	print(max(df_ratings['movieId']))
	print(movie_ids[-1])
	print(max(movie_ids))
	"""

