import pandas as pd
import numpy as np
import random

def util_train_test_split(ratings_matrix):
	"""
	train-test-split
	"""
	index = sorted(random.sample(range(ratings_matrix.shape[0]), ratings_matrix.shape[0]//10), reverse = True)
	test = np.array([ratings_matrix[i] for i in index])
	for i in index:
		train = np.delete(ratings_matrix, i, axis = 0)

	return train, test

def cosine_similarity_fast(ratings_matrix, user=True, e = 1e-7):
	"""
	Calculate cosine similarity matrix
	"""
	temp_matrix = np.copy(ratings_matrix)
	temp_matrix[np.isnan(temp_matrix)] = 0
	if user:
		sim = temp_matrix.dot(temp_matrix.T) + e
	else:
		sim = temp_matrix.T.dot(temp_matrix) + e

	norms = np.array([np.sqrt(np.diagonal(sim))])

	return (sim / norms / norms.T)

def cosine_shifted_similarity_fast(ratings_matrix, user=True):
	"""
	Mean shifted cosine similarity matrix
	"""
	pass

def get_top_N_users(item, ratings_matrix, similarity_vector, neighbors=10):
	"""
	Return similarity vector with N nearest neighbors
	Make sure its a copy of the matrix you are inserting!
	"""
	top_list = [0]*ratings_matrix.shape[0]
	#print(similarity_vector, similarity_vector.shape)
	for j in range(len(ratings_matrix)):
		if not np.isnan(ratings_matrix[j, item]):
			top_list[j] = similarity_vector[j]	
	top_k = np.array(top_list).argsort()[-1*neighbors:][::-1].tolist()

	print(top_k)
	top_list = [0]*ratings_matrix.shape[0]
	for i in range(len(similarity_vector)):
		if(i in top_k):
			top_list[i] = similarity_vector[i]

	print(np.array(top_list))

	return(np.array(top_list))

def predict_by_user_top_k(user_vector, ratings_matrix, similarity_vector, k = 5, neighbors = 10):
	"""
	@Input
		user_vector = 1 x movies with NaN
		ratings_matrix = users x movies with NaN
		similarity_vector = 1 x users

	@Output
		Return user_vector.shape[1] top movies by user-user similarity

	@Note
		Make sure its a copy of the matrix you are inserting!
	"""
	top_predict  = dict()
	user_avg_rating = np.nanmean(user_vector)
	Q = np.nanmean(ratings_matrix, axis=1)
	for i in range(len(ratings_matrix)):
		ratings_matrix[i] -= Q[i]
	
	for item in range(len(user_vector)):
		if np.isnan(user_vector[item]):
			top_N_sim = get_top_N_users(item, ratings_matrix, similarity_vector)
			ratings_matrix[np.isnan(ratings_matrix)] = 0
			predict_rating = user_avg_rating + top_N_sim.dot(ratings_matrix[:, item].T)/np.nansum(np.absolute(top_N_sim))
			top_predict[item] = predict_rating
	
	print(top_predict)
	top_k_movies = [a for a, b in sorted(top_predict.items(), key=lambda kv:kv[1], reverse=True)]	

	return top_k_movies[:5]

def predict_by_item():
	"""
	item-item CF
	"""
	pass
