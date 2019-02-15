import matplotlib.pyplot as plt
import numpy as np
import random

"""
We describe a measure for K, the number of movies we rate before being shown K other similar movies. Users who have rated atleast 20 movies are selected.
We randomly select k rated movies and recommend K more. We then detemine how many of the user's better rated movies (3+ stars) were present in the K 
recommended movies.
"""

import time


def util_train_test_split(ratings_matrix):
	"""
	@Input
		ratings_matrix = users, ratings (includes newly added user vector)

	@Output
		train, test set. Test may contain NaN values
	
	@Note
		Make sure a copy is being sent in the ratings_matrix field!
	"""

	test = np.zeros_like(ratings_matrix)
	test[:] = np.nan
	train = np.copy(ratings_matrix)

	for user in range(ratings_matrix.shape[0]):
		item = np.random.choice(ratings_matrix[user], replace=False, size=ratings_matrix[user].size//5)
		for i in range(len(item)):
			if not np.isnan(item[i]):
				train[user, i] = np.nan
				test[user, i] = ratings_matrix[user, i] # test may contain NaN values!

	print(test)
	print(train)

	return train, test


def get_dict(ratings_matrix):
	train_dict = dict()
	for i in range(ratings_matrix.shape[0]):
		for j in range(ratings_matrix.shape[1]):
			if not np.isnan(ratings_matrix[i, j]):
				train_dict[(i, j)] = ratings_matrix[i, j]

	return train_dict


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
	@Note
		Make sure its a copy of the matrix you are inserting!
	"""
	top_list = [0]*ratings_matrix.shape[0]
	#print(similarity_vector, similarity_vector.shape)
	for j in range(len(ratings_matrix)):
		if not np.isnan(ratings_matrix[j, item]):
			top_list[j] = similarity_vector[j]	
	top_k = np.array(top_list).argsort()[-1*neighbors:][::-1].tolist()

	#print(top_k, 'top k users')
	top_list = [0]*ratings_matrix.shape[0]
	for i in range(len(similarity_vector)):
		if(i in top_k):
			top_list[i] = similarity_vector[i]

	#print(np.array(top_list), 'top N users')

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
			top_N_sim = get_top_N_users(item, np.copy(ratings_matrix), np.copy(similarity_vector), neighbors)
			# Careful! Don't make permanent changes.
			temp_matrix = np.copy(ratings_matrix)
			temp_matrix[np.isnan(temp_matrix)] = 0

			predict_rating = user_avg_rating + top_N_sim.dot(temp_matrix[:, item].T)/np.nansum(np.absolute(top_N_sim))
			top_predict[item] = predict_rating
	
	#print(top_predict, 'top_predict, user')
	top_k_movies = [a for a, b in sorted(top_predict.items(), key=lambda kv:kv[1], reverse=True)]	

	return top_k_movies[:k]


def predict_by_item_naive(user_vector, ratings_matrix, similarity_matrix, k=10):
	"""
	@Input 
		user_vector = 1 x movies with NaN
		ratings_matrix = users x movies with Nan
		similarity_matrix = movies x movies, similarity of every item with other.

	@Output
		Return user_vector with estimated ratings of movies. Ignore voted ones!

	@Note
		Naive item-item filtering. Make sure no permanent changes are being made that violate loop/if conditions...
	"""
	top_predict = dict()
	# Include user_vector when calculating mean for every movie rating?
	item_avg_rating = np.nanmean(ratings_matrix, axis=0)
	user_vector -= item_avg_rating
	for item_idx in range(len(user_vector)):
		if np.isnan(user_vector[item_idx]):
			temp_vector = np.copy(user_vector)
			temp_sim_vector = np.copy(similarity_matrix[item_idx])
			temp_sim_vector[np.isnan(temp_vector)] = 0
			temp_vector[np.isnan(temp_vector)] = 0

			top_predict[item_idx] = item_avg_rating[item_idx] + temp_vector.dot(similarity_matrix[item_idx])/np.sum(np.absolute(temp_sim_vector))

	#print(top_predict, 'top_predict, item')
	top_k_movies = [(a, b) for a, b in sorted(top_predict.items(), key=lambda kv:kv[1], reverse=True)]

	return top_k_movies[:k]	


def get_mse(prediction_matrix, Rating_matrix):
	"""
	Returns mean square error
	"""
	not_nan = prediction_matrix.size - np.count_nonzero(np.isnan(Rating_matrix))
	#print(Rating_matrix, 'Rating_matrix')
	#print(prediction_matrix, 'prediction_matrix')
	diff_matrix = Rating_matrix - prediction_matrix
	#print(diff_matrix, 'diff_matrix')

	diff_matrix[np.isnan(diff_matrix)] = 0

	loss = np.nansum(np.diagonal(diff_matrix.dot(diff_matrix.T)))

	RMSE = np.sqrt(loss/not_nan)

	return RMSE


def train(ratings_matrix, user_matrix, item_matrix, user_bias, item_bias, rating_bias, iterations=5000, alpha=1e-3, beta=1e-1):
	"""
	@Input 
		user_matrix = users x latent_factors. contains new user factors
		item_matrix = items x latent_factors
		user_bias = users x 1
		item_bias = items x 1
		rating_bias = global rating average
		ratings_matrix = users x movies with NaN

	@Output
		optimized_user_matrix = users x latent_factors
		optimized_item_matrix = items x latent_factors

	@Note
	"""
	print('GERONIMO!')
	validation_error = float('inf')

	train_matrix, test_matrix = util_train_test_split(np.copy(ratings_matrix))
	train_dict = get_dict(train_matrix)
	idx = list(train_dict.keys())
	
	for step in range(iterations):
		for user_item in idx:
			if not np.isnan(train_dict[user_item]):
				i, j = user_item
				true_ij = train_dict[user_item]

				prediction_ij = user_matrix[i, :].dot(item_matrix[j, :].T) + user_bias[i] + item_bias[j] + rating_bias
			
				e_ij = true_ij - prediction_ij
				
				#assert 1<0
				user_matrix[i, :] += alpha * ((e_ij * item_matrix[j, :]) - (beta * user_matrix[i, :]))
				item_matrix[j, :] += alpha * ((e_ij * user_matrix[i, :]) - (beta * item_matrix[j, :]))

				user_bias[i] += alpha * (e_ij - (beta * user_bias[i]))
				item_bias[j] += alpha * (e_ij - (beta * item_bias[j]))

		prediction_matrix = user_matrix.dot(item_matrix.T) + user_bias[:, np.newaxis] + item_bias[np.newaxis:, ] + rating_bias
		
		if (step + 1) % 20 == 0:
			validation_RMSE = get_mse(prediction_matrix, test_matrix)
			if validation_error - validation_RMSE < 5e-4:
				print('difference', validation_error - validation_RMSE)
				print('step:', step, 'convergence!')
				break
			else:
				print('difference', validation_error - validation_RMSE)
				print('validation_RMSE:', validation_RMSE, 'step:', step)
				validation_error = validation_RMSE

	return user_matrix, item_matrix, user_bias, item_bias


def predict_by_mf_naive(user_vector, ratings_matrix, latent_factors = 3, k=10):
	"""
	@Input
		user_vector: 1 x movies with NaN
		ratings_matrix: users x movies with NaN values

	@Output
		Return dictionary of top index - rating items

	@Note
		Perform fresh factorization for the new user
	"""
	top_predict=dict()

	ratings_matrix_with_new_user = np.copy(np.vstack((user_vector, ratings_matrix)))

	user_matrix = np.random.normal(scale = 1/latent_factors, size=((ratings_matrix_with_new_user.shape[0], latent_factors)))
	#user_matrix = np.random.rand(ratings_matrix_with_new_user.shape[0], latent_factors) * (1/latent_factors)
	item_matrix = np.random.normal(scale = 1/latent_factors, size=((ratings_matrix_with_new_user.shape[1], latent_factors)))
	#item_matrix = np.random.rand(ratings_matrix_with_new_user.shape[1], latent_factors) * (1/latent_factors)

	user_bias = np.zeros(user_matrix.shape[0])
	item_bias = np.zeros(item_matrix.shape[0])
	rating_bias = np.nanmean(ratings_matrix_with_new_user)

	start_time = time.time()

	optimized_user_matrix, optimized_item_matrix, optimized_user_bias, optimized_item_bias = train(ratings_matrix_with_new_user, 
																									user_matrix, item_matrix,
																									user_bias, item_bias, 
																									rating_bias)

	print('Running time:', time.time() - start_time)
	

	#np.save('optimized_item_matrix', optimized_item_matrix)
	#np.save('optimized_item_bias', optimized_item_bias)
	#np.save('rating_bias', rating_bias)


	user_vector_full = optimized_user_matrix.dot(optimized_item_matrix.T) + optimized_user_bias[:, np.newaxis] + optimized_item_bias.T[np.newaxis:,] + rating_bias

	user_vector_full = user_vector_full[0]

	print(user_vector_full.shape)

	for i in range(len(user_vector_full)):
		if np.isnan(user_vector[i]):
			top_predict[i] = user_vector_full[i]

	top_k_movies = [(a, b) for a, b in sorted(top_predict.items(), key=lambda kv:kv[1], reverse=True)]
	
	return top_k_movies[:k]

def predict_by_mf_online(user_vector, k=10, latent_factors=3, alpha=1e-3, beta=1e-1):
	"""
	@Input
		user_vector: 1 x movies with NaN
		user_latent_matrix: users x L where L are # of latent factors, optimized
		movie_latent_matrix: movies x L where as above, optimized

	@Output
		Return dictionary of top index - rating items

	@Note
		No retraining the weights for new entry. Perform SGD on the vector for given ratings to optimize its latent factors.
		Movie matrix weights are not updated
	"""
	start_time = time.time()
	print('here we go!', start_time)
	top_predict=dict()
	validation_error=float('inf')

	user_latent_vector = np.random.normal(scale=1/latent_factors, size=((1, latent_factors)))
	user_bias = 0
	optimized_item_matrix = np.load('optimized_item_matrix.npy')
	optimized_item_bias = np.load('optimized_item_bias.npy')
	rating_bias = np.load('rating_bias.npy')

	train_dict = dict()
	for j in range(len(user_vector)):
			if not np.isnan(user_vector[j]):
				train_dict[j] = user_vector[j]

	item_idx = list(train_dict.keys())
	for i in range(5000):
		for j in item_idx:
			true_ij = train_dict[j]
			prediction_ij = user_latent_vector.dot(optimized_item_matrix[j].T) + user_bias + optimized_item_bias[j] + rating_bias
			e_ij = true_ij - prediction_ij
			#print(true_ij, prediction_ij, j)
			user_latent_vector += alpha * ((e_ij * optimized_item_matrix[j, :]) - (beta * user_latent_vector))
			user_bias += alpha * (e_ij - (beta * user_bias))
	
		prediction_vector = user_latent_vector.dot(optimized_item_matrix.T) + user_bias + optimized_item_bias + rating_bias
		if (i+1) % 25==0:
			validation_RMSE = get_mse(prediction_vector, user_vector)
			if validation_error < validation_RMSE:
				print('difference', validation_error - validation_RMSE)
				print('step:', i, 'convergence!')
				break
			else:
				print('difference', validation_error - validation_RMSE)
				print('validation_RMSE:', validation_RMSE, 'step:', i)
				validation_error = validation_RMSE			

	user_vector_full = user_latent_vector.dot(optimized_item_matrix.T) + user_bias + optimized_item_bias + rating_bias
	
	#print(user_latent_vector)
	#print(user_latent_vector.dot(optimized_item_matrix.T), user_latent_vector.dot(optimized_item_matrix.T).shape)

	user_vector_full = user_vector_full[0]

	#print(user_vector_full, user_vector_full.shape)

	for i in range(len(user_vector_full)):
		if np.isnan(user_vector[i]):
			#print("")
			top_predict[i] = user_vector_full[i]
			#print(top_predict[i])

	top_k_movies = [a for a, b in sorted(top_predict.items(), key=lambda kv:kv[1], reverse=True)]
	#print(top_k_movies)
	
	return top_k_movies[:k]	


if __name__ == '__main__':
	

	dict_arr = []

	for D in range(10):

		fig = plt.figure()
		ax = plt.axes()
		x = np.linspace(0, 10, 1000)
		ratings_matrix = np.load('ratings_matrix_for_web.npy')

		minimum = float('inf')
		index = -1
		not_nan_idx =[]

		min_user = np.random.randint(len(ratings_matrix))

		for i in range(len(ratings_matrix[min_user])):
			if not np.isnan(ratings_matrix[min_user, i]):
				not_nan_idx.append(i)

		ratings_matrix = np.delete(ratings_matrix, (min_user), axis=0)

		test_vector = np.zeros_like(ratings_matrix[min_user])

		score_dict = dict()
		i_list=[]
		value_list=[]

		for i in [3, 5, 7]:
			score_mf = 0

			known_idx = random.sample(range(len(not_nan_idx)), i)
			test_vector[:] = np.nan
			div = 0
			for j in known_idx:
				test_vector[not_nan_idx[j]] = ratings_matrix[min_user, not_nan_idx[j]]


			top_pred_mf = predict_by_mf_naive(np.copy(test_vector), np.copy(ratings_matrix), k=i)

			for k, p in top_pred_mf:
				print('predicted rating:', p, 'actual rating by user:', ratings_matrix[min_user, k])
				if not np.isnan(ratings_matrix[min_user, k]):
					div += 1
					if ratings_matrix[min_user, k] > 3:
						score = p - ratings_matrix[min_user, k]
						score = score * score
						score_mf += score

			score_mf /= div
			score_dict[i] = score_mf
			i_list.append(i)
			value_list.append(score_mf)
			print('for k =', i, 'score:', score_mf)
		
		#plt.show(i_list, value_list)
		plt.plot(i_list, value_list)
		plt.savefig(str(D)+" "+str(i))
		dict_arr.append(score_dict)


	_3_avg = 0
	_5_avg = 0
	_7_avg = 0

	for i in dict_arr:
		for j in list(i.keys()):
			if j==3:
				_3_avg+=i[j]
			elif j==5:
				_5_avg+=i[j]
			else:
				_7_avg+=i[j]

	_3_avg /= len(dict_arr)
	_5_avg /= len(dict_arr)
	_7_avg /= len(dict_arr)

	plt.plot([3, 5, 7], [_3_avg, _5_avg, _7_avg])
	plt.savefig('k_variance')

	print(dict_arr)

	#plt.show()