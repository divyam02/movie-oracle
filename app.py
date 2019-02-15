from flask import Flask, render_template, request
from pymongo import MongoClient
import ast
from collab_filter import *
import os
import numpy as np

client = MongoClient("mongodb://divyam:moviedb123@ds135305.mlab.com:35305/moviedb")
db = client.moviedb
collection = db.imdb_collection

ratings_matrix_for_web = np.load('ratings_matrix_for_web.npy')
imdb2movie_id = np.load('imdb2movie_id.npy')
movie2index_id = np.load('movie2index_id.npy')

count=0
movie_dict = dict()

app = Flask(__name__)

def get_collection_resources():
	"""
	gets display info for recommended movies
	"""
	pass

def get_imdb_idx(top_pred):
	top_k_movies_list = []
	for i in top_pred:
		movie_idx = movie2index_id[i]
		imdb_idx = imdb2movie_id[0, imdb2movie_id[1].tolist().index(movie_idx)]
		top_k_movies_list.append(collection.find_one({"imdb_id":str(imdb_idx)}))
		#print(movie_idx, imdb_idx)

	return top_k_movies_list

def user_cf_k(movie_dict):
	"""
	1. Get MongoDB resources
	2. Back track 
	"""
	user_vector = np.zeros_like(ratings_matrix_for_web[0])
	user_vector[:] = np.nan
	for i in movie_dict.keys():
		movie_idx = imdb2movie_id[1, imdb2movie_id[0].tolist().index(int(i))]
		vector_idx = movie2index_id.tolist().index(movie_idx)
		user_vector[vector_idx] = movie_dict[i]

	# Reinitialize global variable
	movie_dict=dict()

	similarity_vector = cosine_similarity_fast(np.copy(np.vstack((user_vector, ratings_matrix_for_web))))[0][1:]
	similarity_matrix_items = cosine_similarity_fast(np.copy(ratings_matrix_for_web.T))

	top_pred_user = predict_by_user_top_k(np.copy(user_vector), np.copy(ratings_matrix_for_web), np.copy(similarity_vector), k = 10)
	top_pred_item = predict_by_item_naive(np.copy(user_vector), np.copy(ratings_matrix_for_web), np.copy(similarity_matrix_items), k = 10)
	top_pred_mf = predict_by_mf_naive(user_vector, np.copy(ratings_matrix_for_web), k=10)
	#top_pred_mf = predict_by_mf_online(np.copy(user_vector), k=10)

	#print(similarity_vector, 'similarity_vector')
	#print(similarity_matrix_items, similarity_matrix_items.shape, 'similarity_matrix_items')
	#print(top_pred_user)
	#print(top_pred_item)


	top_k_user = get_imdb_idx(top_pred_user)
	top_k_item = get_imdb_idx(top_pred_item)
	top_k_mf = get_imdb_idx(top_pred_mf)

	#print(top_k_user)
	#print(top_k_item)

	return recommended(top_k_user, top_k_item, top_k_mf)

@app.route("/recommended")
def recommended(top_k_user, top_k_item, top_k_mf):
	return render_template("recommended.html", rec_user = top_k_user, rec_item = top_k_item, rec_mf = top_k_mf)

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/")
def home():
	print("Loaded OK")
	return render_template("home.html")

@app.route("/rate", methods=["POST"])
def rate():
	if request.method=='POST':
		global count
		if count<10:
			rating = request.form['rating']
			movie_obj = request.form['movie_obj']
			count+=1
			movie_dict[movie_obj] = rating
			#print(movie_dict)
			#return render_template("movies.html", gallery=all_movies)
			if count==10:
				# import files! or send up calculated utility matrix.
				count = 0
				return user_cf_k(movie_dict)
				return get_all()
			else:
				return movies()
	else:
		return movies()

@app.route("/get_all")
def get_all():
	all_movies = collection.find()
	

@app.route("/movies", methods=['GET', 'POST'])
def movies():
	all_movies = collection.find()
	return render_template("movies.html", gallery=all_movies)

if __name__=="__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host='0.0.0.0', port=port, debug=True)