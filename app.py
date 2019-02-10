from flask import Flask, render_template, request
from pymongo import MongoClient

client = MongoClient("mongodb://127.0.0.1:27017")
db = client.IMDb_database
collection = db.IMDb_collection_test_2

title = "MONGOBONGO"
heading = "what the title said"

count=0
movie_list = []

app = Flask(__name__)

@app.route("/Divyam")
def Divyam():
	return "Hello World"

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/rate", methods=["POST"])
def rate():
	if request.method=='POST':
		global count
		if count<5:
			rating = request.form['rating']
			movie_obj = request.form['movie_obj']
			print(count)
			print(movie_obj, rating)
			#movie_obj = ast.literal_eval(movie_obj)
			#movie_obj['rating'] = rating
			movie_list.append(movie_obj)
			print(movie_list)
			#return render_template("movies.html", gallery=all_movies)
			return movies()
			if count==5:
				# import files! or send up calculated utility matrix.
				count = 0
				movie_list = []
				return recommend(movie_list)
	else:
		return movies()
	"""
	elif movie_count==5:
		return render_template()	
	"""

@app.route("/recommend")
def recommend(movie_list):
	all_movies = collection.find()
	

@app.route("/movies", methods=['GET', 'POST'])
def movies():
	all_movies = collection.find()
	return render_template("movies.html", gallery=all_movies)

if __name__=="__main__":
	app.run(debug=True)