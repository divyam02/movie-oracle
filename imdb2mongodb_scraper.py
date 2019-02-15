from requests import get
from bs4 import BeautifulSoup
from pymongo import MongoClient
import pprint

def get_page_url(imdb_id):
	"""
	Returns IMDb movie url.
	input format: int<imdb_id>
	output format: str<imdb_url>
	"""
	if(len(str(imdb_id))<6):
		url = "https://www.imdb.com/title/tt00"+str(imdb_id)
	else:
		url = "https://www.imdb.com/title/tt0"+str(imdb_id)
	print(url)
	return url, str(imdb_id)

def get_links():
	"""
	Return list of IMDb movie IDs.
	output format: list(int)
	"""
	with open('web_dataset', 'r') as f:
		movie_ids = f.read()
	return list(map(int, movie_ids.split()))

def get_movie_data(imdb_url, imdb_id):
	"""
	Returns dict() containing movie attributes
	input: str<imdb url>
	"""
	page_url = imdb_url	

	page_response = get(page_url)

	html_soup = BeautifulSoup(page_response.text, 'html.parser')

	img_containers = html_soup.find_all('div', class_='poster')
	title_containers = html_soup.find_all('div', class_='title_wrapper')

	thumbnail_src = img_containers[0].a.img['src']
	movie_title= title_containers[0].h1.text[:-8]
	release_year = title_containers[0].h1.span.a.text
	genres = title_containers[0].find_all('div', class_='subtext')[0].find_all('a')[:-1]
	genre_list = list()

	for i in genres:
		genre_list.append(i.text)

	print(thumbnail_src, type(thumbnail_src))
	print(movie_title, type(movie_title))
	print(release_year, type(release_year))
	print(genre_list, type(genre_list))
	print(imdb_id)

	return {'title': movie_title, 'release_year': release_year,
				'thumbnail': thumbnail_src, 'genre_list': genre_list, 'imdb_id': imdb_id}

def check_genre(genre, genre_count):
	"""
	Returns boolean value for adding to MongoDB database.
	input: list(str)<movie genre>, dict(int)<count per genre>
	"""
	limit = 30
	counts = dict()
	for i in genre:
		if(i in genre_count.keys()):
			counts[i] = genre_count[i]
		"""
		else:
			counts[i] = 0
			genre_count[i] = 0
		"""
	
	movies_stored = sum(genre_count.values())
	if(movies_stored<=200 and counts):
		key = min(counts, key=counts.get)
		print(counts)
		print(genre_count)
		genre_count[key] += 1
		print('Movies stored:', movies_stored)
		return True
	return False

def add_to_mongodb(movie, db):
	"""
	Adds movie to database
	input: dict<movie>, db<MongoDB client>
	"""	
	db.imdb_collection.insert_one(movie)	


if __name__ == '__main__':
	client = MongoClient("mongodb://priyam:divsmlab0@ds135305.mlab.com:35305/moviedb")
	db = client.moviedb
	failed = list()
	imdb_ids = get_links()
	genre_count={'Action':0,
				'Romance':0,
				'Comedy':0,
				'Drama':0,
				'Horror':0,
				'Sci-Fi':0,
				'Animation':0,
				'Fantasy':0}

	for i in imdb_ids:
		if(sum(genre_count.values())<=200):
			try:
				url, imdb_id = get_page_url(i) 
				movie = get_movie_data(url, imdb_id)
				if check_genre(movie['genre_list'], genre_count):
					add_to_mongodb(movie, db)
					print('added',movie['title'],'to database!\n')
				else:
					print('rejected',movie['title'])
			except (AttributeError, IndexError) as e:
				failed.append(get_page_url(i))
		else:
			break

	print(db.list_collection_names())
	pprint.pprint(genre_count)
	with open('error_log', 'w') as f:
		for i in failed:
			f.write(i+" ")
	client.close()