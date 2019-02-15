# movie-oracle
A movie recommendation system, with implementations of:
1. User-User Collaborative Filtering
2. Item-Item Collaborative Filtering
3. Matrix Factorization with Stochastic Gradient Descent

# Approach
## MongoDB database
I started off by building a webscraper for IMDb. I used `links.csv` and `ratings.csv` of the GroupLens dataset to obtaid a list of IMDb URLs. I then wrote a script that processed 200 of the most rated (the number of people who have rated it) movies and separated them into bins, based on some genres. The sampling was done for a closer aproximation of real world movies.

Action | Romance | Comedy | Drama | Horror | Sci-fi | Animation | Fantasy
-------|---------|--------|-------|--------|--------|-----------|-------
30  | 17 | 29 | 37  | 29  | 10  | 20  | 29

## Sources
### IMDb Web Scraper
1. Web-Scraping tutorial with Beautiful Soup: https://www.dataquest.io/blog/web-scraping-beautifulsoup/
2. Beautiful Soup documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
### MongoDB Database
1. Mongo Shell: https://docs.mongodb.com/manual/reference/mongo-shell/
2. MongoDB Python API: https://api.mongodb.com/python/current/
3. Usage: https://docs.mongodb.com/manual/
