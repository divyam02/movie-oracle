# movie-oracle
A movie recommendation system, with implementations of:
1. User-User Collaborative Filtering
2. Item-Item Collaborative Filtering
3. Matrix Factorization with Stochastic Gradient Descent
4. Online Learning for Rapid Recommendations

Web application available on Heroku at this link: https://movie-oracle.herokuapp.com/movies

# Approach
## MongoDB database
I started off by building a webscraper for IMDb. I used `links.csv` and `ratings.csv` of the GroupLens dataset to obtaid a list of IMDb URLs. I then wrote a script that processed 200 of the most rated (the number of people who have rated it) movies and separated them into bins, based on some genres. The sampling was done for a closer aproximation of real world movies. 

Action | Romance | Comedy | Drama | Horror | Sci-fi | Animation | Fantasy
-------|---------|--------|-------|--------|--------|-----------|-------
30  | 17 | 29 | 37  | 29  | 10  | 20  | 29

I used `mlab` to host my MongoDB data set.

## Algorithms

## Sources
### Matrix Factorization
1. Equations and update rules: https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/baalbaki.pdf
2. References for implementation: https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#intromf
### Collaborative Filtering
1. The equations for getting similarity: https://en.wikipedia.org/wiki/Collaborative_filtering
2. References for implementation: https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/
### IMDb Web Scraper
1. Web-Scraping tutorial with Beautiful Soup: https://www.dataquest.io/blog/web-scraping-beautifulsoup/
2. Beautiful Soup documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
### MongoDB Database
1. Mongo Shell: https://docs.mongodb.com/manual/reference/mongo-shell/
2. MongoDB Python API: https://api.mongodb.com/python/current/
3. Usage: https://docs.mongodb.com/manual/
