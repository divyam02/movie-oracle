# movie-oracle
A movie recommendation system, with implementations of:
1. User-User Collaborative Filtering
2. Item-Item Collaborative Filtering
3. Matrix Factorization with Stochastic Gradient Descent
4. Online Learning for Rapid Recommendations

Web application available on Heroku at this link: https://movie-oracle.herokuapp.com/movies
For usage, refer to the `Usage` section below.

# Approach
## MongoDB database
I started off by building a webscraper for IMDb. I used `links.csv` and `ratings.csv` of the GroupLens dataset to obtaid a list of IMDb URLs. I then wrote a script that processed 200 of the most rated (the number of people who have rated it) movies and separated them into bins, based on some genres for 50 users. The sampling was done for a closer aproximation of real world movies. 

Action | Romance | Comedy | Drama | Horror | Sci-fi | Animation | Fantasy
-------|---------|--------|-------|--------|--------|-----------|-------
30  | 17 | 29 | 37  | 10  | 29 | 20  | 29

I used `mlab` to host my MongoDB data set.

## Algorithms
### User-User Collaborative Filtering
I was able to optimize the calculations (especially the `Similarity Matrix`) by using numpy operations. I also enabled it to take weights of the top N similar users. Metric for similarity was Cosine Similarity. The algorithm returns ratings for an item weighed by the similarity of other users to itself.
### Item-Item Collaborative Filtering
Same as User based filtering, I took the transpose of the initial matrix of users and items for ratings and followed a similar structure. Optimized with numpy operations. The algorithm returns ratings for an item weighted by the similarity of that item to others and the user's ratings for those items.
### Matrix Factorization
I attempted to represent this as two matrices with vectors of users and items containing latent factors. I included user and item biases for ratings and L2 regularization for the loss function. I used `stochastic gradient descent` to optimize their latent factors after solving the update rules for myself. I have managed to optimize it somewhat with numpy, and I am obtaining convergence in 400 - 600 iterations. 
### Online Learning Rapid Recommendation
The convergence of validation loss for the previous method usually takes more than 30 seconds, after which Heroku requests a timeout and the application crashes. For that purpose I performed gradient descent on a single vector representing the user input ratings and trained its latent factors using the `optimized item_latent_factor matrix` and `optimized item bias` I calculated in the previous method. This allows for rapid training online to produce recommendations without having to resolve the original ratings matrix.

##  Deciding K
K is the number of ratings the user must input in order to recieve K recommendations. I was able to calculate the RMSE for the predicted ratings of the top K movies the matrix factorization algorithm returns against the actual ratings given by the user. I did this by keeping only K random ratings in the ratings vector of the user and making the rest unrated (NaN). Then using the matrix factorizing algorithm, I got predictions for the other items and calculated RMSE for top K. I then averaged over the RMSE errors I got for various values of K over 10 randomly selected users. 

K  is optimally = 5, at least for my dataset

![alt text](https://github.com/divyam02/movie-oracle/blob/master/mf_plots/k_variance.png)

(The Orange Line is the average RMSE for K. The blue is a usual example)

## Usage
Open the Heroku link. Rate movies individually, ie insert a number from 1 to 5 in the text box and click "Rate!". Wait for the page to refresh, then continue with the next rating. Please note that the app can't take multiple values at once and each movie must be rated one at a time. After five inputs, the application takes you to the Recommendation page.

Run the application locally using `python3 app.py`.

## Bugs
On the front end. Due to some logical bug, Null objects are inserted in the database, while rejecting a movie. If this movie is recommended by the application, it appears as a `()` in the page.

## Directory Structure and Files
No directories, apart from `templates` used to store HTML files and `mf_plots` that store the RMSE plots over values of K, the items rated by a user. 
Main components:
1. `collab_filter.py`: All Matrix Factorization and Collaborative Filtering Algorithms 
2. `app.py`: The main web application, prints debug notes in the console. Run the application locally using `python3 app.py`
3. `determine_k.py`: Used for computing the optimal K value.
4. `imdb2mongodb_scraper.py`: Used for building MongoDB database on `mlab`.
5. `get_opt_matrices.py`: Used to compute `optimized_item_matrix` and `optimized_item_bias`.

## Dependencies
1. `Python 3.7`: Heroku uses the latest version of python.
2. `pymongo`: To communicate with `mlab` and handle all requests
3. `beautifulsoup4`: To use for web scraping
4. `flask`: Front-end and back-end related interfaces.
5. `numpy`: To process matrix and vector operations

## Sources
### Matrix Factorization
1. Equations and update rules: https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/baalbaki.pdf
2. References for implementation: https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#intromf , http://www.albertauyeung.com/post/python-matrix-factorization/
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
