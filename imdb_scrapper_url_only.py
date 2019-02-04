from requests import get
from bs4 import BeautifulSoup
import pprint

page_url = "https://www.imdb.com/title/tt0137523/?ref_=nv_sr_1"

page_response = get(page_url)

html_soup = BeautifulSoup(page_response.text, 'html.parser')

img_containers = html_soup.find_all('div', class_='poster')
title_containers = html_soup.find_all('div', class_='title_wrapper')

thumbnail_src = img_containers[0].a.img['src']
movie_title= title_containers[0].h1.text[:-8]
release_year = title_containers[0].h1.span.a.text

print(thumbnail_src, type(thumbnail_src))
print(movie_title, type(movie_title))
print(release_year, type(release_year))



