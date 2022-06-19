import requests
from bs4 import BeautifulSoup

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# You now can query the soup object
soup.title.string
soup.find('h1') # search by type
soup.find_all('a') # search by type
soup.find(id="wagon") # search by id
soup.find_all("li", class_="pizza") # search by class

############################

import requests
from bs4 import BeautifulSoup

response = requests.get("https://www.imdb.com/list/ls055386972/", headers={"Accept-Language":"en-US"})
soup = BeautifulSoup(response.content, "html.parser")

movies = []
for movie in soup.find_all("div", class_="lister-item-content"):
    title = movie.find("h3").find("a").string
    duration = int(movie.find("span", class_="runtime").string.strip(' min'))
    movies.append({'title': title, 'duration': duration})

print(movies[0:2])
