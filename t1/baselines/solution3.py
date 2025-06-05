### **Exercise 3: Implementing a Simple Recommendation System**
import pandas as pd

url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
column_names = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv(url, sep="\t", names=column_names, usecols=["user_id", "movie_id", "rating"])

url_movies = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movies = pd.read_csv(url_movies, sep="|", encoding="latin-1", names=["movie_id", "title"], usecols=[0, 1])

# Imbinarea titlurilor de filme cu setul de date pentru ratings
ratings = ratings.merge(movies, on="movie_id")

# Filtrarea utilizatorilor care au evaluat mai putin de 10 filme
user_ratings_count = ratings.groupby("user_id").size()
valid_users = user_ratings_count[user_ratings_count >= 10].index
filtered_ratings = ratings[ratings["user_id"].isin(valid_users)]

# Calcularea rating-ului mediu pe film
movie_avg_ratings = filtered_ratings.groupby("title")["rating"].mean()

# Obtinerea celor mai populare 5 filme
top_movies = movie_avg_ratings.sort_values(ascending=False).head(5)

# Afisarea filmelor recomandate
print("Top 5 most popular movies: \n")
print(top_movies)