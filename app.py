from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie data
movies = pd.read_csv("movies.csv")
# Load movie data
movies = pd.read_csv("movies.csv")

# Fill NaN descriptions with empty strings
movies["description"] = movies["description"].fillna("")

# Fit the vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["description"])

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["description"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get movie index
def get_recommendations(title):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if idx.empty:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    recommendations = get_recommendations(movie_title)
    return render_template('result.html', movie=movie_title, recs=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
