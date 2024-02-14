import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

# Load dataset
data = pd.read_csv("Medicine_Details.csv")

# Preprocess data
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
data["Symptoms"] = data["Uses"] + " " + data["Side_effects"]

# TF-IDF Vectorization

tfidf_matrix = tfidf_vectorizer.fit_transform(data["Symptoms"])

# Compute similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Save model to file
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(cosine_sim, "medicine_recommendation_model.joblib")
