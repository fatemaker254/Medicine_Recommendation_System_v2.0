import base64
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
data = pd.read_csv("Medicine_Details.csv")

# Concatenate symptom and side effects columns to create a corpus
data["Symptom_SideEffects"] = data["Uses"] + " " + data["Side_effects"]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(data["Symptom_SideEffects"])

# Compute similarity scores using linear kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to recommend medicines based on symptoms
def recommend_medicines(symptoms, cosine_sim=cosine_sim):
    # Transform input symptoms to TF-IDF vector
    symptoms_tfidf = tfidf_vectorizer.transform([symptoms])

    # Calculate similarity scores
    cosine_scores = linear_kernel(symptoms_tfidf, tfidf_matrix).flatten()

    # Get indices of top 5 medicines with highest similarity scores
    top_indices = cosine_scores.argsort()[:-6:-1]

    # Get the names of top 5 recommended medicines
    top_medicines = data.iloc[top_indices]["Medicine Name"].values

    return top_medicines


# adding background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://plus.unsplash.com/premium_photo-1668714068992-2a146166b860?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
[data-testid="sttextArea"] {{
color: white;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Streamlit UI
st.title("Medicine Recommendation Chatbot")


# Background image adding


symptoms_input = st.text_area("Enter your symptoms:")
if st.button("Submit"):
    if symptoms_input:
        recommended_medicines = recommend_medicines(symptoms_input)
        st.success("Recommended Medicines:")
        for med in recommended_medicines:
            st.markdown(f"- {med}", unsafe_allow_html=True)
            # st.write("-", med, style={"color": "black"})
    else:
        st.warning("Please enter your symptoms.")
