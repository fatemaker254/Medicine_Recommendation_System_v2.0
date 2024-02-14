import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.title("Medicine Recommendation Chatbot")
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Subtitle
st.write("                🤖 : Hi, How are you Today?")

data = pd.read_csv("Medicine_Details.csv")
if "messages" not in st.session_state:
    st.session_state.messages = []


# Concatenate symptom and side effects columns to create a corpus
data["Symptom_SideEffects"] = data["Uses"] + " " + data["Side_effects"]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(data["Symptom_SideEffects"])

# Compute similarity scores using linear kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


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
[data-testid="sttext_input] {{
color: black; /* Change text color to black */
background-color: white; /* Change background color to white */
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.text_input("You:", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        recommendations = recommend_medicines(symptoms=prompt)
        message_content = ""
        if isinstance(recommendations, str):
            message_content = recommendations
        else:
            message_content = "\n".join(f"- {med}" for med in recommendations)
        st.markdown(message_content, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": message_content})
