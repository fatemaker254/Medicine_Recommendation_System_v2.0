import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.title("Medi Recommender")
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

st.write(
    "<font size='+6'>🤖 : Hi, How are you feeling today?</font>",
    unsafe_allow_html=True,
)

st.sidebar.write("Please enter the following details to predict the disease.")

# Load the dataset
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Define categorical columns
categorical_cols = [
    "Disease",
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level",
]

# Encode categorical columns
encoder = OneHotEncoder(drop="first", sparse=False)
encoded_data = encoder.fit_transform(df[categorical_cols])

# Get feature names after encoding
feature_names = encoder.get_feature_names_out(categorical_cols)

# Create a DataFrame with encoded categorical columns
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

# Concatenate the encoded categorical columns with the numerical columns
X = pd.concat([encoded_df, df["Age"]], axis=1)
y = df["Outcome Variable"]

# Train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Get user input for prediction
st.sidebar.subheader("Enter Details for Prediction")
new_data = {}
for col in categorical_cols:
    val = st.sidebar.text_input(col)
    new_data[col] = val
new_data["Age"] = st.sidebar.number_input(
    "Age", 0, 150, 25
)  # Assuming Age is a numerical column

# Ensure new input has the same number of features as the training data
new_input = np.zeros((1, X.shape[1]))  # Initialize array for new input
for i, col in enumerate(categorical_cols):
    if col in new_data:
        col_val = f"{col}_{new_data[col]}"
        if col_val in encoded_df.columns:
            idx = np.where(encoded_df.columns == col_val)[0][0]
            new_input[0, idx] = 1  # Set corresponding feature to 1
new_input[0, -1] = new_data["Age"]  # Set Age feature

# Predict the outcome variable for the new input
prediction = clf.predict(new_input)

if st.sidebar.button("Predict"):
    # Predict the outcome variable for the new input
    prediction = clf.predict(new_input)
    # Display the prediction
    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"Predicted Outcome Variable: {prediction[0]}")


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

    if not all(symptom in tfidf_vectorizer.vocabulary_ for symptom in symptoms.split()):
        return "Some symptoms are not recognized. Please enter valid symptoms."

    # Calculate similarity scores
    cosine_scores = linear_kernel(symptoms_tfidf, tfidf_matrix).flatten()

    # Get indices of top 5 medicines with highest similarity scores
    top_indices = cosine_scores.argsort()[:-6:-1]

    # Get the names of top 5 recommended medicines
    top_medicines = data.iloc[top_indices]["Medicine Name"].values

    return top_medicines


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")
# adding background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url("data:image/jpg;base64,{img}");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
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
