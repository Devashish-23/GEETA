import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Bhagavad Gita AI")

# LOAD CSV
df = pd.read_csv("Bhagwad_Gita.csv.csv")

# SHOW COLUMN NAMES
st.write("CSV Columns:", df.columns.tolist())

# USE FIRST FEW TEXT COLUMNS
text_columns = df.select_dtypes(include='object').columns.tolist()

# FILL EMPTY VALUES
for col in text_columns:
    df[col] = df[col].fillna("")

# COMBINE TEXT
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# TFIDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined_text"])

# SEARCH FUNCTION
def search(query):

    query_vector = vectorizer.transform([query])

    similarity = cosine_similarity(query_vector, X).flatten()

    top_index = similarity.argmax()

    return df.iloc[top_index]

# UI
st.title("🕉️ Bhagavad Gita Chatbot")

query = st.text_input("Ask a question")

if query:

    result = search(query)

    st.write(result)