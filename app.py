import streamlit as st
from retriever import VectorStore
from llm import answer_question
import pandas as pd
import dotenv
import os

dotenv.load_dotenv()
#import numpy as np

DATE_COLUMN = 'tweet_created'

st.session_state.date_column = DATE_COLUMN
st.session_state.mistral_api_key = os.getenv("MISTRAL_API_KEY")

st.title("Chatbot with ChromaDB + Mistral AI")

@st.cache_data
def load_data():
    df1 = pd.read_csv("data/Tweets.csv")
    df1[DATE_COLUMN] = pd.to_datetime(df1[DATE_COLUMN])
    return df1

# Define a function to set up session state
def setup_session_state():
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = load_data()

def load_twitter_data(size=1000):
    df = st.session_state.loaded_data
    data = df.head(size)['text'].tolist()
    return data

def main():
    setup_session_state()

    if "store" not in st.session_state:
        store = VectorStore()
        store.add_texts(load_twitter_data(100))
        st.session_state.store = store

    question = 'what airline is the worst?'
    docs = st.session_state.store.search(question)
    st.write("### Retrieved Documents:")
    for d in docs:
        st.write("- ", d)
    answer = answer_question(question, docs)
    st.write("### Answer:")
    st.write(answer)

main()

# question = st.text_input("Ask something:")

# if st.button("Ask"):
#     with st.spinner("Thinking..."):
#         docs = st.session_state.store.search(question)
#         answer = answer_question(question, docs)

#         st.write("### Answer:")
#         st.write(answer)

#         st.write("---")
#         st.write("### Retrieved Documents:")
#         for d in docs:
#             st.write("- ", d)
