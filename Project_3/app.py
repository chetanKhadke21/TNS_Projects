import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=False)  
def load_summarizer():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_summarizer()# Streamlit UI
st.title("Text Summarization App")
st.write("Enter your text below and get a concise summary:")

# User input
user_input = st.text_area("Enter text here", height=200)

if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text to summarize!")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False)
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])