import streamlit as st
import l6_vectorstore as backend

# General page description
st.title("Youtube Video Chatbot")
st.write("This is a chatbot that answers your questions about Youtube videos. Enter a Youtube video URL and a question to start.")

# Get user input
with st.sidebar:
    with st.form(key='user_input'):
        url = st.text_input("Youtube Video URL", max_chars=100)
        question = st.text_input("Question", max_chars=100)
        submit = st.form_submit_button("Submit")

# Get answer
if (url and question and submit):
    placeholder = st.empty()
    with placeholder:
        st.write("Loading and reading video transcript. This usually takes 20 seconds.")
        answer = backend.get_answer(question, url)
        st.write("Answer: " + answer)