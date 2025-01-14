import streamlit as st

try:
    file_uploaded = st.file_uploader("Upload file")
except:
    st.write('Unable to create vector db')

user_query = st.text_input("Enter your query")

if len(user_query.strip()) > 0:
    st.write(user_query)
