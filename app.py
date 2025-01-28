import streamlit as st

user_query = st.text_input("Enter your query")

if len(user_query.strip()) > 0:
    st.write(user_query)
