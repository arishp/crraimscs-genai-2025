import streamlit as st
from utils import graph

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'ai', 'messages':['ask any question..']}]

for msg in st.session_state['messages']:
    with st.chat_message(msg['role'] if msg['role'] =='user' else 'ai'):
        for part in msg['messages']:
            st.write(part)

if input_query := st.chat_input():
    st.chat_message('user').write(input_query)
    st.session_state['messages'].append({'role': 'user', 'messages': [input_query]})
    try:
        input_prompt = {
                    "messages": [
                        ("user", {input_query}),
                    ]
                }
        response = graph.invoke(input_prompt)
        with st.chat_message('ai'):
            st.write(response)

    except Exception as e:
        print(e)
    
    # finally:
    #     st.session_state['messages'].append({'role': 'ai', 'messages': response})
