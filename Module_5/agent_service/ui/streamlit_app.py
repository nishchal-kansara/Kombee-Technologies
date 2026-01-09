import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Agent Chat UI", layout="centered")

st.title("Agent Chatbot")
st.write("Chat with LangChain Agent using Gemini")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Enter your message")

if st.button("Send"):
    if user_input.strip() != "":
        payload = {
            "question": user_input,
            "session_id": st.session_state.session_id
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            answer = response.json()["answer"]

            st.session_state.chat_history.append(
                ("You", user_input)
            )
            st.session_state.chat_history.append(
                ("Agent", answer)
            )
        else:
            st.error("API error")

for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Agent:** {message}")
