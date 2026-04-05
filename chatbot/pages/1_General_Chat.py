from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import google.generativeai as genai
import os
import time

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}

body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.chat-container {
    max-width: 800px;
    margin: auto;
    padding-bottom: 120px;
}

.user-msg {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    padding: 14px;
    border-radius: 18px;
    margin: 10px 0;
    text-align: right;
    max-width: 70%;
    margin-left: auto;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.bot-msg {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(10px);
    color: white;
    padding: 14px;
    border-radius: 18px;
    margin: 10px 0;
    max-width: 70%;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.input-box {
    position: fixed;
    bottom: 20px;
    left: 0;
    right: 0;
    max-width: 800px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

api_key = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
# Gemini setup
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-flash-latest")

def get_response(prompt, chat):
    return chat.send_message(prompt).text

# Title
st.markdown("<h2 style='text-align:center;'>💬 Saathi</h2>", unsafe_allow_html=True)

language = st.selectbox("🌍 Language", ["English", "Hindi"])

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def general_prompt(q):
    return f"Answer in {language}. Question: {q}"

# Chat display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f'<div class="user-msg">🧑 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {text}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input
st.markdown('<div class="input-box">', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6,1])

    with col1:
        user_input = st.text_input("", placeholder="Message Saathi...", label_visibility="collapsed")

    with col2:
        submit = st.form_submit_button("➤")

    if submit and user_input:

        # Add user message
        st.session_state.chat_history.append(("You", user_input))

        # Typing animation
        with st.spinner("Saathi is typing..."):
            full_response = get_response(general_prompt(user_input), st.session_state.chat)

        # Streaming effect
        streamed = ""
        placeholder = st.empty()

        for char in full_response:
            streamed += char
            placeholder.markdown(f'<div class="bot-msg">🤖 {streamed}</div>', unsafe_allow_html=True)
            time.sleep(0.01)

        st.session_state.chat_history.append(("Bot", full_response))

st.markdown('</div>', unsafe_allow_html=True)