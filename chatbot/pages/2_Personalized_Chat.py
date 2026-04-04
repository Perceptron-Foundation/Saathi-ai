from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime
import pandas as pd
import time

# 🔥 Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}

body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.chat-container {
    max-width: 800px;
    margin: auto;
    padding-bottom: 140px;
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

.block-container {
    padding-bottom: 120px;
}
</style>
""", unsafe_allow_html=True)

# 🔑 Gemini setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-flash-latest")

def get_response(prompt, chat):
    return chat.send_message(prompt).text

# 🎯 Title
st.markdown("<h2 style='text-align:center;'>📊 Personalized Saathi</h2>", unsafe_allow_html=True)

language = st.selectbox("🌍 Language", ["English", "Hindi"])

# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "glucose_history" not in st.session_state:
    st.session_state.glucose_history = []

if "last_glucose" not in st.session_state:
    st.session_state.last_glucose = None

# ---------------- GLUCOSE INPUT ----------------
st.markdown("### 🩺 Enter Glucose Level")

glucose = st.number_input("mg/dL", 50, 400, 120)

# 🚨 Instant alerts
if glucose < 70:
    st.warning("⚠️ Low sugar! Take glucose immediately.")
elif 70 <= glucose <= 180:
    st.success("✅ Normal range")
elif 180 < glucose <= 250:
    st.info("⚠️ Slightly high")
else:
    st.error("🚨 Very high! Monitor carefully")

# 📈 Store glucose with timestamp
if st.session_state.last_glucose != glucose:
    st.session_state.glucose_history.append({
        "time": datetime.now().strftime("%H:%M"),
        "glucose": glucose
    })
    st.session_state.last_glucose = glucose

# 🧠 Prompt
def personalized_prompt(q):
    return f"""
You are Saathi, a personalized diabetes assistant.

Glucose: {glucose} mg/dL

Give safe, simple advice in {language}.

Question: {q}
"""

# ---------------- CHAT DISPLAY ----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f'<div class="user-msg">🧑 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {text}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown('<div class="input-box">', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6,1])

    with col1:
        user_input = st.text_input("", placeholder="Ask about your health...", label_visibility="collapsed")

    with col2:
        submit = st.form_submit_button("➤")

    if submit and user_input:

        # Add user message
        st.session_state.chat_history.append(("You", user_input))

        # Typing animation
        with st.spinner("Saathi is thinking..."):
            full_response = get_response(personalized_prompt(user_input), st.session_state.chat)

        # Streaming effect
        
        message_placeholder = st.empty()
        streamed = ""

        for char in full_response:
            streamed += char
            message_placeholder.markdown(
                f'<div class="bot-msg">🤖 {streamed}</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.008)

        st.session_state.chat_history.append(("Bot", full_response))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- GRAPH ----------------
st.markdown("### 📊 Your Health Insights")
if st.session_state.glucose_history:

    st.markdown("### 📈 Glucose Trend")

    df = pd.DataFrame(st.session_state.glucose_history)
    df.set_index("time", inplace=True)

    st.line_chart(df)

    # 📊 Dashboard
    values = [item["glucose"] for item in st.session_state.glucose_history]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Latest", f"{values[-1]} mg/dL")

    with col2:
        st.metric("Average", f"{round(sum(values)/len(values),1)} mg/dL")


# 🧠 Daily AI Report
# 🧠 Daily AI Report (optimized)
if len(st.session_state.glucose_history) >= 3:

    values = [item["glucose"] for item in st.session_state.glucose_history]

    if "last_report_values" not in st.session_state:
        st.session_state.last_report_values = []

    if values != st.session_state.last_report_values:

        report_prompt = f"""
        You are a diabetes assistant.

        Glucose readings: {values}

        Analyze:
        - Trend
        - Give 2 short suggestions
        """

        with st.spinner("Generating daily report..."):
            report = get_response(report_prompt, st.session_state.chat)

        st.session_state.report = report
        st.session_state.last_report_values = values

    if "report" in st.session_state:
        st.markdown("### 🧠 Daily Health Report")
        st.success(st.session_state.report)

if len(st.session_state.glucose_history) >= 2:

    last = st.session_state.glucose_history[-1]["glucose"]
    prev = st.session_state.glucose_history[-2]["glucose"]

    if last > prev:
        st.warning("📈 Your sugar is increasing today")
    elif last < prev:
        st.success("📉 Your sugar is improving")
    else:
        st.info("⚖️ Stable trend")

st.markdown(
    "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
    unsafe_allow_html=True
)