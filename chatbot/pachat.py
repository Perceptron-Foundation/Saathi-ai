from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
import pandas as pd
import streamlit as st
import os
import google.generativeai as genai

# Configure API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load model
model = genai.GenerativeModel("models/gemini-flash-latest")

def get_gemini_response(prompt, chat):
    response = chat.send_message(prompt)
    return response.text

# Page config
st.set_page_config(page_title="Saathi", page_icon="🤖", layout="centered")

# CSS
st.markdown("""
<style>
.chat-container { padding: 10px; }
.user-msg {
    background-color: #DCF8C6;
    color: black;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    text-align: right;
}
.bot-msg {
    background-color: #F1F0F0;
    color: black;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    text-align: left;
}
button[kind="primary"] {
    height: 42px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Saathi")
st.caption("Your Type 1 Diabetes Companion")

# Sidebar
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox("Choose Language", ["English", "Hindi"])
mode = st.sidebar.radio("Choose Mode", ["General Chat", "Personalized Chat"])

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.glucose_history = []

# Session state init
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "glucose_history" not in st.session_state:
    st.session_state.glucose_history = []

if "last_glucose" not in st.session_state:
    st.session_state.last_glucose = None

# Personalized input
glucose = None
if mode == "Personalized Chat":
    st.subheader("📊 Enter Your Health Data")

    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)",
        min_value=50,
        max_value=400,
        value=120
    )

    # Instant feedback
    if glucose < 70:
        st.warning("⚠️ Low sugar! Suggested: drink juice or glucose tablet")
    elif 70 <= glucose <= 180:
        st.success("✅ Normal range")
    elif 180 < glucose <= 250:
        st.info("⚠️ Slightly high, consider light activity")
    else:
        st.error("🚨 Very high! Monitor closely")

    # ✅ Store with timestamp (fixed)
    if st.session_state.last_glucose != glucose:
        st.session_state.glucose_history.append({
            "time": datetime.now().strftime("%H:%M"),
            "glucose": glucose
        })
        st.session_state.last_glucose = glucose

    # Insight
    insight_prompt = f"""
    You are Saathi, a diabetes assistant.

    Glucose level: {glucose} mg/dL

    Give 1 short, practical health tip.
    """

    with st.spinner("Generating insight..."):
        insight = get_gemini_response(insight_prompt, st.session_state.chat)

    st.info(f"💡 Saathi Insight: {insight}")

# Prompt functions
def general_prompt(user_input, language):
    return f"Answer in {language}. Question: {user_input}"

def personalized_prompt(user_input, glucose, language):
    return f"Glucose: {glucose} mg/dL. Answer in {language}. Question: {user_input}"

# Quick questions
st.subheader("💡 Try these questions")

def handle_quick_question(question):

    if mode == "General Chat":
        prompt = general_prompt(question, language)
    else:
        prompt = personalized_prompt(question, glucose, language)

    with st.spinner("Saathi is thinking..."):
        response = get_gemini_response(prompt, st.session_state.chat)

    # Add both question + answer
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Bot", response))


col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is Type 1 Diabetes?"):
        handle_quick_question("What is Type 1 Diabetes?")

with col2:
    if st.button("Low sugar kya karein?"):
        handle_quick_question("What should I do if my sugar is low?")

with col3:
    if st.button("Can I eat rice?"):
        handle_quick_question("Can I eat rice with diabetes?")

# Chat input (form)
st.markdown("### 💬 Ask something")

with st.form(key="chat_form", clear_on_submit=True):

    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input("", placeholder="Ask something...")

    with col2:
        submit = st.form_submit_button("➤")

    if submit and user_input:

        if mode == "General Chat":
            prompt = general_prompt(user_input, language)
        else:
            prompt = personalized_prompt(user_input, glucose, language)

        with st.spinner("Saathi is thinking..."):
            response = get_gemini_response(prompt, st.session_state.chat)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Chat UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f'<div class="user-msg">🧑 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {text}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Graph + Insights
if st.session_state.glucose_history:

    st.subheader("📈 Glucose Trend")

    df = pd.DataFrame(st.session_state.glucose_history)
    df.set_index("time", inplace=True)

    st.line_chart(df)

    # Trend insight
    if len(st.session_state.glucose_history) >= 2:
        last = st.session_state.glucose_history[-1]["glucose"]
        prev = st.session_state.glucose_history[-2]["glucose"]

        if last > prev:
            st.warning("📈 Sugar increased from last reading")
        elif last < prev:
            st.success("📉 Sugar improving")
        else:
            st.info("⚖️ Sugar stable")

    # Dashboard
    st.subheader("📊 Health Dashboard")

    values = [item["glucose"] for item in st.session_state.glucose_history]

    st.metric("Latest", f"{values[-1]} mg/dL")
    st.metric("Average", f"{round(sum(values)/len(values),1)} mg/dL")