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
button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Saathi")
st.caption("Your Type 1 Diabetes Companion")

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "glucose_history" not in st.session_state:
    st.session_state.glucose_history = []

if "last_glucose" not in st.session_state:
    st.session_state.last_glucose = None

if "page" not in st.session_state:
    st.session_state.page = "general"

# ---------------- NAVBAR ----------------
st.markdown("###")

nav1, nav2, nav3 = st.columns([2,2,2])

with nav1:
    if st.button("💬 General Chat"):
        st.session_state.page = "general"

with nav2:
    if st.button("📊 Personalized Chat"):
        st.session_state.page = "personalized"

with nav3:
    language = st.selectbox("🌍 Language", ["English", "Hindi"])

# ---------------- PROMPTS ----------------
def general_prompt(user_input, language):
    return f"Answer in {language}. Question: {user_input}"

def personalized_prompt(user_input, glucose, language):
    return f"Glucose: {glucose} mg/dL. Answer in {language}. Question: {user_input}"

# ---------------- QUICK QUESTIONS ----------------
st.subheader("💡 Try these questions")

def handle_quick_question(question):
    if st.session_state.page == "general":
        prompt = general_prompt(question, language)
    else:
        prompt = personalized_prompt(question, glucose, language)

    with st.spinner("Saathi is thinking..."):
        response = get_gemini_response(prompt, st.session_state.chat)

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

# ---------------- PERSONALIZED SECTION ----------------
glucose = None

if st.session_state.page == "personalized":
    st.subheader("📊 Enter Your Health Data")

    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)",
        min_value=50,
        max_value=400,
        value=120
    )

    # Instant feedback
    if glucose < 70:
        st.warning("⚠️ Low sugar! Take quick glucose")
    elif 70 <= glucose <= 180:
        st.success("✅ Normal range")
    elif 180 < glucose <= 250:
        st.info("⚠️ Slightly high")
    else:
        st.error("🚨 Very high!")

    # Store glucose
    if st.session_state.last_glucose != glucose:
        st.session_state.glucose_history.append({
            "time": datetime.now().strftime("%H:%M"),
            "glucose": glucose
        })
        st.session_state.last_glucose = glucose

    # AI Insight
    insight_prompt = f"""
    Glucose level: {glucose}
    Give 1 short health tip.
    """

    with st.spinner("Generating insight..."):
        insight = get_gemini_response(insight_prompt, st.session_state.chat)

    st.info(f"💡 Insight: {insight}")

# ---------------- CHAT INPUT ----------------
st.markdown("### 💬 Ask something")

with st.form("chat_form", clear_on_submit=True):

    col1, col2 = st.columns([5,1])

    with col1:
        user_input = st.text_input("", placeholder="Ask something...")

    with col2:
        submit = st.form_submit_button("➤")

    if submit and user_input:

        if st.session_state.page == "general":
            prompt = general_prompt(user_input, language)
        else:
            prompt = personalized_prompt(user_input, glucose, language)

        with st.spinner("Saathi is thinking..."):
            response = get_gemini_response(prompt, st.session_state.chat)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# ---------------- CHAT DISPLAY ----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f'<div class="user-msg">🧑 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {text}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- GRAPH + DASHBOARD ----------------
if st.session_state.page == "personalized" and st.session_state.glucose_history:

    st.subheader("📈 Glucose Trend")

    df = pd.DataFrame(st.session_state.glucose_history)
    df.set_index("time", inplace=True)

    st.line_chart(df)

    # Trend insight
    if len(st.session_state.glucose_history) >= 2:
        last = st.session_state.glucose_history[-1]["glucose"]
        prev = st.session_state.glucose_history[-2]["glucose"]

        if last > prev:
            st.warning("📈 Sugar increased")
        elif last < prev:
            st.success("📉 Sugar improving")
        else:
            st.info("⚖️ Stable")

    # Dashboard
    st.subheader("📊 Health Dashboard")

    values = [item["glucose"] for item in st.session_state.glucose_history]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Latest", f"{values[-1]} mg/dL")

    with col2:
        st.metric("Average", f"{round(sum(values)/len(values),1)} mg/dL")