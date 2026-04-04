import streamlit as st

st.set_page_config(page_title="Saathi", page_icon="🤖")

st.markdown("<h1 style='text-align: center;'>🤖 Saathi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your AI Health Companion</p>", unsafe_allow_html=True)

st.markdown("### 🚀 Choose your experience")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_General_Chat.py", label="💬 General Chat")

with col2:
    st.page_link("pages/2_Personalized_Chat.py", label="📊 Personalized Chat")