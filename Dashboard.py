import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import os

# Page Configuration
st.set_page_config(page_title="AI Sentiment Analysis Dashboard", layout="wide")

# 1. Load the Sentiment AI Model (Cached for performance)
@st.cache_resource
def load_model():
    # Using a high-performance multilingual model
    return pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

analyzer = load_model()

# --- Custom Logic to Refine Singlish/English Results ---
def refined_analyze(text):
    # Added common positive keywords for better accuracy
    positive_keywords = ['niyamai', 'lassanai', 'sathutui', 'hondayi', 'good', 'super', 'pattayi', 'love', 'maru']
    text_lower = text.lower()
    
    # Analyze first 512 characters
    result = analyzer(text[:512])[0]
    label = result['label'].upper()
    score = result['score']
    
    # Manual override for positive keywords if model misclassifies
    for word in positive_keywords:
        if word in text_lower and "NEGATIVE" in label:
            return "POSITIVE (Verified)", 0.90
            
    return label, score

# 2. Sidebar Navigation
st.sidebar.title("Project Controls")
page = st.sidebar.radio("Select Module:", ["Real-time Analysis", "Data Dashboard"])

# --- Page 1: Real-time Analysis ---
if page == "Real-time Analysis":
    st.title("🧠 AI Sentiment & Mental Health Monitor")
    user_input = st.text_area("Input Text:", placeholder="Example: The service was excellent and very fast!")

    if st.button("Analyze Now"):
        if user_input.strip():
            with st.spinner('AI is processing...'):
                label, score = refined_analyze(user_input)

            if "POSITIVE" in label:
                st.success(f"Outcome: **POSITIVE** (Confidence: {score:.2f})")
                st.balloons()
            elif "NEGATIVE" in label:
                st.error(f"Outcome: **NEGATIVE / DEPRESSIVE SIGNS** (Confidence: {score:.2f})")
            else:
                st.info(f"Outcome: **NEUTRAL** (Confidence: {score:.2f})")
        else:
            st.warning("Input area is empty. Please enter some text.")

#