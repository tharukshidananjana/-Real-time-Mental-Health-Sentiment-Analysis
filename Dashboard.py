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
    return pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

analyzer = load_model()

# --- Custom Logic to Refine Singlish Results ---
def refined_analyze(text):
    positive_keywords = ['niyamai', 'lassanai', 'sathutui', 'hondayi', 'good', 'super', 'pattayi', 'love', 'maru']
    text_lower = text.lower()
    
    result = analyzer(text[:512])[0]
    label = result['label'].upper()
    score = result['score']
    
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
    user_input = st.text_area("Input Text:", placeholder="Example: Oya hari lassanai!")

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
            st.warning("Input area is empty.")

# --- Page 2: Data Visualization Dashboard ---
elif page == "Data Dashboard":
    st.title("📊 Multi-Platform Dataset Insights")
    
    datasets = {
        "Daraz Reviews": "Analyzed_Daraz online shopping App_Final.csv",
        "Walmart Reviews": "Analyzed_Walmart_Final.csv",
        "Shein Reviews": "Analyzed_Shein_Final.csv",
        "Alibaba Reviews": "Analyzed_Alibaba_Final.csv",
        "AliExpress Reviews": "Analyzed_Aliexpress_Final.csv",
        "Amazon Reviews": "Analyzed_Amazon_shopping_Final.csv",
        "Singlish Converted Data": "Analyzed_Converted_Data_Final.csv",
        "Romanized Sinhala": "Analyzed_Romanized_Sinhala_Final.csv"
    }

    dataset_choice = st.selectbox("Select Dataset:", list(datasets.keys()))
    file_path = datasets[dataset_choice]

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.metric("Total Records Processed", len(df))
        
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Total Count']

            fig = px.pie(sentiment_counts, values='Total Count', names='Sentiment', 
                         hole=0.4, color='Sentiment', 
                         color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
            
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View Raw Data"):
                st.dataframe(df)
        else:
            st.error("Sentiment column missing.")
    else:
        st.error(f"File not found: {file_path}")