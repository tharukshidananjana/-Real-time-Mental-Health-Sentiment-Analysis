import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 2. Load AI Model (Cached for optimized performance) ---
@st.cache_resource
def load_sentiment_model():
    """
    Loads the multilingual DistilBERT model. 
    @st.cache_resource ensures the model is only loaded once into memory.
    """
    return pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

analyzer = load_sentiment_model()

# --- 3. Refined Analysis Logic ---
def get_refined_sentiment(text):
    """
    Handles Singlish/English nuances by combining AI predictions 
    with a keyword heuristic override for better accuracy.
    """
    positive_slang = ['niyamai', 'lassanai', 'sathutui', 'hondayi', 'good', 'super', 'pattayi', 'love', 'maru']
    text_lower = text.lower()
    
    # Run AI Analysis
    result = analyzer(text[:512])[0]
    label = result['label'].upper()
    score = result['score']
    
    # Heuristic: Override AI if specific positive keywords are present
    for word in positive_slang:
        if word in text_lower and "NEGATIVE" in label:
            return "POSITIVE (Verified)", 0.95
            
    return label, score

# --- 4. Sidebar Navigation ---
st.sidebar.title("Project Controls")
app_mode = st.sidebar.radio("Select Module:", ["Real-time Analysis", "Data Dashboard"])

# --- MODULE 1: Real-time Analysis ---
if app_mode == "Real-time Analysis":
    st.title("🧠 Real-time Sentiment Intelligence")
    st.markdown("Instantly analyze customer emotions in English or Singlish.")
    
    user_input = st.text_area("Input User Review:", placeholder="e.g., Delivery eka niyamai, thanks!")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            label, confidence = get_refined_sentiment(user_input)
            
            # UI Feedback based on sentiment
            if "POSITIVE" in label:
                st.success(f"Sentiment: {label} | Confidence: {confidence:.2f}")
                st.balloons()
            elif "NEGATIVE" in label:
                st.error(f"Sentiment: {label} | Confidence: {confidence:.2f}")
            else:
                st.info(f"Sentiment: {label} | Confidence: {confidence:.2f}")
        else:
            st.warning("Please enter some text to begin analysis.")

# --- MODULE 2: Data Dashboard ---
elif app_mode == "Data Dashboard":
    st.title("📊 Strategic Business Insights")
    st.markdown("Interactive visualization of customer feedback across multi-platform datasets.")

    # Dictionary mapping for all analyzed CSV files
    data_files = {
        "Alibaba": "Analyzed_Alibaba_Final.csv",
        "Walmart": "Analyzed_Walmart_Final.csv",
        "Shein": "Analyzed_Shein_Final.csv",
        "Amazon": "Analyzed_Amazon shopping_Final.csv",
        "AliExpress": "Analyzed_Aliexpress_Final.csv",
        "Daraz": "Analyzed_Daraz online shopping App_Final.csv",
        "Romanized Sinhala": "Analyzed_Romanized_Sinhala_Final.csv",
        "Converted Data": "Analyzed_Converted_Data_Final.csv"
    }
    
    platform = st.selectbox("Select Dataset to Visualize:", list(data_files.keys()))
    file_path = data_files[platform]

    if os.path.exists(file_path):
        # Load Data
        df = pd.read_csv(file_path)
        df['sentiment'] = df['sentiment'].str.upper()

        # Sidebar Filters
        st.sidebar.subheader("Dashboard Filters")
        sentiments = df['sentiment'].unique().tolist()
        selected_sentiments = st.sidebar.multiselect("Filter by Sentiment:", sentiments, default=sentiments)
        
        filtered_df = df[df['sentiment'].isin(selected_sentiments)]

        # --- High-Level KPIs ---
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Reviews Analyzed", f"{len(filtered_df):,}")
        kpi2.metric("Platform Name", platform)
        kpi3.metric("Engine", "DistilBERT-ML")

        st.divider()

        # --- Visualizations ---
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Sentiment Distribution")
            fig = px.pie(
                filtered_df, 
                names='sentiment', 
                hole=0.4, 
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Word Cloud: Trending Keywords")
            
            # Logic to find the correct text column for the Word Cloud
            # We look for common review column names used in your preprocessing
            possible_text_cols = ['cleaned_review', 'Singlish', 'cleaned_text', 'review_body']
            target_col = next((c for c in possible_text_cols if c in filtered_df.columns), None)
            
            if target_col:
                text_corpus = " ".join(filtered_df[target_col].astype(str))
            else:
                # Fallback: Use the column before the last one (usually where the review sits)
                text_corpus = " ".join(filtered_df.iloc[:, -2].astype(str))
            
            if text_corpus.strip() and len(text_corpus) > 10:
                wc = WordCloud(background_color='white', width=800, height=400, colormap='viridis').generate(text_corpus)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("No sufficient text data found to generate a Word Cloud.")

        # --- Detailed Data View ---
        st.subheader("Raw Analyzed Data Explorer")
        st.dataframe(filtered_df, use_container_width=True)

        # --- Exporting Results ---
        csv_binary = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Analysis as CSV",
            data=csv_binary,
            file_name=f"{platform}_Sentiment_Report.csv",
            mime='text/csv'
        )
    else:
        st.error(f"Dataset for {platform} not found. Ensure the analysis script has been executed.")