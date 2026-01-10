import pandas as pd
import re
from transformers import pipeline

# 1. Load the Dataset
try:
    # Ensure 'Alibaba.csv' is in your project folder
    df = pd.read_csv('Alibaba.csv')
    print("Step 1: File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")

# 2. Data Preprocessing (Cleaning)
text_column = 'content' 

if text_column in df.columns:
    # Remove empty rows in the content column
    df = df.dropna(subset=[text_column])

    def clean_text(text):
        text = str(text).lower() 
        text = re.sub(r'http\S+', '', text) 
        # Keeps English and Sinhala characters, removes special symbols
        text = re.sub(r'[^\w\s\u0D80-\u0DFF]', '', text) 
        text = " ".join(text.split())
        return text

    # Apply cleaning to the entire column
    df['cleaned_review'] = df[text_column].apply(clean_text)
    print("Step 2: Data cleaning completed!")

    # ---------------------------------------------------------
    # 3. Sentiment Analysis (DistilBERT - Lightweight Model)
    # ---------------------------------------------------------

    print("Step 3: Initializing Lightweight AI Model... Please wait.")
    
    # Corrected Model ID: 'lxyuan/' prefix added to avoid unauthorized/not found errors
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    def get_sentiment(text):
        # AI processes the text
        result = sentiment_analyzer(str(text)[:512])[0]
        # DistilBERT returns labels like 'positive', 'neutral', or 'negative'
        label = result['label'].lower()
        
        if 'negative' in label:
            return "Negative"
        elif 'neutral' in label:
            return "Neutral"
        else:
            return "Positive"

    # ---------------------------------------------------------
    # 4. Full Dataset Analysis & Saving Results
    # ---------------------------------------------------------

    print("Step 4: Starting Sentiment Analysis on the ENTIRE dataset...")
    print("This model is faster and uses less memory.")
    
    # Apply sentiment analysis to all rows
    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

    # Save the final results to a new CSV file
    output_file = 'Analyzed_Alibaba_Final.csv' 
    df.to_csv(output_file, index=False)

    print(f"\n--- Process Completed Successfully! ---")
    print(f"Total rows analyzed: {len(df)}")
    print(f"Results saved to: {output_file}")

else:
    print(f"Error: Column '{text_column}' not found. Available columns: {list(df.columns)}")