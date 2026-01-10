import pandas as pd
import re
from transformers import pipeline

# 1. Load the Dataset
try:
    # Loading the 'converted_data.csv' file
    df = pd.read_csv('converted_data.csv')
    print("Step 1: File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")

# 2. Data Preprocessing (Cleaning)
# Selecting the 'Singlish' column for analysis
text_column = 'Singlish' 

if 'df' in locals() and text_column in df.columns:
    # Remove empty rows in the selected column
    df = df.dropna(subset=[text_column])

    def clean_text(text):
        text = str(text).lower() 
        text = re.sub(r'http\S+', '', text) 
        # Keeps English and Sinhala characters, removes special symbols
        text = re.sub(r'[^\w\s\u0D80-\u0DFF]', '', text) 
        text = " ".join(text.split())
        return text

    # Apply cleaning and store in a new column
    df['cleaned_text'] = df[text_column].apply(clean_text)
    print("Step 2: Data cleaning completed!")

    # ---------------------------------------------------------
    # 3. Sentiment Analysis (Lightweight DistilBERT Model)
    # ---------------------------------------------------------

    print("Step 3: Initializing Lightweight AI Model... Please wait.")
    
    # Using a fast and efficient multilingual AI model
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    def get_sentiment(text):
        # AI processes the text (up to 512 characters)
        result = sentiment_analyzer(str(text)[:512])[0]
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
    print("Note: This may take a few minutes for large files (30,000+ rows).")
    
    # Apply sentiment analysis to every row
    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    # Save final results to a new CSV file
    output_file = 'Analyzed_Converted_Data_Final.csv' 
    df.to_csv(output_file, index=False)

    print(f"\n--- Process Completed Successfully! ---")
    print(f"Total rows analyzed: {len(df)}")
    print(f"Final results saved to: {output_file}")

else:
    if 'df' in locals():
        print(f"Error: Column '{text_column}' not found. Available columns: {list(df.columns)}")