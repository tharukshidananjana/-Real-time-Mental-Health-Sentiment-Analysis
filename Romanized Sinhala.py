import pandas as pd
import re
from transformers import pipeline

# 1. Load the Dataset
try:
    # Since this file has no headers, we load it with header=None
    df = pd.read_csv('Romanized Sinhala.csv', header=None)
    
    # We rename the first column (Column 0) to 'Singlish' for easier processing
    df = df.rename(columns={0: 'Singlish'})
    
    print("Step 1: File loaded and column renamed successfully!")
except Exception as e:
    print(f"Error loading file: {e}")

# 2. Data Preprocessing (Cleaning)
text_column = 'Singlish' 

if 'df' in locals() and text_column in df.columns:
    # Remove empty rows to avoid errors
    df = df.dropna(subset=[text_column])

    def clean_text(text):
        text = str(text).lower() 
        text = re.sub(r'http\S+', '', text) 
        # Keep English letters and Sinhala characters, remove special symbols
        text = re.sub(r'[^\w\s\u0D80-\u0DFF]', '', text) 
        text = " ".join(text.split())
        return text

    # Store the cleaned text in a new column
    df['cleaned_text'] = df[text_column].apply(clean_text)
    print("Step 2: Data cleaning completed!")

    # ---------------------------------------------------------
    # 3. Sentiment Analysis (Lightweight DistilBERT Model)
    # ---------------------------------------------------------

    print("Step 3: Initializing Lightweight AI Model... Please wait.")
    
    # Using the best-suited multilingual model for Singlish analysis
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    def get_sentiment(text):
        # AI processes the text (max 512 characters)
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
    
    # Apply the analysis to the entire dataframe
    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    # Save the processed results to a new CSV file
    output_file = 'Analyzed_Romanized_Sinhala_Final.csv' 
    df.to_csv(output_file, index=False)

    print(f"\n--- Process Completed Successfully! ---")
    print(f"Total rows analyzed: {len(df)}")
    print(f"Final results saved to: {output_file}")

else:
    print(f"Error: Could not find valid data in 'Romanized Sinhala.csv' to analyze.")