import pandas as pd
import re
from transformers import pipeline

# 1. Load the Entire Dataset
try:
    # Using 'utf-16' and 'on_bad_lines' to ensure all rows are read without crashing
    df = pd.read_csv(
        'Romanized Sinhala.csv', 
        header=None, 
        encoding='utf-16', 
        on_bad_lines='skip', 
        engine='python'
    )
    
    # Select only the first column and rename it
    df = df[[0]] 
    df.columns = ['Singlish']
    
    print(f"Step 1: File loaded! Total rows detected: {len(df)}")
except Exception as e:
    print(f"Error loading file: {e}")
    df = None

# 2. Data Preprocessing (Cleaning)
if df is not None:
    # Remove any completely empty rows
    df = df.dropna(subset=['Singlish'])

    def clean_text(text):
        text = str(text).lower() 
        text = re.sub(r'http\S+', '', text) 
        # Keep only letters and Sinhala script
        text = re.sub(r'[^\w\s\u0D80-\u0DFF]', '', text) 
        text = " ".join(text.split())
        return text

    df['cleaned_text'] = df['Singlish'].apply(clean_text)
    print("Step 2: Data cleaning completed!")

    # 3. Initialize the AI Model
    print("Step 3: Initializing AI Model... Please wait.")
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        )

        def get_sentiment(text):
            if not str(text).strip(): return "Neutral"
            # Limit text to 512 characters for the AI model
            result = sentiment_analyzer(str(text)[:512])[0]
            label = result['label'].lower()
            
            if 'negative' in label: return "Negative"
            elif 'neutral' in label: return "Neutral"
            else: return "Positive"

        # 4. Analyze All Rows
        print(f"Step 4: Analyzing all {len(df)} rows. This might take a minute...")
        
        # Apply the sentiment function to every single row
        df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

        # Keep only the original text and the result for the final file
        final_df = df[['Singlish', 'sentiment']]
        
        # Save to CSV
        output_file = 'Analyzed_Romanized_Sinhala_Final.csv' 
        # 'utf-8-sig' ensures Sinhala characters open correctly in Excel
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n--- Success! ---")
        print(f"Total rows analyzed: {len(final_df)}")
        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"\nAI Error: {e}")
else:
    print("Error: Could not load the data.")