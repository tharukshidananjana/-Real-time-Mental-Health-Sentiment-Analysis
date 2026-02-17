📊 Multi-Platform AI Sentiment Intelligence Dashboard
This project is a sophisticated Sentiment Analysis and Data Visualization Dashboard designed to extract actionable insights from customer reviews across major e-commerce platforms like Alibaba, Amazon, Daraz, and Walmart.

It utilizes a Multilingual AI model and a custom heuristic engine to accurately interpret feedback in both English and Romanized Sinhala (Singlish).

🚀Live Link : https://e-commerce-sentiment-dashboard.streamlit.app/

🌟 Key Features
Multilingual Analysis: Powered by the DistilBERT multilingual model to handle diverse customer languages.

Custom Heuristic Engine: Includes a specialized logic layer to correctly identify local Sri Lankan slang (e.g., niyamai, pattayi, maru) that standard AI models might miss.

Real-time Analysis Module: A dedicated interface for instant "on-the-fly" sentiment testing.

Dynamic Data Filtering: Users can filter datasets by sentiment (Positive, Negative, Neutral) to isolate specific business issues.

Visual Business Intelligence: * Pie Charts: For high-level sentiment distribution.

Word Clouds: For instant identification of trending keywords and customer pain points.

Report Exporting: Built-in functionality to export filtered analysis results as CSV files for corporate reporting.

🛠️ Tech Stack
Language: Python 3.x

Web Framework: Streamlit

Machine Learning: Hugging Face Transformers (PyTorch backend)

Data Processing: Pandas, NumPy

Visualization: Plotly Express, WordCloud, Matplotlib

🚀 Installation & Setup
1. Prerequisites
Ensure you have Python installed. Then, install the required libraries:

Bash

pip install streamlit pandas plotly transformers wordcloud matplotlib torch
2. Project Structure
Your folder should contain:

Dashboard.py (The main application)

requirements.txt (Dependency list)

Analysis scripts (e.g., converted_data.py, Romanized Sinhala.py)

Generated CSV files (e.g., Analyzed_Alibaba_Final.csv)

3. Running the Application
First, generate the analyzed data by running your specific platform scripts. Then, launch the dashboard:

Bash

streamlit run Dashboard.py
📈 Use Case: Business Impact
This tool allows a business manager to:

Identify Trends: See that "Delivery" is the most frequent word in Negative reviews for Daraz.

Verify Slang: Ensure that positive Singlish feedback like "eka maru" is correctly counted as a Positive sentiment.

Data Portability: Export the filtered "Negative" reviews to send to the Logistics team for improvement.

👤 Author
[H.K.Tharukshi Dhananjana] Aspiring Data Scientist