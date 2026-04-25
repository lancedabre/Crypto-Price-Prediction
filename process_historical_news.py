import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from tqdm import tqdm

# Ensure langdetect gives consistent results
DetectorFactory.seed = 0

def detect_english(text):
    """Safely checks if a tweet is in English."""
    try:
        # langdetect crashes if the text is just a URL or empty space
        return detect(str(text)) == 'en'
    except:
        return False

def process_historical_data(csv_filename="bitcoin_tweets.csv"):
    print("🚀 Starting NLP Historical Processing...")

    # 1. Load the raw data
    print(f"Loading {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # Keep only what we need to save memory
    df = df[['date', 'text']].dropna()

    # 2. Filter for English tweets
    print("🌍 Detecting languages and filtering for English (this will take a few minutes)...")
    # Enable progress bar for pandas apply
    tqdm.pandas()
    
    # Create a mask of which rows are English
    is_english = df['text'].progress_apply(detect_english)
    df = df[is_english].copy()
    print(f"✅ Filtered down to {len(df)} English tweets.")

    # 3. Apply VADER Sentiment
    print("🧠 Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        return analyzer.polarity_scores(str(text))['compound']
        
    df['Sentiment_Score'] = df['text'].progress_apply(get_sentiment)

    # 4. Filter out exact 0.0 scores (removes bot noise/unrecognized slang)
    df = df[df['Sentiment_Score'] != 0.0]

    # 5. Enforce the Team Data Contract (Crucial for the ML merge!)
    print("📅 Formatting dates to match the ML Data Contract...")
    # Convert to datetime, strip timezones, and floor to midnight
    df['Date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.floor('D')
    df = df.dropna(subset=['Date']) # Drop any rows where date conversion failed
    
    # 6. Group by Date and calculate the daily average
    print("📊 Aggregating daily averages...")
    final_df = df.groupby('Date')[['Sentiment_Score']].mean()

    # 7. Save to CSV
    output_filename = "historical_sentiment.csv"
    final_df.to_csv(output_filename)
    print(f"🎉 Success! Saved exactly {len(final_df)} days of sentiment to {output_filename}")

if __name__ == "__main__":
    process_historical_data("bitcoin_tweets.csv") # Change this if your Kaggle file has a different name