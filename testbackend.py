import pandas as pd
import yfinance as yf
import joblib

# Import your team's modules!
from nlp_pipeline import get_daily_sentiment
from ml_pipeline import prepare_live_features

def run_integration_test():
    print("\n🚀 Starting Backend Integration Test...\n")

    # ---------------------------------------------------------
    # 1. Load the ML Model
    # ---------------------------------------------------------
    print("[1/4] Loading XGBoost Model...")
    try:
        model = joblib.load('models/xgboost_model.pkl')
        print("      ✅ Model loaded successfully!")
    except Exception as e:
        print(f"      ❌ Failed to load model: {e}")
        print("      Did you remember to run 'python3 ml_pipeline.py' first?")
        return

    # ---------------------------------------------------------
    # 2. Test the NLP Pipeline
    # ---------------------------------------------------------
    print("\n[2/4] Fetching Live News Sentiment (Testing NLP)...")
    try:
        sentiment_df = get_daily_sentiment()
        # Grab the sentiment score from the very last row (today)
        current_sentiment = float(sentiment_df['Sentiment_Score'].iloc[-1])
        print(f"      ✅ Sentiment processed. Latest score: {current_sentiment:.3f}")
    except Exception as e:
        print(f"      ❌ NLP Pipeline failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. Test Market Data Fetching
    # ---------------------------------------------------------
    print("\n[3/4] Fetching Live Market Data...")
    try:
        # Fetch data quietly
        df = yf.download("BTC-USD", period="20d", progress=False)
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
        df['Daily_Return'] = df['Close'].pct_change()

        # Extract today's metrics (Handling yfinance format quirks safely)
        latest = df.iloc[-1]
        current_price = float(latest['Close'].iloc[0] if isinstance(latest['Close'], pd.Series) else latest['Close'])
        current_volume = float(latest['Volume'].iloc[0] if isinstance(latest['Volume'], pd.Series) else latest['Volume'])
        sma_7 = float(latest['SMA_7'].iloc[0] if isinstance(latest['SMA_7'], pd.Series) else latest['SMA_7'])
        sma_14 = float(latest['SMA_14'].iloc[0] if isinstance(latest['SMA_14'], pd.Series) else latest['SMA_14'])
        daily_return = float(latest['Daily_Return'].iloc[0] if isinstance(latest['Daily_Return'], pd.Series) else latest['Daily_Return'])
        
        print(f"      ✅ Market data fetched. Current BTC Price: ${current_price:,.2f}")
    except Exception as e:
        print(f"      ❌ Market Data fetch failed: {e}")
        return

    # ---------------------------------------------------------
    # 4. Test the Prediction Engine
    # ---------------------------------------------------------
    print("\n[4/4] Executing Prediction Engine...")
    try:
        # Pass everything into YOUR function
        live_features = prepare_live_features(
            current_price=current_price,
            current_volume=current_volume,
            sma_7=sma_7,
            sma_14=sma_14,
            daily_return=daily_return,
            yesterday_sentiment=current_sentiment
        )
        
        # Get prediction and confidence probability
        prediction = model.predict(live_features)[0]
        probability = model.predict_proba(live_features)[0]

        # ---------------------------------------------------------
        # Print the Final Output
        # ---------------------------------------------------------
        print("\n" + "="*45)
        print(" 🎯 INTEGRATION TEST RESULTS")
        print("="*45)
        if prediction == 1:
            print(f" Forecast:   UPWARD 🟢")
            print(f" Confidence: {probability[1]*100:.1f}%")
        else:
            print(f" Forecast:   DOWNWARD 🔴")
            print(f" Confidence: {probability[0]*100:.1f}%")
        print("="*45 + "\n")

    except Exception as e:
        print(f"      ❌ Prediction failed: {e}")

if __name__ == "__main__":
    run_integration_test()