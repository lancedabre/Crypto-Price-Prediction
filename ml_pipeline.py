import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import GridSearchCV
from nlp_pipeline import get_daily_sentiment
# ---------------------------------------------------------
# 1. DATA FETCHING & CONTRACT ENFORCEMENT
# ---------------------------------------------------------
def fetch_historical_data(ticker="BTC-USD", start="2021-02-06", end="2024-01-01"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end)
    
    # --- 🛠️ THE FIX: CLEAN UP YFINANCE COLUMNS ---
    # 1. If yfinance gave us MultiIndex columns (e.g. 'Close', 'BTC-USD'), keep only the top level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Strip any weird trailing spaces from the column names
    df.columns = df.columns.astype(str).str.strip()
    # ----------------------------------------------
    
    # ENFORCING THE DATA CONTRACT
    df.index = pd.to_datetime(df.index).tz_localize(None).floor('D')
    df.index.name = 'Date'
    
    # Keep only what we need
    df = df[['Close', 'Volume']]
    return df



# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
def engineer_features(df):
    print("Engineering technical features...")
    
    # Calculate Simple Moving Averages (SMA)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    
    # NEW: Calculate the Ratio (Distance from the SMA) instead of absolute price
    df['Price_to_SMA7'] = df['Close'] / df['SMA_7']
    df['Price_to_SMA14'] = df['Close'] / df['SMA_14']
    
    # Calculate daily returns (percentage change)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # NEW: Calculate Volatility (How crazy is the market right now?)
    df['Volatility_7d'] = df['Daily_Return'].rolling(window=7).std()
    
    # Create Lagged Features
    df['Return_Lag1'] = df['Daily_Return'].shift(1)
    df['Sentiment_Lag1'] = df['Sentiment_Score'].shift(1)
    
    # The TARGET: 1 if tomorrow's price is higher than today's, 0 if lower
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    return df

# ---------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------
def train_model():
    # 1. Fetch historical price data
    df = fetch_historical_data()
    
    # --- 🛠️ THE FIX: LOAD THE KAGGLE CSV ---
    print("Loading historical sentiment data...")
    # Read the CSV your NLP teammate made
    sentiment_df = pd.read_csv('historical_sentiment.csv')
    
    # Enforce the Data Contract on the loaded CSV (just to be safe)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df.set_index('Date', inplace=True)
    # ----------------------------------------
    
    # 3. Merge them together! 
    # An 'inner' join is now safe because both dataframes have years of history.
    df = df.join(sentiment_df, how='inner') 
    
    df = engineer_features(df)
    
    
    # 2. Define our features (X) and target (y)
    # 2. Define our features (X) and target (y)
    features = ['Return_Lag1', 'Volume', 'Price_to_SMA7', 'Price_to_SMA14', 'Volatility_7d', 'Sentiment_Lag1']
    X = df[features]
    y = df['Target']
    
    # 3. Chronological Train/Test Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 4. Train the XGBoost Classifier
    print("Training XGBoost Model...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate basic accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model trained! Test Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Save the model for Member 3 (Deployment Lead) to use
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    print("Model saved to models/xgboost_model.pkl")

# ---------------------------------------------------------
# 4. LIVE INFERENCE (For Member 3 to use in the app)
# ---------------------------------------------------------
def prepare_live_features(current_price, current_volume, sma_7, sma_14, daily_return, yesterday_sentiment, volatility_7d, return_lag1):
    """Takes live data from the app and formats it for the model."""
    live_data = pd.DataFrame([{
        'Return_Lag1': return_lag1,
        'Volume': current_volume,
        'Price_to_SMA7': current_price / sma_7 if sma_7 else 1,
        'Price_to_SMA14': current_price / sma_14 if sma_14 else 1,
        'Volatility_7d': volatility_7d,
        'Sentiment_Lag1': yesterday_sentiment
    }])
    return live_data

# Run the training script if this file is executed directly
if __name__ == "__main__":
    train_model()