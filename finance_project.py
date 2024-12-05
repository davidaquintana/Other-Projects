import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch stock data using Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.reset_index()
    stock_data = stock_data[["Date", "Close"]]
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    return stock_data

# Feature Engineering: Add lag features and rolling averages
def add_features(data, lags=5):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
    df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

# Train and evaluate model
def train_model(data, model_type='random_forest', test_size=0.2):
    X = data.drop(['Date', 'Close'], axis=1)
    y = data['Close']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'random_forest' or 'xgboost'.")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{model_type.capitalize()} RMSE: {rmse}")
    return model

# Predict next value
def predict_next(model, recent_data):
    return model.predict([recent_data])[0]

# Main logic
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'  # Example ticker
    start_date = "2020-01-01"
    end_date = "2023-12-01"
    
    # Fetch stock data
    print(f"Fetching data for {ticker}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Add features
    print("Adding features...")
    stock_data = add_features(stock_data)
    
    # Train model
    print("Training model...")
    model = train_model(stock_data, model_type='xgboost')  # Change to 'random_forest' if needed
    
    # Predict next value using the most recent data
    print("Predicting next value...")
    recent_data = stock_data.drop(['Date', 'Close'], axis=1).iloc[-1].values
    predicted_price = predict_next(model, recent_data)
    last_close = stock_data['Close'].iloc[-1]
    
    print(f"Last Close: {last_close}")
    print(f"Predicted Next Close: {predicted_price}")
    
    # Plotting results
    plt.figure(figsize=(8, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], label="Actual Prices")
    plt.axhline(y=predicted_price, color='r', linestyle='--', label="Predicted Next Close")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.title(f"{ticker} Stock Price Prediction")
    plt.show()
