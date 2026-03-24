import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str, save_path: str = "data") -> pd.DataFrame:
    """
    Fetches historical stock data with improved error handling.
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return pd.DataFrame()
            
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # If specific range fails, try period="1mo"
        if df.empty:
            df = stock.history(period="1mo")
            
        if df.empty:
            print(f"Warning: No historical data found for {ticker}")
            return pd.DataFrame()
        
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Save to CSV (cleaned name)
        file_name = f"{ticker}_data.csv"
        full_path = os.path.join(save_path, file_name)
        df.to_csv(full_path)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
