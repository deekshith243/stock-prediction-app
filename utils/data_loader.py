import yfinance as yf
import pandas as pd
import os
import time

def fetch_stock_data(ticker: str, start_date: str, end_date: str, save_path: str = "data", retries: int = 3) -> pd.DataFrame:
    """
    Fetches historical stock data using yf.download() with retry logic for reliability.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            print(f"Fetching {ticker} (Attempt {attempt + 1})...")
            # Use yf.download for better reliability than Ticker.history()
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=15)
            
            # Handle MultiIndex columns (common in yfinance 0.2.x)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not df.empty:
                # Ensure the save directory exists
                os.makedirs(save_path, exist_ok=True)
                # Save to CSV
                file_name = f"{ticker}_data.csv"
                full_path = os.path.join(save_path, file_name)
                df.to_csv(full_path)
                return df
                
            # If specific range fails, try a broader period (1y or 2y)
            if attempt == retries - 1:
                print(f"Attempting fallback period for {ticker}...")
                df = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=15)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                if not df.empty:
                    return df

        except Exception as e:
            print(f"Error on attempt {attempt + 1} for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2) # Wait before retry
                
    print(f"No data found for {ticker} after {retries} attempts.")
    return pd.DataFrame()
