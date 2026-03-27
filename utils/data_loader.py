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
            # Cloud-optimized yf.download call
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                timeout=20, 
                auto_adjust=True, 
                threads=False
            )
            
            # Handle MultiIndex columns (common in yfinance 0.2.x)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Log Data Shape for Debugging
            if df is not None and not df.empty:
                print(f"Successfully fetched {ticker}. Shape: {df.shape}")
                
                # Ensure 'Close' column exists
                if 'Close' not in df.columns:
                    print(f"Warning: 'Close' column missing for {ticker}. Available: {df.columns.tolist()}")
                    # Sometimes 'Adj Close' is returned if auto_adjust is false, but we set it true.
                    # Fallback to last column if 'Close' is missing but data exists
                    if len(df.columns) > 0:
                        df['Close'] = df.iloc[:, 0] 

                # Ensure directory exists
                os.makedirs(save_path, exist_ok=True)
                file_name = f"{ticker}_data.csv"
                df.to_csv(os.path.join(save_path, file_name))
                return df
                
            print(f"Data for {ticker} is empty on attempt {attempt + 1}.")
            
            # If specifically the date range fails, try a broader period on last attempt
            if attempt == retries - 1:
                print(f"Final fallback: Fetching 1y period for {ticker}...")
                df = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=20, auto_adjust=True, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    return df

        except Exception as e:
            print(f"Error on attempt {attempt + 1} for {ticker}: {e}")
            
        if attempt < retries - 1:
            time.sleep(2) # Wait before retry
                
    print(f"Critical: Ticker '{ticker}' failed after {retries} attempts.")
    return pd.DataFrame()
