import yfinance as yf
import pandas as pd
import os
import time

def fetch_stock_data(ticker: str, start_date: str, end_date: str, save_path: str = "data", retries: int = 2) -> pd.DataFrame:
    """
    Fetches historical stock data with robust crypto support and fallbacks.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame()

    def try_fetch(t, s, e, method="download"):
        try:
            if method == "download":
                df = yf.download(t, start=s, end=e, progress=False, timeout=15, auto_adjust=True)
            else:
                # Fallback for crypto/unstable tickers
                ticker_obj = yf.Ticker(t)
                df = ticker_obj.history(period="7d", interval="1d")
            
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Close' not in df.columns and len(df.columns) > 0:
                    df['Close'] = df.iloc[:, 0]
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    for attempt in range(retries + 1):
        # 1. Primary Method (Download)
        df = try_fetch(ticker, start_date, end_date, method="download")
        
        # 2. Secondary Method (History fallback for crypto)
        if df.empty and ("-USD" in ticker or "Crypto" in ticker):
            df = try_fetch(ticker, start_date, end_date, method="history")
            
        if not df.empty:
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, f"{ticker}_data.csv"))
            return df
            
        if attempt < retries:
            time.sleep(1.5) # Small delay between retries
                
    return pd.DataFrame()
