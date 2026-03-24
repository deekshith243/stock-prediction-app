import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_stock_data(ticker: str, start_date: str, end_date: str, save_path: str = "data") -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance and saves it to a CSV file.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TCS.NS').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_path (str): Directory to save the CSV file.
        
    Returns:
        pd.DataFrame: The fetched historical data.
    """
    try:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol.")
        
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Save to CSV
        file_name = f"{ticker}_{start_date}_{end_date}.csv".replace("-", "")
        full_path = os.path.join(save_path, file_name)
        df.to_csv(full_path)
        print(f"Data saved successfully to {full_path}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the fetcher
    test_df = fetch_stock_data("AAPL", "2023-01-01", "2023-12-31")
    if not test_df.empty:
        print(test_df.head())
