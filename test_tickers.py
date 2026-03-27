import sys
import os
sys.path.append(os.getcwd())
from utils.data_loader import fetch_stock_data
from datetime import datetime, timedelta

tickers = ["AAPL", "TSLA", "BTC-USD"]
start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
end = datetime.now().strftime('%Y-%m-%d')

for t in tickers:
    print(f"Testing {t}...")
    df = fetch_stock_data(t, start, end)
    if not df.empty:
        print(f"✅ {t} success. Shape: {df.shape}")
    else:
        print(f"❌ {t} failed.")
