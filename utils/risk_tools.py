import pandas as pd
import numpy as np

def calculate_risk_price_points(current_price: float, df: pd.DataFrame, direction: str = "BUY"):
    """
    Suggests Stop Loss and Target Profit based on ATR-like volatility.
    """
    if df.empty or len(df) < 20:
        volatility = current_price * 0.02 # Fallback 2%
    else:
        # Simple ATR approximation using rolling standard deviation
        volatility = df['Close'].rolling(window=20).std().iloc[-1]
    
    if direction == "BUY" or direction == "STRONG BUY":
        stop_loss = current_price - (volatility * 1.5)
        target_profit = current_price + (volatility * 3.0) # 1:2 Risk-Reward
    elif direction == "SELL" or direction == "STRONG SELL":
        stop_loss = current_price + (volatility * 1.5)
        target_profit = current_price - (volatility * 3.0)
    else:
        stop_loss = current_price * 0.95
        target_profit = current_price * 1.05

    return {
        "stop_loss": round(stop_loss, 2),
        "target_profit": round(target_profit, 2),
        "risk_amount": round(volatility * 1.5, 2)
    }
