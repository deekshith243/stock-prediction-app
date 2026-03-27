import pandas as pd
import numpy as np

def backtest_strategy(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Simulates a basic trading strategy based on RSI and MACD signals.
    """
    if df.empty or 'RSI' not in df.columns or 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
        return {
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "equity_curve": [initial_capital],
            "success": False,
            "error": "Required indicators (RSI, MACD, MACD_Signal) missing."
        }

    capital = initial_capital
    position = 0
    trades = 0
    wins = 0
    equity_curve = [initial_capital]
    buy_price = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # RELAXED strategy for demonstration: 
        # Buy if RSI < 45 OR MACD > Signal
        if position == 0 and (row['RSI'] < 45 or row['MACD'] > row['MACD_Signal']):
            position = capital / row['Close']
            capital = 0
            buy_price = row['Close']
            trades += 1
            
        # Sell if RSI > 55 OR MACD < Signal
        elif position > 0 and (row['RSI'] > 55 or row['MACD'] < row['MACD_Signal']):
            capital = position * row['Close']
            if row['Close'] > buy_price:
                wins += 1
            position = 0
            
        # Update equity curve
        current_equity = capital if position == 0 else position * row['Close']
        equity_curve.append(current_equity)

    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    return {
        "initial_capital": initial_capital,
        "final_capital": equity_curve[-1],
        "total_return_pct": total_return * 100,
        "win_rate": win_rate,
        "total_trades": trades,
        "equity_curve": equity_curve,
        "success": True
    }
