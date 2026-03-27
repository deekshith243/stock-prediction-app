import pandas as pd
import numpy as np

def backtest_strategy(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Simulates a basic trading strategy based on RSI and MACD signals.
    Returns performance metrics and equity curve.
    """
    if df.empty or 'RSI' not in df.columns or 'MACD' not in df.columns:
        return {"error": "Technical indicators missing for backtesting"}

    capital = initial_capital
    position = 0
    trades = 0
    wins = 0
    equity_curve = [initial_capital]
    
    # Simple strategy: 
    # Buy if RSI < 35 and MACD > MACD_Signal
    # Sell if RSI > 65 or MACD < MACD_Signal
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # BUY signal
        if position == 0 and row['RSI'] < 35 and row['MACD'] > row['MACD_Signal']:
            position = capital / row['Close']
            capital = 0
            buy_price = row['Close']
            trades += 1
            
        # SELL signal
        elif position > 0 and (row['RSI'] > 65 or row['MACD'] < row['MACD_Signal']):
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
