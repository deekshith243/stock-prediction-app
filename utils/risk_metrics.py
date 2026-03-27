import numpy as np
import pandas as pd

def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """Calculates the maximum drawdown from peak."""
    if df.empty or 'Close' not in df.columns:
        return 0.0
    
    rolling_max = df['Close'].cummax()
    daily_drawdown = df['Close'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    return round(float(max_drawdown), 4)

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
    """Calculates the annualized Sharpe Ratio."""
    if df.empty or 'Close' not in df.columns or len(df) < 2:
        return 0.0
    
    returns = df['Close'].pct_change().dropna()
    if returns.empty: return 0.0
    
    avg_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    
    if volatility < 1e-6:
        return 0.0
        
    sharpe = (avg_return - risk_free_rate) / volatility
    return round(float(sharpe), 2)

def get_risk_assessment_metrics(df: pd.DataFrame):
    """Returns a dictionary of professional risk metrics."""
    return {
        "max_drawdown": calculate_max_drawdown(df),
        "sharpe_ratio": calculate_sharpe_ratio(df),
        "volatility": df['Close'].pct_change().std() * np.sqrt(252)
    }
