import pandas as pd
import numpy as np

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates MACD and Signal Line."""
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2):
    """Calculates Bollinger Bands (Middle, Upper, Lower)."""
    middle_band = df['Close'].rolling(window=window).mean()
    std_dev = df['Close'].rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return middle_band, upper_band, lower_band

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all technical indicators to the dataframe."""
    df = df.copy()
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df)
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df)
    return df

def get_indicator_interpretation(df: pd.DataFrame):
    """Provides textual interpretation of the latest indicator values."""
    latest = df.iloc[-1]
    interpretations = {}

    # RSI
    rsi = latest['RSI']
    if rsi < 30: interpretations['RSI'] = "Oversold (Buying Opportunity)"
    elif rsi > 70: interpretations['RSI'] = "Overbought (Selling Pressure)"
    else: interpretations['RSI'] = "Neutral"

    # MACD
    if latest['MACD'] > latest['MACD_Signal']: interpretations['MACD'] = "Bullish Crossover"
    else: interpretations['MACD'] = "Bearish Crossover"

    # Bollinger Bands
    price = latest['Close']
    if price > latest['BB_Upper']: interpretations['BB'] = "Above Upper Band (Overextended)"
    elif price < latest['BB_Lower']: interpretations['BB'] = "Below Lower Band (Undervalued)"
    else: interpretations['BB'] = "Within Range"

    return interpretations
