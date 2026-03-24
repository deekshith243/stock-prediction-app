import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

def plot_candlestick(df: pd.DataFrame, ticker: str):
    """
    Generates an interactive candlestick chart using Plotly.
    
    Args:
        df (pd.DataFrame): Dataframe with Open, High, Low, Close.
        ticker (str): The stock ticker symbol.
    """
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker)])
    
    fig.update_layout(title=f"{ticker} Candlestick Chart",
                      yaxis_title="Price",
                      xaxis_title="Date",
                      template="plotly_dark")
    return fig

def plot_moving_averages(df: pd.DataFrame, ticker: str):
    """
    Plots the closing price along with moving averages (50, 200).
    
    Args:
        df (pd.DataFrame): Dataframe with 'Close' column.
        ticker (str): The stock ticker symbol.
    """
    df_copy = df.copy()
    df_copy['MA50'] = df_copy['Close'].rolling(window=50).mean()
    df_copy['MA200'] = df_copy['Close'].rolling(window=200).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['Close'], name="Close", line=dict(color='white', width=1.5)))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['MA50'], name="50-day MA", line=dict(color='yellow', width=1.5)))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['MA200'], name="200-day MA", line=dict(color='blue', width=1.5)))
    
    fig.update_layout(title=f"{ticker} Closing Price & MAs",
                      yaxis_title="Price",
                      xaxis_title="Date",
                      template="plotly_dark")
    return fig
