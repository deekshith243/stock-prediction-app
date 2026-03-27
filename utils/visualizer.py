import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

def plot_candlestick(df: pd.DataFrame, ticker: str):
    """
    Generates an interactive candlestick chart with Volume.
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                       row_width=[0.2, 0.7])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name=ticker), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color='rgba(100, 100, 255, 0.5)'), row=2, col=1)
    
    fig.update_layout(title=f"{ticker} Professional Chart",
                      template="plotly_dark", height=600,
                      xaxis_rangeslider_visible=False)
    return fig

def plot_forecast_with_confidence(days_range, forecast, lower_bound, upper_bound, model_name="ARIMA"):
    """
    Plots forecast with shaded confidence bands.
    """
    fig = go.Figure()
    
    # Shaded Confidence Interval
    fig.add_trace(go.Scatter(
        x=days_range + days_range[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(0, 255, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name=f"95% Confidence Band"
    ))
    
    # Forecast Line
    fig.add_trace(go.Scatter(
        x=days_range, y=forecast,
        name=f"{model_name} Forecast",
        line=dict(color='cyan', width=4)
    ))
    
    fig.update_layout(title=f"{model_name} Predictive Trajectory",
                      template="plotly_dark", height=450,
                      xaxis_title="Future Horizon", yaxis_title="Price")
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
