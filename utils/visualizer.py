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

def plot_forecast_with_confidence(days_range, forecast, lower_bound, upper_bound, historical_x=None, historical_y=None, model_name="Ensemble AI"):
    """
    Plots forecast with optionally including historical context for alignment.
    """
    fig = go.Figure()
    
    # 1. Historical Data (if provided)
    if historical_x is not None and historical_y is not None:
        fig.add_trace(go.Scatter(
            x=historical_x, y=historical_y,
            name="Actual Close",
            line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
        ))
        # Bridge the last historical point to the first forecast point
        bridge_x = [historical_x[-1], days_range[0]]
        bridge_y = [historical_y[-1], forecast[0]]
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            showlegend=False,
            line=dict(color='cyan', width=4, dash='dot'),
            hoverinfo="skip"
        ))

    # 2. Shaded Confidence Interval
    fig.add_trace(go.Scatter(
        x=days_range + days_range[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(0, 255, 255, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name="95% Confidence Band"
    ))
    
    # 3. Forecast Line
    fig.add_trace(go.Scatter(
        x=days_range, y=forecast,
        name=f"{model_name} Forecast",
        line=dict(color='cyan', width=4)
    ))
    
    fig.update_layout(title=f"{model_name} Predictive Trajectory",
                      template="plotly_dark", height=450,
                      xaxis_title="Horizon", yaxis_title="Price ($)",
                      hovermode="x unified")
    return fig

def plot_moving_averages(df: pd.DataFrame, ticker: str):
    """
    Plots the closing price along with moving averages (50, 200).
    """
    df_copy = df.copy()
    if 'MA50' not in df_copy.columns:
        df_copy['MA50'] = df_copy['Close'].rolling(window=50).mean()
    if 'MA200' not in df_copy.columns:
        df_copy['MA200'] = df_copy['Close'].rolling(window=200).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['Close'], name="Close", line=dict(color='rgba(255, 255, 255, 0.8)', width=2)))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['MA50'], name="50-day MA", line=dict(color='#f1c40f', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['MA200'], name="200-day MA", line=dict(color='#3498db', width=1.5, dash='dot')))
    
    fig.update_layout(title=f"{ticker} Institutional Trend Analysis (MAs)",
                      yaxis_title="Price ($)",
                      xaxis_title="Date",
                      template="plotly_dark",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
