import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

def calculate_portfolio_performance(holdings: dict):
    """
    Calculates current value and P/L for a dictionary of holdings.
    """
    data = []
    total_invested = 0
    total_current_value = 0
    
    for ticker, info in holdings.items():
        try:
            stock = yf.Ticker(ticker)
            # Try fast_info first, then fallback to history
            try:
                current_price = stock.fast_info['last_price']
            except:
                hist = stock.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            if current_price is None or pd.isna(current_price):
                print(f"Skipping {ticker}: Price unavailable")
                continue
                
            qty = info['qty']
            avg_price = info['avg_price']
            
            invested = qty * avg_price
            current_value = qty * current_price
            pl = current_value - invested
            pl_pct = (pl / invested) * 100 if invested != 0 else 0
            
            data.append({
                "Ticker": ticker,
                "Qty": qty,
                "Avg Price": round(float(avg_price), 2),
                "Current Price": round(float(current_price), 2),
                "Invested": round(float(invested), 2),
                "Current Value": round(float(current_value), 2),
                "P/L": round(float(pl), 2),
                "P/L %": round(float(pl_pct), 2)
            })
            
            total_invested += invested
            total_current_value += current_value
        except Exception as e:
            print(f"Error updating {ticker}: {e}")
            
    df = pd.DataFrame(data)
    summary = {
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current_value, 2),
        "total_pl": round(total_current_value - total_invested, 2),
        "total_pl_pct": round(((total_current_value - total_invested) / total_invested * 100), 2) if total_invested != 0 else 0
    }
    
    return df, summary

def plot_allocation_chart(df: pd.DataFrame):
    """Generates a pie chart of portfolio allocation with robust error handling."""
    try:
        if df.empty or 'Current Value' not in df.columns or df['Current Value'].sum() <= 0:
            # Return a blank figure with a message
            fig = go.Figure()
            fig.add_annotation(text="No allocation data available", showarrow=False, font_size=20)
            fig.update_layout(template="plotly_dark", xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig
            
        fig = px.pie(df, values='Current Value', names='Ticker', title='Portfolio Allocation', hole=0.4)
        fig.update_layout(template="plotly_dark")
        return fig
    except Exception as e:
        print(f"Plotly error: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Error generating chart", showarrow=False, font_size=20)
        fig.update_layout(template="plotly_dark", xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
