import pandas as pd
import plotly.express as px
import yfinance as yf

def calculate_portfolio_performance(holdings: dict):
    """
    Calculates current value and P/L for a dictionary of holdings.
    
    Args:
        holdings (dict): { ticker: { "qty": float, "avg_price": float } }
    """
    data = []
    total_invested = 0
    total_current_value = 0
    
    for ticker, info in holdings.items():
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.fast_info['last_price']
            
            qty = info['qty']
            avg_price = info['avg_price']
            
            invested = qty * avg_price
            current_value = qty * current_price
            pl = current_value - invested
            pl_pct = (pl / invested) * 100 if invested != 0 else 0
            
            data.append({
                "Ticker": ticker,
                "Qty": qty,
                "Avg Price": round(avg_price, 2),
                "Current Price": round(current_price, 2),
                "Invested": round(invested, 2),
                "Current Value": round(current_value, 2),
                "P/L": round(pl, 2),
                "P/L %": round(pl_pct, 2)
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
    """Generates a pie chart of portfolio allocation."""
    if df.empty:
        return None
    fig = px.pie(df, values='Current Value', names='Ticker', title='Portfolio Allocation', hole=0.4)
    fig.update_layout(template="plotly_dark")
    return fig
