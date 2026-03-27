import pandas as pd

def analyze_portfolio(portfolio_dict: dict) -> dict:
    """
    Analyzes portfolio concentration and suggests diversification.
    """
    if not portfolio_dict:
        return {"status": "Empty Portfolio", "warnings": [], "suggestions": []}
        
    total_value = sum(item['qty'] * item['avg_price'] for item in portfolio_dict.values())
    analysis = {"total_value": total_value, "holdings": [], "warnings": [], "suggestions": []}
    
    # Simple Sector Map (Mock)
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "TSLA": "Consumer Cyclical",
        "BTC-USD": "Crypto", "ETH-USD": "Crypto", "GOOGL": "Communication Services",
        "AMZN": "Consumer Cyclical", "NVDA": "Technology", "JPM": "Financial Services"
    }
    
    sector_weight = {}
    
    for ticker, data in portfolio_dict.items():
        weight = (data['qty'] * data['avg_price']) / total_value
        sector = sector_map.get(ticker, "Other")
        
        sector_weight[sector] = sector_weight.get(sector, 0) + weight
        analysis['holdings'].append({"ticker": ticker, "weight": weight, "sector": sector})
        
        # Check Concentration
        if weight > 0.40:
            analysis['warnings'].append(f"High Concentration: {ticker} occupies {weight:.1%} of your portfolio.")
            analysis['suggestions'].append(f"Consider reducing exposure to {ticker} to lower single-asset risk.")
            
    # Check Sector Exposure
    for sector, weight in sector_weight.items():
        if weight > 0.60:
            analysis['warnings'].append(f"Sector Overexposure: {sector} accounts for {weight:.1%} of assets.")
            analysis['suggestions'].append(f"Look for opportunities in other sectors to improve diversification.")
            
    if not analysis['warnings']:
        analysis['status'] = "Healthy & Diversified"
    else:
        analysis['status'] = "Action Required"
        
    return analysis

def detect_market_regime(df_market: pd.DataFrame) -> str:
    """
    Classifies the market regime based on simple trend following logic.
    """
    if df_market.empty or 'Close' not in df_market.columns:
        return "Unknown"
        
    price = df_market['Close'].iloc[-1]
    ma50 = df_market['Close'].rolling(window=50).mean().iloc[-1]
    ma200 = df_market['Close'].rolling(window=200).mean().iloc[-1]
    
    if price > ma50 > ma200:
        return "Bullish (Accumulation)"
    elif price < ma50 < ma200:
        return "Bearish (Distribution)"
    else:
        return "Sideways (Consolidation)"
