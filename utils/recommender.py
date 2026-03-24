import pandas as pd

def get_recommendation(last_price, next_day_pred, rsi, sentiment):
    """
    Combined logic for Buy/Sell/Hold recommendation.
    """
    score = 0
    signals = []

    # 1. Prediction Signal
    price_change = (next_day_pred - last_price) / last_price
    if price_change > 0.01:
        score += 1
        signals.append("Bullish Forecast (+1%)")
    elif price_change < -0.01:
        score -= 1
        signals.append("Bearish Forecast (-1%)")

    # 2. Indicator Signal (RSI)
    if rsi < 30:
        score += 1
        signals.append("Oversold (RSI < 30)")
    elif rsi > 70:
        score -= 1
        signals.append("Overbought (RSI > 70)")

    # 3. Sentiment Signal
    if sentiment['label'] == "Positive":
        score += 1
        signals.append("Positive Sentiment")
    elif sentiment['label'] == "Negative":
        score -= 1
        signals.append("Negative Sentiment")

    # Final logic
    if score >= 1:
        recommendation = "BUY"
        color = "green"
    elif score <= -1:
        recommendation = "SELL"
        color = "red"
    else:
        recommendation = "HOLD"
        color = "gray"

    return {
        "action": recommendation,
        "color": color,
        "score": score,
        "signals": signals,
        "explanation": f"Recommendation based on {', '.join(signals) if signals else 'Neutral Market'}."
    }
