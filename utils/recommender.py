import pandas as pd

def get_recommendation(last_price, next_day_pred, rsi, sentiment, macd=0, macd_signal=0, bb_upper=0, bb_lower=0):
    """
    Combined logic for Buy/Sell/Hold recommendation with advanced signals.
    """
    score = 0
    signals = []

    # 1. Prediction Signal
    price_change = (next_day_pred - last_price) / last_price
    if price_change > 0.02:
        score += 2
        signals.append("Strong Bullish Forecast (>2%)")
    elif price_change > 0.005:
        score += 1
        signals.append("Bullish Forecast")
    elif price_change < -0.02:
        score -= 2
        signals.append("Strong Bearish Forecast (<-2%)")
    elif price_change < -0.005:
        score -= 1
        signals.append("Bearish Forecast")

    # 2. RSI Signal
    if rsi < 30:
        score += 2
        signals.append("Oversold (RSI < 30)")
    elif rsi < 45:
        score += 1
        signals.append("Approaching Oversold")
    elif rsi > 70:
        score -= 2
        signals.append("Overbought (RSI > 70)")
    elif rsi > 55:
        score -= 1
        signals.append("Approaching Overbought")

    # 3. MACD Crossover Signal
    if macd > macd_signal:
        score += 1
        signals.append("MACD Bullish Cross")
    else:
        score -= 1
        signals.append("MACD Bearish Cross")

    # 4. Bollinger Band Signal
    if last_price > bb_upper:
        score -= 1
        signals.append("BB Upper Breakout (Possible Rejection)")
    elif last_price < bb_lower:
        score += 1
        signals.append("BB Lower Breakout (Possible Rebound)")

    # 5. Sentiment Signal
    if sentiment['label'] == "Positive":
        score += 1
        signals.append("Positive Market Sentiment")
    elif sentiment['label'] == "Negative":
        score -= 1
        signals.append("Negative Market Sentiment")

    # Final logic
    if score >= 4:
        recommendation = "STRONG BUY"
        color = "#00ff00" # Bright Green
        explanation = "High conviction alignment across models and technicals."
    elif score >= 2:
        recommendation = "BUY"
        color = "#00ffcc" # Greenish Cyan
        explanation = "Technical indicators suggest an accumulation phase."
    elif score <= -4:
        recommendation = "STRONG SELL"
        color = "#ff0000" # Bright Red
        explanation = "Extreme overbought or bearish breakdown detected."
    elif score <= -2:
        recommendation = "SELL"
        color = "#ff4b4b" # Light Red
        explanation = "Indicators suggest potential downward momentum."
    else:
        recommendation = "HOLD"
        color = "#f1c40f" # Yellow/Gold
        explanation = "Market signals are neutral or consolidating."

    # Entry/Exit Suggestions (Simple heuristics)
    ideal_buy = last_price * 0.985 if bb_lower == 0 else max(bb_lower, last_price * 0.98)
    ideal_sell = last_price * 1.05 if bb_upper == 0 else min(bb_upper, last_price * 1.06)

    return {
        "action": recommendation,
        "color": color,
        "score": score,
        "signals": signals,
        "explanation": explanation,
        "entry": round(ideal_buy, 2),
        "exit": round(ideal_sell, 2)
    }
