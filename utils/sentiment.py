import yfinance as yf

def get_sentiment(ticker: str):
    """
    Fetches recent news and performs simple keyword-based sentiment analysis.
    """
    positive_words = {'growth', 'bull', 'gain', 'rise', 'higher', 'buy', 'upgrade', 'beat', 'success', 'jump'}
    negative_words = {'fall', 'bear', 'loss', 'drop', 'lower', 'sell', 'downgrade', 'miss', 'risk', 'plunge'}
    
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {"score": 0.0, "label": "Neutral", "snippet": "No recent news found."}
            
        scores = []
        for item in news[:5]:
            title = item.get('title', '').lower()
            score = 0
            words = title.split()
            for word in words:
                if word in positive_words: score += 1
                if word in negative_words: score -= 1
            scores.append(score)
            
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0: label = "Positive"
        elif avg_score < 0: label = "Negative"
        else: label = "Neutral"
        
        return {
            "score": round(avg_score, 2),
            "label": label,
            "snippet": news[0].get('title', 'N/A')
        }
    except Exception as e:
        return {"score": 0.0, "label": "Neutral", "snippet": "Sentiment Service Unavailable"}
