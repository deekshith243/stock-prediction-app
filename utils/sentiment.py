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
            return {"score": 0.0, "label": "Neutral", "snippet": "No recent news found.", "headlines": []}
            
        processed_news = []
        scores = []
        for item in news[:5]:
            title = item.get('title', '')
            link = item.get('link', '#')
            publisher = item.get('publisher', 'Unknown')
            
            # Sentiment check
            score = 0
            words = title.lower().split()
            for word in words:
                if word in positive_words: score += 1
                if word in negative_words: score -= 1
            scores.append(score)
            
            processed_news.append({
                "title": title,
                "link": link,
                "publisher": publisher,
                "sentiment": "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            })
            
        avg_score = sum(scores) / len(scores) if scores else 0
        label = "Positive" if avg_score > 0.1 else "Negative" if avg_score < -0.1 else "Neutral"
        
        return {
            "score": round(avg_score, 2),
            "label": label,
            "snippet": processed_news[0]['title'] if processed_news else "N/A",
            "headlines": processed_news
        }
    except Exception as e:
        return {"score": 0.0, "label": "Neutral", "snippet": "Sentiment Service Unavailable", "headlines": []}
