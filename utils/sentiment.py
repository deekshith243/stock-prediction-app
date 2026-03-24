import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def get_sentiment(ticker: str):
    """
    Fetches recent news for a ticker and calculates sentiment score.
    
    Returns:
        dict: Sentiment results (score, label, snippet).
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {"score": 0.0, "label": "Neutral", "snippet": "No recent news found."}
        
        sid = SentimentIntensityAnalyzer()
        scores = []
        snippets = []
        
        for item in news[:5]: # Take top 5 news items
            title = item.get('title', '')
            score = sid.polarity_scores(title)['compound']
            scores.append(score)
            snippets.append(title)
            
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.05:
            label = "Positive"
        elif avg_score < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return {
            "score": round(avg_score, 2),
            "label": label,
            "snippet": snippets[0] if snippets else "N/A"
        }
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"score": 0.0, "label": "Neutral", "snippet": "Error fetching news."}
