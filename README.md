# 📈 GrowthFlow AI: Advanced Stock Forecasting Platform

GrowthFlow AI is a high-performance, modular stock prediction and portfolio management dashboard built with Python, Streamlit, and State-of-the-Art Machine Learning models.

## 🚀 Key Features

### 🤖 Multi-Model AI Predictions
- **Deep LSTM**: Recurrent Neural Networks for complex pattern recognition.
- **Facebook Prophet**: Robust additive modeling for trend and seasonality.
- **ARIMA**: Statistical time-series forecasting with auto-parameter selection.
- **Linear Regression**: High-speed trend baseline.

### 📊 Technical Intelligence
- **Indicators**: Real-time RSI, MACD, and Bollinger Bands.
- **Sentiment Analysis**: NLP-powered news sentiment scoring using NLTK VADER.
- **Strategy Signal**: Unified BUY/SELL/HOLD recommendations based on combined AI & Technical data.

### 💼 Smart Portfolio Tracker
- Real-time Profit/Loss tracking.
- Interactive Asset Allocation visualization.
- Dynamic valuation based on live market prices.

## 🛠️ Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/deekshith243/stock-prediction-app.git
   cd stock-prediction-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🌐 Live Demo
[View Live on Streamlit Cloud](https://share.streamlit.io/) <!-- Replace with actual link after deployment -->

## 📁 Project Structure
- `app.py`: Main dashboard and multi-page navigation.
- `models/`: ML model implementations (LSTM, Prophet, ARIMA, LR).
- `utils/`: Core logic for indicators, sentiment, portfolio, and data loading.
- `data/`: Local cache for historical CSV files.

## 🛡️ Production Optimized
- **CPU Efficient**: Force-CPU mode for TensorFlow to ensure stability on shared cloud instances.
- **Fast Loading**: Integrated Streamlit caching (`@st.cache_data`, `@st.cache_resource`).
- **Robust**: Comprehensive error handling for tickers and API failures.

---
*Built with ❤️ by [Deekshith](https://github.com/deekshith243)*
