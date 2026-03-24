import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import utilities
from utils.data_loader import fetch_stock_data
from utils.preprocessing import handle_missing_values, scale_features, create_sequences, split_data
from utils.visualizer import plot_candlestick, plot_moving_averages
from utils.indicators import add_indicators
from utils.sentiment import get_sentiment
from utils.evaluator import compare_models
from utils.recommender import get_recommendation
from utils.portfolio import calculate_portfolio_performance, plot_allocation_chart
from utils.alerts import show_alert_ui

# Import models
from models.rf_model import train_rf_model, forecast_future_rf
from models.linear_regression import train_linear_regression, forecast_future_lr
from models.arima_model import train_arima

# --- Page Config ---
st.set_page_config(page_title="GrowthFlow AI | Full-Stack Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styles ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4451; }
    .sidebar .sidebar-content { background-color: #1e2130; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white; }
    h1, h2, h3 { color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# --- Navigation ---
st.sidebar.markdown("# 📈 GrowthFlow AI")
st.sidebar.markdown("*Institutional-Grade Predictive Analytics*")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Navigate", ["📊 Dashboard Home", "📈 Market Analysis", "🔮 AI Predictions", "💼 Portfolio Tracker"])

# --- Helper Functions (with Caching) ---
@st.cache_data
def get_cached_data(ticker, start, end):
    try:
        df = fetch_stock_data(ticker, start, end)
        if not df.empty:
            return handle_missing_values(df)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_ml_results(df, seq_length):
    try:
        data_scaled, scaler = scale_features(df[['Close']].values)
        X, y = create_sequences(data_scaled, seq_length)
        X_train, y_train, X_test, y_test = split_data(X, y)
        
        # Train RandomForest (Lightweight Alternative to LSTM)
        rf_preds, rf_model = train_rf_model(X_train, y_train, X_test)
        rf_forecast = forecast_future_rf(rf_model, X_test[-1], days=7)
        
        # Train LR
        lr_preds, lr_model = train_linear_regression(X_train, y_train, X_test)
        lr_forecast = forecast_future_lr(lr_model, X_test[-1], days=7)
        
        # ARIMA (Statsmodels)
        arima_forecast, _ = train_arima(df['Close'].values, forecast_days=7)
        
        return {
            "y_test": y_test,
            "rf_preds": rf_preds, "rf_forecast": rf_forecast,
            "lr_preds": lr_preds, "lr_forecast": lr_forecast,
            "arima_forecast": arima_forecast,
            "scaler": scaler,
            "success": True
        }
    except Exception as e:
        st.error(f"Error in ML Processing: {e}")
        return {"success": False}

# --- PAGE: Dashboard Home ---
if page == "📊 Dashboard Home":
    st.title("GrowthFlow AI Dashboard")
    st.write("Welcome to the next generation of stock intelligence. Use the sidebar to explore market trends, AI forecasts, and manage your portfolio.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Market Activity")
        ticker = st.text_input("Quick Look (Ticker)", value="AAPL").upper()
        df = get_cached_data(ticker, str(datetime.now() - timedelta(days=30)), str(datetime.now()))
        if not df.empty:
            st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)
    
    with col2:
        st.subheader("AI Performance Summary")
        st.info("Current average model accuracy across top 10 tickers: **94.2%** (Simulated)")
        st.write("Our Lightweight RandomForest and ARIMA models provide high-speed, reliable signals for 3.14+ environments.")

# --- PAGE: Market Analysis ---
elif page == "📈 Market Analysis":
    st.title("Technical Analysis & Sentiment")
    ticker = st.sidebar.text_input("Analysis Ticker", value="TSLA").upper()
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    
    df = get_cached_data(ticker, str(start_date), str(datetime.now().date()))
    if not df.empty:
        df_indicators = add_indicators(df)
        sentiment = get_sentiment(ticker)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", f"{df['Close'].diff().iloc[-1]:.2f}")
        c2.metric("Market Sentiment", sentiment['label'], f"Score: {sentiment['score']}")
        c3.metric("RSI Signal", f"{df_indicators['RSI'].iloc[-1]:.1f}", "Neutral" if 30 < df_indicators['RSI'].iloc[-1] < 70 else "Alert")
        
        st.plotly_chart(plot_candlestick(df_indicators, ticker), use_container_width=True)
        st.plotly_chart(plot_moving_averages(df_indicators, ticker), use_container_width=True)
    else:
        st.error("Please enter a valid ticker.")

# --- PAGE: AI Predictions ---
elif page == "🔮 AI Predictions":
    st.title("ML Forecasting Hub")
    ticker = st.sidebar.text_input("Forecast Ticker", value="AAPL").upper()
    seq_length = st.sidebar.slider("Sequence Length", 30, 100, 60)
    
    df = get_cached_data(ticker, str(datetime.now() - timedelta(days=730)), str(datetime.now().date()))
    if not df.empty:
        results = get_ml_results(df, seq_length)
        if not results.get('success', True):
             st.stop()
             
        scaler = results['scaler']
        
        def inv(p): return scaler.inverse_transform(p.reshape(-1, 1)).flatten()
        
        st.subheader("7-Day Forecast Comparison")
        days_range = [f"Day{i+1}" for i in range(7)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days_range, y=inv(results['rf_forecast']), name="RandomForest (ML)", line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=days_range, y=results['arima_forecast'], name="ARIMA (Statistical)", line=dict(color='magenta')))
        fig.add_trace(go.Scatter(x=days_range, y=inv(results['lr_forecast']), name="Linear Regression", line=dict(color='white')))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Evaluation")
        metrics_df = compare_models({
            "RandomForest": (results['y_test'], results['rf_preds']),
            "Linear Regression": (results['y_test'], results['lr_preds'])
        })
        st.table(metrics_df)
    else:
        st.warning("Data not available for selected ticker.")

# --- PAGE: Portfolio Tracker ---
elif page == "💼 Portfolio Tracker":
    st.title("My Smart Portfolio")
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {"AAPL": {"qty": 10, "avg_price": 150.0}, "TSLA": {"qty": 5, "avg_price": 200.0}}
    
    # Add new stock
    with st.expander("➕ Add New Holding"):
        new_ticker = st.text_input("Ticker").upper()
        new_qty = st.number_input("Quantity", min_value=0.1)
        new_price = st.number_input("Avg Buy Price", min_value=0.1)
        if st.button("Add to Portfolio"):
            st.session_state.portfolio[new_ticker] = {"qty": new_qty, "avg_price": new_price}
            st.success(f"Added {new_ticker}")
            st.rerun()

    if st.session_state.portfolio:
        df_port, summary = calculate_portfolio_performance(st.session_state.portfolio)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Value", f"${summary['total_current_value']:,}")
        c2.metric("Total Profit/Loss", f"${summary['total_pl']:,}", f"{summary['total_pl_pct']}%")
        c3.metric("Total Invested", f"${summary['total_invested']:,}")
        
        st.markdown("---")
        st.subheader("Allocation & Holdings")
        col_list, col_chart = st.columns(2)
        
        with col_list:
            st.dataframe(df_port, use_container_width=True)
        
        with col_chart:
            st.plotly_chart(plot_allocation_chart(df_port), use_container_width=True)
    else:
        st.info("Your portfolio is empty. Add some stocks to get started.")

st.sidebar.markdown("---")
show_alert_ui()
st.sidebar.markdown("---")
st.sidebar.write("v2.0.0 | GrowthFlow AI")
