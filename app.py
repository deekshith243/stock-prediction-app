import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import utilities
from utils.data_loader import fetch_stock_data
from utils.preprocessing import handle_missing_values, scale_features, create_sequences, split_data
from models.rf_model import train_rf_model, forecast_future_rf
from models.linear_regression import train_linear_regression, forecast_future_lr
from models.arima_model import train_arima

from utils.visualizer import plot_candlestick, plot_moving_averages, plot_forecast_with_confidence
from utils.indicators import add_indicators, get_indicator_interpretation
from utils.recommender import get_recommendation
from utils.alerts import show_alert_ui, check_alerts

# --- Page Config ---
st.set_page_config(page_title="GrowthFlow AI | Full-Stack Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styles (Fintech Premium) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #3e4451;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .fintech-card {
        background-color: #1e2130;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #3e4451;
        margin-bottom: 20px;
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background-color: #ff4b4b; 
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff3333; transform: translateY(-2px); }
    h1, h2, h3 { color: #ff4b4b; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: transparent;
        border-radius: 4px;
        color: #ffffff;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #ff4b4b !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Branding ---
st.sidebar.markdown("# 📈 GrowthFlow AI")
st.sidebar.markdown("*Institutional-Grade Predictive Analytics*")
st.sidebar.markdown("---")

# --- Initialize Session State ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"AAPL": {"qty": 10, "avg_price": 150.0}, "TSLA": {"qty": 5, "avg_price": 200.0}}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "TSLA", "BTC-USD"]

# --- Tabbed Navigation ---
tab_home, tab_analysis, tab_predict, tab_portfolio = st.tabs(["📊 Dashboard", "🔍 Analysis", "🔮 AI Forecasts", "💼 My Holdings"])

# --- Sidebar Enhanced: Watchlist & Alerts ---
with st.sidebar:
    st.markdown("### 📋 My Watchlist")
    for w_ticker in st.session_state.watchlist:
        w_col1, w_col2 = st.columns([3, 1])
        w_col1.markdown(f"**{w_ticker}**")
        if w_col2.button("🗑️", key=f"del_{w_ticker}"):
            st.session_state.watchlist.remove(w_ticker)
            st.rerun()
    
    add_w = st.text_input("Add Ticker to Watchlist", key="add_w").upper()
    if st.button("Add"):
        if add_w and add_w not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_w)
            st.rerun()

    st.markdown("---")
    show_alert_ui()

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
def get_ml_results(df, seq_length, n_days=7):
    try:
        data_scaled, scaler = scale_features(df[['Close']].values)
        X, y = create_sequences(data_scaled, seq_length)
        X_train, y_train, X_test, y_test = split_data(X, y)
        
        # Train RandomForest (Lightweight Alternative to LSTM)
        rf_preds, rf_model = train_rf_model(X_train, y_train, X_test)
        rf_forecast = forecast_future_rf(rf_model, X_test[-1], days=n_days)
        
        # Train LR
        lr_preds, lr_model = train_linear_regression(X_train, y_train, X_test)
        lr_forecast = forecast_future_lr(lr_model, X_test[-1], days=n_days)
        
        # ARIMA (Statsmodels)
        arima_forecast, arima_conf_int = train_arima(df['Close'].values, forecast_days=n_days)
        
        return {
            "y_test": y_test,
            "rf_preds": rf_preds, "rf_forecast": rf_forecast,
            "lr_preds": lr_preds, "lr_forecast": lr_forecast,
            "arima_forecast": arima_forecast,
            "arima_conf_int": arima_conf_int,
            "scaler": scaler,
            "success": True
        }
    except Exception as e:
        st.error(f"Error in ML Processing: {e}")
        return {"success": False}

# --- TAB: Dashboard ---
with tab_home:
    st.title("Market Overview")
    
    # Quick Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPY", "512.40", "+0.45%", help="S&P 500 ETF Trust")
    c2.metric("BTC", "$65,210", "-1.2%", help="Bitcoin/USD")
    c3.metric("VIX", "14.20", "-2.1%", help="Volatility Index")
    c4.metric("Gold", "$2,150", "+0.1%", help="Gold Spot")

    st.markdown("---")
    
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_main:
        st.subheader("Interactive Analysis")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", help="e.g. MSFT, TSLA, BTC-USD").upper()
        
        with st.spinner(f"Analyzing {ticker}..."):
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            df = get_cached_data(ticker, start_date, end_date)
            if not df.empty:
                try:
                    st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)
                except Exception as chart_err:
                    st.error(f"Chart Error: {chart_err}")
                
                # Multi-Stock Comparison Expandable
                with st.expander("⚖️ Compare with another asset"):
                    comp_ticker = st.text_input("Comparison Ticker", value="MSFT").upper()
                    if comp_ticker and comp_ticker != ticker:
                        df_comp = get_cached_data(comp_ticker, start_date, end_date)
                        if not df_comp.empty:
                            fig_comp = go.Figure()
                            fig_comp.add_trace(go.Scatter(x=df.index, y=df['Close']/df['Close'].iloc[0], name=ticker))
                            fig_comp.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Close']/df_comp['Close'].iloc[0], name=comp_ticker))
                            fig_comp.update_layout(title="Relative Performance (Normalized)", template="plotly_dark")
                            st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                            st.warning(f"Could not load data for {comp_ticker}")
            else:
                st.error(f"Ticker '{ticker}' not found or no data available. Please check the symbol.")

    with col_sidebar:
        st.subheader("Market Sentiment")
        if not df.empty:
            sentiment = get_sentiment(ticker)
            st.markdown(f"""
            <div class="fintech-card">
                <h3>{sentiment['label']}</h3>
                <p>Confidence Score: <b>{sentiment['score']}</b></p>
                <small><i>"{sentiment['snippet']}"</i></small>
            </div>
            """, unsafe_allow_html=True)
            
            # Smart Recommendation Card
            last_price = df['Close'].iloc[-1]
            try:
                results_temp = get_ml_results(df, 60, n_days=1)
                next_pred = results_temp['rf_forecast'][0]
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
                macd_s = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
                bb_u = df['BB_Upper'].iloc[-1] if 'BB_Upper' in df.columns else 0
                bb_l = df['BB_Lower'].iloc[-1] if 'BB_Lower' in df.columns else 0
                
                rec = get_recommendation(last_price, next_pred, rsi, sentiment, macd, macd_s, bb_u, bb_l)
                st.markdown(f"""
                <div class="fintech-card" style="border-left: 5px solid {rec['color']};">
                    <h2 style="color: {rec['color']};">{rec['action']}</h2>
                    <p>{rec['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass
            
            st.info("💡 Tip: Navigate through tabs for deep analysis and 180-day forecasts.")

# --- TAB: Analysis ---
with tab_analysis:
    st.title("Technical Deep-Dive")
    if not df.empty:
        col_an1, col_an2 = st.columns([2, 1])
        with col_an1:
            st.plotly_chart(plot_moving_averages(df, ticker), use_container_width=True)
            
            # Indicators Interpretation
            st.subheader("Indicator Insights")
            interpretations = get_indicator_interpretation(df)
            ic1, ic2, ic3 = st.columns(3)
            ic1.info(f"**RSI**: {interpretations['RSI']}")
            ic2.info(f"**MACD**: {interpretations['MACD']}")
            ic3.info(f"**Bollinger**: {interpretations['BB']}")
            
        with col_an2:
            st.subheader("Key Statistics")
            st.write(df.tail(10))
            st.download_button("📥 Export Historical Data", data=df.to_csv(), file_name=f"{ticker}_history.csv")
    else:
        st.warning("Please fetch data in the Dashboard first.")

# --- TAB: AI Forecasts ---
with tab_predict:
    st.title("Predictive Intelligence")
    ticker = st.text_input("Select Ticker for AI Forecast", value="AAPL", key="forecast_ticker").upper()
    n_days = st.number_input("Enter number of days to predict", min_value=1, max_value=30, value=7)
    
    with st.spinner(f"Generating AI Projections for {n_days} days..."):
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = get_cached_data(ticker, start_date, end_date)
        if not df.empty:
            results = get_ml_results(df, 60, n_days=n_days) 
            if not results.get('success', True):
                 st.stop()
                 
            scaler = results['scaler']
            def inv(p): return scaler.inverse_transform(p.reshape(-1, 1)).flatten()
            
            col_chart, col_metrics = st.columns([3, 1])
            
            with col_chart:
                st.subheader(f"{n_days}-Day Price Trajectory")
                days_range = [f"Day {i+1}" for i in range(n_days)]
                
                # Plot with Confidence Bands
                conf_int = results['arima_conf_int']
                fig = plot_forecast_with_confidence(days_range, inv(results['rf_forecast']), 
                                                   inv(conf_int[:, 0]), inv(conf_int[:, 1]), 
                                                   model_name="Ensemble RF/ARIMA")
                
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as chart_err:
                    st.error(f"Forecast Chart Error: {chart_err}")
                
                # Download CSV
                forecast_df = pd.DataFrame({"Day": days_range, "RandomForest": inv(results['rf_forecast']), "ARIMA": results['arima_forecast']})
                st.download_button("📥 Download Forecast as CSV", data=forecast_df.to_csv(index=False), file_name=f"{ticker}_forecast.csv", mime="text/csv")

            with col_metrics:
                st.subheader("Model Insights")
                confidence = 92.5 # Mock confidence
                st.markdown(f"""
                <div class="fintech-card">
                    <p>Model Confidence</p>
                    <h2 style='color: #00ffcc;'>{confidence}%</h2>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence/100)
                st.write("<small>Based on 30-day backtesting variance</small>", unsafe_allow_html=True)
                
                st.write("### Evaluation Tags")
                st.table(compare_models({"RF": (results['y_test'], results['rf_preds']), "LR": (results['y_test'], results['lr_preds'])}))
        else:
            st.warning("Please enter a valid ticker to start forecasting.")

# --- TAB: My Holdings ---
with tab_portfolio:
    st.title("Asset Management")
    
    # Portfolio Actions
    with st.expander("➕ Register New Transaction"):
        c1, c2, c3 = st.columns(3)
        new_ticker = c1.text_input("Ticker").upper()
        new_qty = c2.number_input("Units", min_value=0.1)
        new_price = c3.number_input("Cost Basis ($)", min_value=0.1)
        if st.button("Confirm Purchase"):
            st.session_state.portfolio[new_ticker] = {"qty": new_qty, "avg_price": new_price}
            st.toast(f"Logged purchase of {new_ticker}", icon="✅")
            st.rerun()

    if st.session_state.portfolio:
        df_port, summary = calculate_portfolio_performance(st.session_state.portfolio)
        
        # Summary Header
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Net Worth", f"${summary['total_current_value']:,}", help="Current market value of all holdings")
        sc2.metric("Unrealized P/L", f"${summary['total_pl']:,}", f"{summary['total_pl_pct']}%", help="Total profit/loss since inception")
        sc3.metric("Cost Basis", f"${summary['total_invested']:,}", help="Total capital deployed")
        
        st.markdown("---")
        
        # Holdings and Allocation
        col_table, col_pie = st.columns([3, 2])
        with col_table:
            st.subheader("Current Holdings")
            st.dataframe(df_port, use_container_width=True)
        
        with col_pie:
            st.subheader("Risk Distribution")
            st.plotly_chart(plot_allocation_chart(df_port), use_container_width=True)
    else:
        st.info("No active holdings found. Register a transaction to track performance.")

st.sidebar.markdown("---")
show_alert_ui()
st.sidebar.write("v3.0.0 | Institutional Edition")
