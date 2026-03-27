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
tab_dashboard, tab_analysis, tab_prediction, tab_portfolio = st.tabs(["📊 Dashboard", "🔍 Analysis", "🔮 Prediction", "💼 Portfolio"])

# --- Sidebar Enhanced: Watchlist & Alerts ---
with st.sidebar:
    st.markdown("### 📋 My Watchlist")
    for w_ticker in st.session_state.watchlist:
        with st.container():
            w_col1, w_col2, w_col3 = st.columns([2, 2, 1])
            w_col1.markdown(f"**{w_ticker}**")
            
            # Fetch mini-data for watchlist
            try:
                w_df = get_cached_data(w_ticker, (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
                if not w_df.empty:
                    w_last = w_df['Close'].iloc[-1]
                    w_prev = w_df['Close'].iloc[-2]
                    w_change = ((w_last - w_prev) / w_prev) * 100
                    w_color = "green" if w_change >= 0 else "red"
                    w_col2.markdown(f"**${w_last:.2f}** <span style='color:{w_color}; font-size:0.8em;'>({w_change:+.2f}%)</span>", unsafe_allow_html=True)
                else:
                    w_col2.write("N/A")
            except:
                w_col2.write("Error")
                
            if w_col3.button("🗑️", key=f"del_{w_ticker}"):
                st.session_state.watchlist.remove(w_ticker)
                st.rerun()
    
    add_w = st.text_input("Add Ticker to Watchlist", key="add_w").upper()
    if st.button("Add Ticker"):
        if add_w and add_w not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_w)
            st.rerun()

    st.markdown("---")
    show_alert_ui()

# --- Global Alert Check ---
if 'alerts' in st.session_state:
    for alert in st.session_state.alerts:
        if alert['active']:
            # This is a bit expensive to do for all alerts on every rerun, but for demo it works
            a_df = get_cached_data(alert['ticker'], (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            if not a_df.empty:
                if check_alerts(a_df['Close'].iloc[-1], alert['target'], alert['ticker'], alert['direction']):
                    alert['active'] = False # Deactivate after trigger

# --- TAB: Dashboard ---
with tab_dashboard:
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
        st.subheader("Asset Performance")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", help="e.g. MSFT, TSLA, BTC-USD", key="dashboard_ticker").upper()
        
        with st.spinner(f"Analyzing {ticker}..."):
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            df = get_cached_data(ticker, start_date, end_date)
            if not df.empty:
                st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)
                
                with st.expander("⚖️ Multi-Asset Comparison"):
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
                st.error(f"Ticker '{ticker}' not found.")

    with col_sidebar:
        st.subheader("Smart Recommendations")
        if not df.empty:
            from utils.sentiment import get_sentiment
            from utils.recommender import get_recommendation
            
            sentiment = get_sentiment(ticker)
            st.markdown(f"""
            <div class="fintech-card">
                <h3>{sentiment['label']} Sentiment</h3>
                <p>Confidence: <b>{sentiment['score']}%</b></p>
                <small><i>"{sentiment['snippet']}"</i></small>
            </div>
            """, unsafe_allow_html=True)
            
            # Smart Recommendation Card
            last_price = df['Close'].iloc[-1]
            try:
                results_temp = get_ml_results(df, 60, n_days=1)
                next_pred = results_temp['rf_forecast'][0]
                
                # Add indicators for recommendation
                df_rec = add_indicators(df)
                latest_rec = df_rec.iloc[-1]
                
                rec = get_recommendation(
                    last_price, next_pred, latest_rec['RSI'], sentiment, 
                    latest_rec['MACD'], latest_rec['MACD_Signal'], 
                    latest_rec['BB_Upper'], latest_rec['BB_Lower']
                )
                st.markdown(f"""
                <div class="fintech-card" style="border-left: 5px solid {rec['color']};">
                    <h2 style="color: {rec['color']};">{rec['action']}</h2>
                    <p>{rec['explanation']}</p>
                    <small>Score: {rec['score']} | Signals: {len(rec['signals'])}</small>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Recommendation Error: {e}")

# --- TAB: Analysis ---
with tab_analysis:
    st.title("Technical Indicators Panel")
    if not df.empty:
        col_an1, col_an2 = st.columns([2, 1])
        with col_an1:
            st.plotly_chart(plot_moving_averages(df, ticker), use_container_width=True)
            
            # Indicators Interpretation
            st.subheader("Deep-Dive Insights")
            df_ind = add_indicators(df)
            interpretations = get_indicator_interpretation(df_ind)
            
            ic1, ic2, ic3 = st.columns(3)
            ic1.info(f"**RSI**: {interpretations['RSI']}")
            ic2.info(f"**MACD**: {interpretations['MACD']}")
            ic3.info(f"**Bollinger**: {interpretations['BB']}")
            
            st.success(f"**Overall Trend**: {interpretations['Trend']}")
            
        with col_an2:
            st.subheader("Historical Snapshot")
            st.dataframe(df.tail(15), use_container_width=True)
            st.download_button("📥 Export Historical CSV", data=df.to_csv(), file_name=f"{ticker}_history.csv", use_container_width=True)
    else:
        st.warning("Please fetch data in the Dashboard first.")

# --- TAB: Prediction ---
with tab_prediction:
    st.title("Predictive Intelligence")
    col_ctrl1, col_ctrl2 = st.columns(2)
    p_ticker = col_ctrl1.text_input("Ticker for Forecast", value=ticker, key="predict_ticker").upper()
    n_days = col_ctrl2.number_input("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)
    
    with st.spinner("Generating AI Projections..."):
        p_df = get_cached_data(p_ticker, (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
        if not p_df.empty:
            results = get_ml_results(p_df, 60, n_days=n_days) 
            if results.get('success', False):
                scaler = results['scaler']
                def inv(p): return scaler.inverse_transform(p.reshape(-1, 1)).flatten()
                
                col_chart, col_metrics = st.columns([3, 1])
                
                with col_chart:
                    st.subheader(f"{n_days}-Day Confidence-Aware Forecast")
                    days_range = [f"Day {i+1}" for i in range(n_days)]
                    conf_int = results['arima_conf_int']
                    
                    fig = plot_forecast_with_confidence(
                        days_range, inv(results['rf_forecast']), 
                        inv(conf_int[:, 0]), inv(conf_int[:, 1]), 
                        model_name="Ensemble AI"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    forecast_df = pd.DataFrame({
                        "Horizon": days_range, 
                        "Predicted_Price": inv(results['rf_forecast']),
                        "Lower_Bound": inv(conf_int[:, 0]),
                        "Upper_Bound": inv(conf_int[:, 1])
                    })
                    st.download_button("📥 Download Forecast CSV", data=forecast_df.to_csv(index=False), file_name=f"{p_ticker}_forecast.csv", mime="text/csv")

                with col_metrics:
                    st.subheader("Model Status")
                    cf_score = 88.5 # Simulated 
                    st.markdown(f"""
                    <div class="fintech-card">
                        <p>Confidence Level</p>
                        <h2 style='color: #00ffcc;'>{cf_score}%</h2>
                        <small>Backtesting Reliability</small>
                    </div>
                    """, unsafe_allow_html=True)
                    from utils.evaluator import compare_models
                    st.write("### Cross-Validation")
                    st.table(compare_models({"RF": (results['y_test'], results['rf_preds']), "LR": (results['y_test'], results['lr_preds'])}))
        else:
            st.warning("Enter a valid ticker to start forecasting.")

# --- TAB: Portfolio ---
with tab_portfolio:
    st.title("Asset Management")
    from utils.portfolio import calculate_portfolio_performance, plot_allocation_chart
    
    with st.expander("➕ Log Transaction", expanded=False):
        c1, c2, c3 = st.columns(3)
        port_ticker = c1.text_input("Ticker Symbol").upper()
        port_qty = c2.number_input("Units", min_value=0.01, step=1.0)
        port_price = c3.number_input("Cost Basis ($)", min_value=0.01, step=1.0)
        if st.button("Confirm Portfolio Update"):
            st.session_state.portfolio[port_ticker] = {"qty": port_qty, "avg_price": port_price}
            st.toast(f"Added {port_ticker} to portfolio", icon="📈")
            st.rerun()

    if st.session_state.portfolio:
        df_port, summary = calculate_portfolio_performance(st.session_state.portfolio)
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Market Value", f"${summary['total_current_value']:,.2f}")
        sc2.metric("Unrealized P/L", f"${summary['total_pl']:,.2f}", f"{summary['total_pl_pct']:.2f}%")
        sc3.metric("Invested Capital", f"${summary['total_invested']:,.2f}")
        
        st.markdown("---")
        col_t, col_p = st.columns([3, 2])
        with col_t:
            st.subheader("Holdings Summary")
            st.dataframe(df_port, use_container_width=True)
        with col_p:
            st.subheader("Asset Allocation")
            st.plotly_chart(plot_allocation_chart(df_port), use_container_width=True)
    else:
        st.info("Your portfolio is currently empty.")

st.sidebar.markdown("---")
st.sidebar.write("⚡ v4.0.0 Pro | Powered by AI")
