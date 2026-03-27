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

from utils.risk_metrics import get_risk_assessment_metrics
from utils.backtester import backtest_strategy
from utils.portfolio_advisor import analyze_portfolio, detect_market_regime

# Elite Features Imports
from models.rf_model import get_rf_feature_importance
from utils.risk_tools import calculate_risk_price_points

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
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# --- Helper Functions (with Caching) ---
@st.cache_data
def get_cached_data(ticker, start, end):
    try:
        df = fetch_stock_data(ticker, start, end)
        if df is not None and not df.empty:
            return handle_missing_values(df)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_ml_results(df, seq_length, n_days=7):
    try:
        data_scaled, scaler = scale_features(df[['Close']].values)
        X, y = create_sequences(data_scaled, seq_length)
        X_train, y_train, X_test, y_test = split_data(X, y)
        
        # Models
        rf_preds, rf_model = train_rf_model(X_train, y_train, X_test)
        rf_forecast = forecast_future_rf(rf_model, X_test[-1], days=n_days)
        
        lr_preds, lr_model = train_linear_regression(X_train, y_train, X_test)
        lr_forecast = forecast_future_lr(lr_model, X_test[-1], days=n_days)
        
        arima_forecast, arima_conf_int = train_arima(df['Close'].values, forecast_days=n_days)
        
        return {
            "y_test": y_test,
            "rf_preds": rf_preds, "rf_forecast": rf_forecast, "rf_model": rf_model,
            "lr_preds": lr_preds, "lr_forecast": lr_forecast,
            "arima_forecast": arima_forecast,
            "arima_conf_int": arima_conf_int,
            "scaler": scaler,
            "success": True
        }
    except Exception as e:
        print(f"ML Processing Error: {e}")
        return {"success": False}

# --- Tabbed Navigation ---
tab_dashboard, tab_analysis, tab_prediction, tab_strategy, tab_portfolio = st.tabs([
    "📊 Dashboard", "🔍 Analysis", "🔮 Prediction", "⚙️ Pro Strategy", "💼 Portfolio"
])

# --- Sidebar Enhanced: Watchlist & Alerts ---
with st.sidebar:
    st.markdown("### 📋 My Watchlist")
    for w_ticker in st.session_state.watchlist:
        with st.container():
            w_col1, w_col2, w_col3 = st.columns([2, 2, 1])
            w_col1.markdown(f"**{w_ticker}**")
            
            # Fetch mini-data for watchlist (use 7 days for better reliability)
            try:
                start_w = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                end_w = datetime.now().strftime('%Y-%m-%d')
                w_df = get_cached_data(w_ticker, start_w, end_w)
                
                if not w_df.empty and len(w_df) >= 2:
                    w_last = w_df['Close'].iloc[-1]
                    w_prev = w_df['Close'].iloc[-2]
                    w_change = ((w_last - w_prev) / w_prev) * 100
                    w_color = "green" if w_change >= 0 else "red"
                    w_col2.markdown(f"**${w_last:.2f}** <span style='color:{w_color}; font-size:0.8em;'>({w_change:+.2f}%)</span>", unsafe_allow_html=True)
                elif not w_df.empty:
                    w_col2.markdown(f"**${w_df['Close'].iloc[-1]:.2f}**")
                else:
                    w_col2.write("<small style='color: #888;'>Data unavailable</small>", unsafe_allow_html=True)
            except Exception as e:
                w_col2.write("<small style='color: #888;'>Data unavailable</small>", unsafe_allow_html=True)
                
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
            a_df = get_cached_data(alert['ticker'], (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            if not a_df.empty:
                if check_alerts(a_df['Close'].iloc[-1], alert['target'], alert['ticker'], alert['direction']):
                    alert['active'] = False 

# --- Sidebar: Live Mode ---
st.sidebar.markdown("---")
live_mode = st.sidebar.toggle("⚡ Pro Live Refresh (30s)", value=False)
if live_mode:
    st.sidebar.caption("Next refresh in: 30s")

# --- Sidebar: Elite Reporting ---
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Elite Reporting")
if st.sidebar.button("📊 Generate Institutional Report"):
    try:
        r_df = get_cached_data(ticker, start_date, end_date)
        r_df = add_indicators(r_df)
        r_sent = get_sentiment(ticker)
        
        # Build Report Data
        report_data = {
            "Symbol": ticker,
            "Price": f"${r_df['Close'].iloc[-1]:.2f}",
            "Sentiment": r_sent['label'],
            "Sentiment_Score": r_sent['score'],
            "Risk_Level": get_indicator_interpretation(r_df).get('Risk', 'N/A'),
            "Recommended_Action": get_recommendation(r_df['Close'].iloc[-1], r_df['Close'].iloc[-1] * 1.01, r_df['RSI'].iloc[-1], r_sent).get('action', 'HOLD')
        }
        
        report_df = pd.DataFrame([report_data])
        st.sidebar.download_button("Download Full Report (.csv)", data=report_df.to_csv(index=False), file_name=f"{ticker}_Elite_Report.csv", mime="text/csv")
        st.sidebar.success("Report Ready!")
    except Exception as e:
        st.sidebar.error("Report generation failed.")

# --- TAB: Dashboard ---
with tab_dashboard:
    st.title("Elite Market Terminal")
    
    # Elite Heatmap Section
    with st.expander("📊 Elite Market Heatmap", expanded=True):
        h_cols = st.columns(len(st.session_state.watchlist) if st.session_state.watchlist else 1)
        for idx, w_ticker in enumerate(st.session_state.watchlist):
            try:
                h_df = get_cached_data(w_ticker, (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
                if not h_df.empty:
                    h_val = h_df['Close'].iloc[-1]
                    h_change = ((h_val - h_df['Close'].iloc[-2]) / h_df['Close'].iloc[-2]) * 100
                    h_color = "#00ffcc" if h_change >= 0 else "#ff4b4b"
                    h_cols[idx % 5].markdown(f"""
                    <div style="background: {h_color}22; border: 1px solid {h_color}; padding: 15px; border-radius: 8px; text-align: center;">
                        <small>{w_ticker}</small><br>
                        <b style="color: {h_color}; font-size: 1.2em;">{h_change:+.2f}%</b><br>
                        <small>${h_val:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            except: pass

    st.markdown("---")
    
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
                
                df_rec = add_indicators(df)
                latest_rec = df_rec.iloc[-1]
                interpretations = get_indicator_interpretation(df_rec)
                
                rec = get_recommendation(
                    last_price, next_pred, latest_rec['RSI'], sentiment, 
                    latest_rec['MACD'], latest_rec['MACD_Signal'], 
                    latest_rec['BB_Upper'], latest_rec['BB_Lower']
                )
                
                # Risk Price Points
                risk_pts = calculate_risk_price_points(last_price, df, direction=rec['action'])
                
                st.markdown(f"""
                <div class="fintech-card" style="border-left: 5px solid {rec['color']};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="color: {rec['color']}; margin: 0;">{rec['action']}</h2>
                        <span style="background: {rec['color']}22; color: {rec['color']}; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; border: 1px solid {rec['color']};">STRENGTH: {rec['strength']}%</span>
                    </div>
                    <p style="margin-top: 10px;">{rec['explanation']}</p>
                    <hr style="border-color: #3e4451; margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <small style="color: #888;">STOP LOSS</small><br>
                            <b style="color: #ff4b4b;">${risk_pts['stop_loss']}</b>
                        </div>
                        <div>
                            <small style="color: #888;">PROFIT TARGET</small><br>
                            <b style="color: #00ffcc;">${risk_pts['target_profit']}</b>
                        </div>
                    </div>
                </div>
                
                <div class="fintech-card">
                    <h3>🧠 Explainable AI (Why this prediction?)</h3>
                    <p><small style="color: #888;">Feature Importance contribution to forecast</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                # XAI Chart
                res_xai = get_ml_results(df, 60, n_days=1)
                xai_data = get_rf_feature_importance(res_xai.get('rf_model')) if res_xai.get('success') else {}
                if xai_data:
                    fig_xai = px.bar(x=list(xai_data.values()), y=list(xai_data.keys()), orientation='h', template="plotly_dark")
                    fig_xai.update_traces(marker_color='cyan')
                    fig_xai.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Weight", yaxis_title="")
                    st.plotly_chart(fig_xai, use_container_width=True)

                st.markdown(f"""
                <div class="fintech-card">
                    <h3>📰 Elite News Sentiment</h3>
                    <p><b>Outlook:</b> <span style="color: {'#00ffcc' if sentiment['label'] == 'Positive' else '#ff4b4b' if sentiment['label'] == 'Negative' else '#f1c40f'};">{sentiment['label']}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Latest Headlines"):
                    for news_item in sentiment.get('headlines', []):
                        st.markdown(f"- [{news_item['title']}]({news_item['link']}) ({news_item['publisher']})")
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
            
            with st.expander("📈 Correlation Intelligence"):
                corr_ticker = st.text_input("Compare correlation with:", value="SPY").upper()
                if corr_ticker and corr_ticker != ticker:
                    df_corr = get_cached_data(corr_ticker, start_date, end_date)
                    if not df_corr.empty:
                        corr_val = df['Close'].corr(df_corr['Close'])
                        st.metric(f"Correlation: {ticker} vs {corr_ticker}", f"{corr_val:.2f}", 
                                help="High correlation (>0.7) means they move together.")
            
            # Indicators Interpretation
            st.subheader("Deep-Dive Insights")
            df_ind = add_indicators(df)
            interpretations = get_indicator_interpretation(df_ind)
            
            ic1, ic2, ic3 = st.columns(3)
            ic1.info(f"**RSI**: {interpretations.get('RSI', 'N/A')}")
            ic2.info(f"**MACD**: {interpretations.get('MACD', 'N/A')}")
            ic3.info(f"**Bollinger**: {interpretations.get('BB', 'N/A')}")
            
            st.success(f"**Overall Trend**: {interpretations.get('Trend', 'Trend data unavailable')}")
            
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
                    
                    # Create actual future dates for x-axis
                    last_date = p_df.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
                    
                    conf_int = results['arima_conf_int']
                    # Ensure conf_int is handled as a numpy array for indexing
                    if isinstance(conf_int, pd.DataFrame):
                        conf_int = conf_int.values
                        
                    fig = plot_forecast_with_confidence(
                        future_dates, inv(results['rf_forecast']), 
                        conf_int[:, 0], conf_int[:, 1], 
                        historical_x=p_df.index[-15:], 
                        historical_y=p_df['Close'].iloc[-15:],
                        model_name="Ensemble AI"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    forecast_df = pd.DataFrame({
                        "Date": future_dates, 
                        "Predicted_Price": inv(results['rf_forecast']),
                        "Lower_Bound": conf_int[:, 0],
                        "Upper_Bound": conf_int[:, 1]
                    })
                    st.download_button("📥 Download Forecast CSV", data=forecast_df.to_csv(index=False), file_name=f"{p_ticker}_forecast.csv", mime="text/csv")

                with col_metrics:
                    st.subheader("Asset Risk Profile")
                    risk_m = get_risk_assessment_metrics(p_df)
                    st.markdown(f"""
                    <div class="fintech-card">
                        <small style="color: #888;">SHARPE RATIO</small>
                        <h2 style='color: #00ffcc;'>{risk_m['sharpe_ratio']:.2f}</h2>
                        <small>Risk-Adjusted Return</small>
                    </div>
                    <div class="fintech-card">
                        <small style="color: #888;">MAX DRAWDOWN</small>
                        <h2 style='color: #ff4b4b;'>{risk_m['max_drawdown']:.2%}</h2>
                        <small>Peak-to-Trough Loss</small>
                    </div>
                    """, unsafe_allow_html=True)
                    from utils.evaluator import compare_models
                    st.write("### Cross-Validation")
                    st.table(compare_models({"RF": (results['y_test'], results['rf_preds']), "LR": (results['y_test'], results['lr_preds'])}))
        else:
            st.warning("Enter a valid ticker to start forecasting.")

# --- TAB: Pro Strategy ---
with tab_strategy:
    st.title("Algorithmic Trading Simulation")
    s_ticker = st.text_input("Ticker to Backtest", value=ticker, key="strat_ticker").upper()
    
    with st.spinner("Simulating AI Strategy..."):
        s_df = get_cached_data(s_ticker, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
        if not s_df.empty:
            s_df = add_indicators(s_df)
            bt_results = backtest_strategy(s_df)
            
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Return", f"{bt_results['total_return_pct']:.2f}%")
            sc2.metric("Win Rate", f"{bt_results['win_rate']:.1f}%")
            sc3.metric("Total Trades", bt_results['total_trades'])
            sc4.metric("Ending Capital", f"${bt_results['final_capital']:,.2f}")
            
            st.markdown("---")
            # Equity Curve Chart
            df_bt = pd.DataFrame({"Equity": bt_results['equity_curve']}, index=s_df.index)
            fig_bt = px.line(df_bt, y="Equity", title=f"Equity Growth Simulation: {s_ticker}", template="plotly_dark")
            fig_bt.update_traces(line_color="cyan", line_width=3)
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning("Enter a valid ticker for backtesting.")

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
        
        # AI Advisor Integration
        advisor = analyze_portfolio(st.session_state.portfolio)
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Market Value", f"${summary['total_current_value']:,.2f}")
        sc2.metric("Unrealized P/L", f"${summary['total_pl']:,.2f}", f"{summary['total_pl_pct']:.2f}%")
        sc3.metric("Portfolio Status", advisor['status'])
        
        # Portfolio AI Advisor Alerts
        if advisor['warnings']:
            with st.expander("🤖 AI Portfolio Advisor Alert", expanded=True):
                for warn in advisor['warnings']:
                    st.warning(warn)
                for sug in advisor['suggestions']:
                    st.info(f"💡 Suggestion: {sug}")
        
        st.markdown("---")
        col_t, col_p = st.columns([3, 2])
        with col_t:
            st.subheader("Holdings Summary")
            st.dataframe(df_port, use_container_width=True)
            
            # Trade History log
            st.subheader("📝 Trade Activity Log")
            if st.session_state.trade_history:
                st.dataframe(pd.DataFrame(st.session_state.trade_history), use_container_width=True)
            else:
                st.caption("No trade activity recorded yet.")
        with col_p:
            st.subheader("Asset Allocation")
            st.plotly_chart(plot_allocation_chart(df_port), use_container_width=True)
    else:
        st.info("Your portfolio is currently empty.")

st.sidebar.markdown("---")
st.sidebar.write("⚡ v4.5.0 Pro | AI Trading Assistant")

# --- Auto-Refresh Logic ---
if live_mode:
    import time
    time.sleep(30)
    st.rerun()
