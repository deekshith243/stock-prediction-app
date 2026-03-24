import streamlit as st

def check_alerts(current_price, target_price, ticker):
    """
    Checks if current price meets target and shows a UI notification.
    In a real app, this could send emails via SendGrid or AWS SES.
    """
    if current_price >= target_price:
        st.toast(f"🔔 ALERT: {ticker} has reached your target of ${target_price}!")
        return True
    return False

def show_alert_ui():
    """Sidebar UI for setting alerts."""
    st.sidebar.subheader("🔔 Price Alerts")
    alert_ticker = st.sidebar.text_input("Alert Ticker", value="AAPL").upper()
    target = st.sidebar.number_input("Target Price", value=200.0)
    if st.sidebar.button("Set Alert"):
        st.sidebar.success(f"Alert set for {alert_ticker} at ${target}")
