import streamlit as st

def check_alerts(current_price, target_price, ticker, direction="above"):
    """
    Checks if current price meets target and shows a UI notification.
    """
    if direction == "above" and current_price >= target_price:
        st.toast(f"🚀 ALERT: {ticker} has crossed ABOVE your target of ${target_price}!", icon="📈")
        return True
    elif direction == "below" and current_price <= target_price:
        st.toast(f"⚠️ ALERT: {ticker} has dropped BELOW your target of ${target_price}!", icon="📉")
        return True
    return False

def show_alert_ui():
    """Sidebar UI for setting alerts."""
    st.sidebar.markdown("### 🔔 Price Alerts")
    with st.sidebar.expander("Configure New Alert", expanded=False):
        alert_ticker = st.text_input("Ticker", value="AAPL", key="alert_ticker_input").upper()
        target = st.number_input("Target Price ($)", value=200.0, step=1.0)
        direction = st.selectbox("Trigger when price is:", ["above", "below"], index=0)
        
        if st.button("Set Smart Alert", use_container_width=True):
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            st.session_state.alerts.append({"ticker": alert_ticker, "target": target, "direction": direction, "active": True})
            st.success(f"Alert set for {alert_ticker} {direction} ${target}")
