# 🚨 failure Handling & Troubleshooting: Streamlit Cloud

Use this guide to instantly resolve any deployment issues on Streamlit Cloud.

| Error | Cause | Fix |
|---|---|---|
| **ModuleNotFoundError** | Missing dependency in `requirements.txt`. | Add the missing package to `requirements.txt` and push to GitHub. |
| **Out of Memory (OOM)** | Large data range processed by Prophet. | Reduce the date range in the sidebar. |
| **yfinance empty DataFrame** | Invalid ticker or API rate limiting. | Verify the ticker on Yahoo Finance. The app handles this via `st.error()` without crashing. |
| **App Stuck in "Running"** | Cache is corrupted. | Go to Streamlit Cloud dashboard -> Settings -> **Clear Cache** and reboot the app. |

### 🛠️ Runtime Best Practices
- **Reboot often**: Use the "Reboot" button in the Streamlit Cloud menu if UI becomes unresponsive.
- **Python Version**: Select **3.10** in the "Advanced Settings" during deployment for best compatibility.
- **CPU Only**: Ensure `CUDA_VISIBLE_DEVICES = -1` is set in your model code (Integrated in `models/lstm_model.py`).
