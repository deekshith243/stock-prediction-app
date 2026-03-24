# 🚨 failure Handling & Troubleshooting: Streamlit Cloud

Use this guide to instantly resolve any deployment issues on Streamlit Cloud.

| Error | Cause | Fix |
|---|---|---|
| **ModuleNotFoundError** | Missing dependency in `requirements.txt`. | Add the missing package to `requirements.txt` and push to GitHub. |
| **Out of Memory (OOM)** | LSTM or Prophet training using too much RAM. | Reduce the `seq_length` (slider) or use a shorter date range in the sidebar. |
| **TensorFlow DLL Error** | Incompatible TF version or missing libraries. | Ensure `tensorflow-cpu` is used instead of `tensorflow`. |
| **yfinance empty DataFrame** | Invalid ticker or API rate limiting. | Verify the ticker on Yahoo Finance. The app handles this via `st.error()` without crashing. |
| **Prophet Build Failure** | Compilation issues with `pystan` or `cmdstanpy`. | Ensure `prophet==1.1.5` is pinned; Streamlit Cloud has a pre-built wheel for this. |
| **App Stuck in "Running"** | Cache is corrupted or model is in a loop. | Go to Streamlit Cloud dashboard -> Settings -> **Clear Cache** and reboot the app. |

### 🛠️ Runtime Best Practices
- **Reboot often**: Use the "Reboot" button in the Streamlit Cloud menu if UI becomes unresponsive.
- **Python Version**: Select **3.10** in the "Advanced Settings" during deployment for best compatibility.
- **CPU Only**: Ensure `CUDA_VISIBLE_DEVICES = -1` is set in your model code (Integrated in `models/lstm_model.py`).
