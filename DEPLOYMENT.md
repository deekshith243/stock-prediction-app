# 🚀 Deployment Guide: Streamlit Cloud

Follow these exact steps to deploy your **GrowthFlow AI** dashboard with zero errors.

### 1. Prerequisites
- A GitHub account.
- Your project pushed to a GitHub repository.
- A [Streamlit Cloud](https://share.streamlit.io/) account (free).

### 2. Repository Checklist
Ensure your repo contains:
- `app.py` (Main entry point)
- `requirements.txt` (Pinned dependencies)
- `models/` folder (All model scripts)
- `utils/` folder (All utility scripts)

### 3. Deployment Steps
1. Log in to **Streamlit Cloud**.
2. Click **"New App"**.
3. Repository: `deekshith243/stock-prediction-app`
4. Branch: `main`
5. Main file path: `app.py`
6. **Advanced Settings**:
   - Python version: **3.10**
6. Click **"Deploy!"**.

### 4. Post-Deployment Verification
- Navigate to the **"AI Predictions"** tab.
- Enter a common ticker like `AAPL` or `TSLA`.
- Verify that the models train (spinners appear instantly) and the forecast chart displays.

### 5. Troubleshooting
- **ModuleNotFoundError**: Ensure the module is listed in `requirements.txt`.
- **Resource Limits**: The app is now extremely lightweight. If it restarts, check your internet connection or the yfinance API status.
