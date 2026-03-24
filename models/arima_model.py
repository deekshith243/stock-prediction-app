import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(data, forecast_days=7):
    """
    Trains a simple ARIMA model using statsmodels (Lightweight).
    """
    try:
        # Simple (5,1,0) ARIMA for speed and stability
        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
        return forecast, model_fit
    except Exception as e:
        print(f"ARIMA error: {e}")
        return np.zeros(forecast_days), None
