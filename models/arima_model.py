import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(data, forecast_days=7):
    """
    Trains a simple ARIMA model and returns forecast with confidence intervals.
    """
    try:
        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast_obj = model_fit.get_forecast(steps=forecast_days)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05) # 95% CI
        return forecast, conf_int
    except Exception as e:
        print(f"ARIMA error: {e}")
        return np.zeros(forecast_days), np.zeros((forecast_days, 2))
