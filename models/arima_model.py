import pmdarima as pm
import numpy as np

def train_arima(data, forecast_days=7):
    """
    Trains an ARIMA model on the series and forecasts N days.
    """
    try:
        model = pm.auto_arima(data, seasonal=False, stepwise=True, suppress_warnings=True)
        forecast = model.predict(n_periods=forecast_days)
        return forecast, model
    except Exception as e:
        print(f"ARIMA error: {e}")
        return np.zeros(forecast_days), None
