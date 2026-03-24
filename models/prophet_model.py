from prophet import Prophet
import pandas as pd
import numpy as np

def train_prophet(df: pd.DataFrame, forecast_days: int = 7):
    """
    Trains a Prophet model and returns the forecast.
    
    Args:
        df: DataFrame with 'Close' column and Index as Date.
        forecast_days: Number of days to forecast.
    """
    try:
        # Prophet requires 'ds' and 'y' columns
        prophet_df = df.reset_index()[['Date', 'Close']]
        prophet_df.columns = ['ds', 'y']
        
        # Remove timezone if exists
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Extract only the predicted future values
        predicted_values = forecast['yhat'].tail(forecast_days).values
        
        return predicted_values, model, forecast
    except Exception as e:
        print(f"Prophet error: {e}")
        return np.zeros(forecast_days), None, None

def plot_prophet_components(model, forecast):
    """Returns prophet component plots."""
    if model and forecast is not None:
        return model.plot_components(forecast)
    return None
