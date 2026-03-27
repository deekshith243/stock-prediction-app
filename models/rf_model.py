import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_rf_model(X_train, y_train, X_test):
    """
    Trains a RandomForestRegressor for stock price prediction.
    """
    # Flatten X if it's 3D (like for LSTM)
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_flat = X_train
        X_test_flat = X_test

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_flat, y_train)
    
    preds = model.predict(X_test_flat)
    return preds, model

def forecast_future_rf(model, last_sequence, days=7):
    """
    Generates multi-step future forecasts using the trained RF model.
    """
    # Ensure last_sequence is flat
    current_batch = last_sequence.flatten().reshape(1, -1)
    future_preds = []
    
    # For a simple RF, we can't easily do recursive multi-step if it was trained on sequences
    # But for this dashboard, we'll provide a 7-day trend extrapolation or recursive prediction
    # Here we do recursive (feeding last pred back in)
    temp_batch = current_batch.copy()
    
    for _ in range(days):
        pred = model.predict(temp_batch)[0]
        future_preds.append(pred)
        
        # Shift and append (assuming seq_length was used)
        # For RF, we just shift the flat array
        temp_batch = np.roll(temp_batch, -1)
        temp_batch[0, -1] = pred
        
    return np.array(future_preds)

def get_rf_feature_importance(model, feature_names=None):
    """
    Returns the feature importance from the trained RF model.
    """
    try:
        import pandas as pd
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # Use provided names or default labels
            if feature_names is None:
                feature_names = ["Price Momentum", "Recent Trend", "Volatility Factor", "Volume Intensity", "Sentiment Weight"]
                # Adjust length to match importances
                if len(importance) > len(feature_names):
                    feature_names += [f"Feature {i}" for i in range(len(feature_names), len(importance))]
                feature_names = feature_names[:len(importance)]
            
            df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance
            }).sort_values(by="importance", ascending=False)
            return df
        return None
    except Exception:
        return None
