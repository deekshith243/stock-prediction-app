import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train, X_test):
    """
    Trains a simple Linear Regression model and returns predictions.
    """
    # Flatten X if it's 3D (samples, time_steps, features) for LR
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = LinearRegression()
    model.fit(X_train_flat, y_train)
    predictions = model.predict(X_test_flat)
    return predictions, model

def forecast_future_lr(model, last_sequence, days=7):
    """
    LR-based future forecasting.
    """
    future_predictions = []
    current_seq = last_sequence.copy()
    
    # Needs to be flat for LR
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, -1))
        future_predictions.append(pred[0])
        
        # Shift
        new_row = current_seq.flatten()
        # Simple shift logic for flat sequence
        new_row = np.append(new_row[1:], pred[0])
        current_seq = new_row.reshape(last_sequence.shape)
        
    return np.array(future_predictions)
