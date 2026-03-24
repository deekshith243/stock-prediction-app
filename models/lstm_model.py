import numpy as np
import tensorflow as tf
import os

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    Builds a robust multi-layer LSTM model with Dropout.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(X_train, y_train, X_test, model=None, epochs=20, batch_size=32):
    """
    Trains the model and returns predictions for X_test.
    """
    if model is None:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    predictions = model.predict(X_test)
    return predictions, model

def forecast_future(model, last_sequence, days=7):
    """
    Forecasts the next N days.
    """
    future_predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]), verbose=0)
        future_predictions.append(pred[0, 0])
        
        # Shift sequence and add prediction
        new_row = current_seq[-1].copy()
        new_row[0] = pred[0, 0] # Update first feature (usually Close)
        current_seq = np.append(current_seq[1:], [new_row], axis=0)
        
    return np.array(future_predictions)
