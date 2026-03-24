import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values by forward filling and backward filling.
    
    Args:
        df (pd.DataFrame): Dataframe with potential missing values.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    return df.ffill().bfill()

def scale_features(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scales the data to human-friendly range [0, 1] using MinMaxScaler.
    
    Args:
        data (np.ndarray): Input feature array.
        
    Returns:
        Tuple[np.ndarray, MinMaxScaler]: Scaled data and the scaler object for inverse transformation.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences for LSTM training.
    
    Args:
        data (np.ndarray): The scaled data.
        seq_length (int): Number of time steps to look back.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and y (targets) for LSTM.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length, 0])  # Assuming first column is the target (Close)
        
    return np.array(X), np.array(y)

def split_data(X: np.ndarray, y: np.ndarray, train_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training and testing sets.
    
    Args:
        X (np.ndarray): Sequences features.
        y (np.ndarray): Targets.
        train_split (float): Ratio for training set.
        
    Returns:
        Tuple: X_train, y_train, X_test, y_test.
    """
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test Preprocessing
    dummy_data = np.random.rand(100, 1)
    scaled, scaler = scale_features(dummy_data)
    X, y = create_sequences(scaled, 10)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
