import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """
    Calculates RMSE and MAE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def compare_models(results_dict):
    """
    Creates a comparison table for different models.
    
    Args:
        results_dict: { model_name: (y_true, y_pred) }
    """
    summary = []
    for model_name, (y_true, y_pred) in results_dict.items():
        rmse, mae = calculate_metrics(y_true, y_pred)
        summary.append({
            "Model": model_name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })
        
    df = pd.DataFrame(summary)
    # Mark best model (lowest RMSE)
    df['Status'] = df['RMSE'].apply(lambda x: 'Best ⭐' if x == df['RMSE'].min() else '')
    return df
