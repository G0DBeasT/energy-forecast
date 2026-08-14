import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred) -> dict:
    """Calculate key time series regression evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mape": float(mape),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
