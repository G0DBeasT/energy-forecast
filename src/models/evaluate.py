"""
Model Evaluation Metrics & Horizon Performance Diagnostics.

Calculates standard regression metrics (MAPE, MAE, RMSE, R2, WAPE)
and provides horizon-segmented performance breakdowns.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive time series regression metrics.

    Metrics:
    - MAPE: Mean Absolute Percentage Error (%)
    - MAE: Mean Absolute Error (MW)
    - RMSE: Root Mean Squared Error (MW)
    - R2: Coefficient of Determination
    - WAPE: Weighted Absolute Percentage Error (%)
    - MAX_ERR: Maximum Absolute Residual (MW)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {"mape": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "wape": float("nan")}

    mape = float(mean_absolute_percentage_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))
    wape = float(np.sum(np.abs(y_t - y_p)) / np.sum(np.abs(y_t)))
    max_err = float(np.max(np.abs(y_t - y_p)))

    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "wape": wape,
        "max_err": max_err,
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Legacy wrapper for backwards compatibility."""
    return calculate_metrics(y_true, y_pred)


def evaluate_horizon_segments(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate forecast performance across different lead-time segments.
    (e.g., 1-24h Day 1, 25-168h Week 1, 169-720h Month 1).
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.dropna()

    results = []
    segments = [
        ("Next-Day (1-24h)", 0, 24),
        ("Next-Week (1-168h)", 0, 168),
        ("Next-Month (1-720h)", 0, 720),
    ]

    for label, start, end in segments:
        seg_df = df.iloc[start:end]
        if len(seg_df) > 0:
            m = calculate_metrics(seg_df["y_true"].values, seg_df["y_pred"].values)
            m["segment"] = label
            m["n_observations"] = len(seg_df)
            results.append(m)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.set_index("segment")
    return res_df
