"""
Unit tests for DynamicUncertaintyEstimator and MultiHorizonForecastEngine.
"""

import pytest
import numpy as np
import pandas as pd
from src.forecast.uncertainty import DynamicUncertaintyEstimator


def test_dynamic_uncertainty_growth():
    """Test that uncertainty standard error expands monotonically with forecast lead time."""
    estimator = DynamicUncertaintyEstimator(base_sigma_mw=130.0, lead_time_growth_alpha=0.45)

    sigma_1h = estimator.calculate_sigma_at_horizon(1)
    sigma_24h = estimator.calculate_sigma_at_horizon(24)
    sigma_168h = estimator.calculate_sigma_at_horizon(168)
    sigma_720h = estimator.calculate_sigma_at_horizon(720)

    # Monotonic expansion
    assert sigma_1h < sigma_24h < sigma_168h < sigma_720h


def test_uncertainty_intervals_attachment():
    """Test that attached intervals have correct ordering (lower < forecast < upper)."""
    estimator = DynamicUncertaintyEstimator()
    dates = pd.date_range("2024-01-01", periods=24, freq="1h")
    preds = pd.Series(5000.0, index=dates)

    df_bounds = estimator.attach_prediction_intervals(preds)

    assert "lower_bound_90" in df_bounds.columns
    assert "upper_bound_90" in df_bounds.columns
    assert "lower_bound_95" in df_bounds.columns
    assert "upper_bound_95" in df_bounds.columns

    # Verify bound relationships
    assert (df_bounds["lower_bound_95"] <= df_bounds["lower_bound_90"]).all()
    assert (df_bounds["lower_bound_90"] <= df_bounds["forecast_mw"]).all()
    assert (df_bounds["forecast_mw"] <= df_bounds["upper_bound_90"]).all()
    assert (df_bounds["upper_bound_90"] <= df_bounds["upper_bound_95"]).all()
