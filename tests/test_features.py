"""
Unit tests for FeatureEngineer, temporal features, cyclical encodings, and leakage safety.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.pipeline import FeatureEngineer
from src.config import TARGET_COL, TEMP_COL, HUMIDITY_COL


def create_sample_hourly_df(n_hours: int = 1000) -> pd.DataFrame:
    """Helper to generate a clean synthetic hourly dataframe."""
    dates = pd.date_range("2023-01-01", periods=n_hours, freq="1h")
    np.random.seed(42)
    return pd.DataFrame({
        TARGET_COL: 4000 + 500 * np.sin(2 * np.pi * np.arange(n_hours) / 24.0) + np.random.normal(0, 50, n_hours),
        TEMP_COL: 25 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24.0),
        HUMIDITY_COL: 50 + 10 * np.cos(2 * np.pi * np.arange(n_hours) / 24.0),
    }, index=dates)


def test_feature_engineer_temporal_and_calendar():
    """Test extraction of calendar, temporal, and holiday proximity features."""
    df = create_sample_hourly_df(100)
    engineer = FeatureEngineer(country="IN")
    df_feat = engineer.add_temporal_features(df)
    df_feat = engineer.add_calendar_holiday_features(df_feat)

    assert "hour" in df_feat.columns
    assert "day_of_week" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    assert "is_holiday" in df_feat.columns
    assert "days_until_holiday" in df_feat.columns
    assert "days_since_holiday" in df_feat.columns
    assert "season" in df_feat.columns


def test_feature_engineer_cyclical():
    """Test sine and cosine cyclical encodings."""
    df = create_sample_hourly_df(48)
    engineer = FeatureEngineer()
    df_feat = engineer.add_fourier_cyclical_features(df)

    assert "sin_hour" in df_feat.columns
    assert "cos_hour" in df_feat.columns
    assert "sin_week" in df_feat.columns
    assert "cos_week" in df_feat.columns

    # Verify cyclical bounding: sin and cos must lie in [-1, 1]
    assert (df_feat["sin_hour"] >= -1.0).all() and (df_feat["sin_hour"] <= 1.0).all()
    assert (df_feat["cos_hour"] >= -1.0).all() and (df_feat["cos_hour"] <= 1.0).all()


def test_feature_engineer_rolling_leak_safety():
    """Test that rolling aggregations are strictly shifted by 1 to prevent data leakage."""
    df = create_sample_hourly_df(300)
    engineer = FeatureEngineer()
    df_feat = engineer.add_hourly_rolling_aggregations(df)

    # The rolling mean for step t should NOT include target value at step t
    # Specifically, rolling_mean_24h at index 24 should equal mean(df[0:24])
    expected_mean = df[TARGET_COL].iloc[0:24].mean()
    actual_mean = df_feat["rolling_mean_24h"].iloc[24]
    assert actual_mean == pytest.approx(expected_mean, rel=1e-4)
