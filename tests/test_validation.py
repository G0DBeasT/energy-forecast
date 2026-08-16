"""
Unit tests for DataQualityValidator.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.validation import DataQualityValidator, DataQualityError
from src.config import TARGET_COL, TEMP_COL, HUMIDITY_COL


def test_validator_valid_dataset():
    """Test validator passes on clean valid time series."""
    dates = pd.date_range("2024-01-01", periods=48, freq="1h")
    df = pd.DataFrame({
        TARGET_COL: np.random.uniform(4000, 6000, 48),
        TEMP_COL: np.random.uniform(15, 35, 48),
        HUMIDITY_COL: np.random.uniform(30, 80, 48),
    }, index=dates)

    validator = DataQualityValidator()
    report = validator.validate(df, allow_missing=False)
    assert report["passed"] is True
    assert report["total_rows"] == 48


def test_validator_detects_negative_demand():
    """Test validator raises DataQualityError when negative demand is present."""
    dates = pd.date_range("2024-01-01", periods=24, freq="1h")
    df = pd.DataFrame({
        TARGET_COL: [5000.0] * 23 + [-100.0],
        TEMP_COL: [25.0] * 24,
    }, index=dates)

    validator = DataQualityValidator()
    with pytest.raises(DataQualityError, match="non-positive or negative demand"):
        validator.validate(df)


def test_validator_detects_missing_values():
    """Test validator raises DataQualityError when unhandled NaNs exist."""
    dates = pd.date_range("2024-01-01", periods=24, freq="1h")
    df = pd.DataFrame({
        TARGET_COL: [5000.0] * 23 + [np.nan],
        TEMP_COL: [25.0] * 24,
    }, index=dates)

    validator = DataQualityValidator()
    with pytest.raises(DataQualityError, match="Unresolved missing values"):
        validator.validate(df, allow_missing=False)


def test_validator_detects_non_monotonic_timestamps():
    """Test validator raises DataQualityError on scrambled timestamps."""
    dates = [pd.Timestamp("2024-01-01 02:00"), pd.Timestamp("2024-01-01 01:00")]
    df = pd.DataFrame({TARGET_COL: [5000.0, 5100.0]}, index=dates)

    validator = DataQualityValidator()
    with pytest.raises(DataQualityError, match="monotonically increasing"):
        validator.validate(df)
