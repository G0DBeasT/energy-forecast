"""
Unit tests for data loaders, data cleaner, and provenance tracking.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.synthetic import SyntheticGridLoader
from src.data.clean import DataCleaner
from src.data.csv_loader import CSVGridLoader
from src.config import TARGET_COL, DATE_COL, TEMP_COL, HUMIDITY_COL


def test_synthetic_grid_loader_generation():
    """Test that SyntheticGridLoader produces a valid DataFrame with expected columns and non-null dates."""
    loader = SyntheticGridLoader(start_date="2024-01-01", end_date="2024-01-07", seed=42)
    df = loader.generate()

    assert isinstance(df.index, pd.DatetimeIndex)
    assert TARGET_COL in df.columns
    assert TEMP_COL in df.columns
    assert HUMIDITY_COL in df.columns
    assert len(df) == 7 * 24
    assert loader.data_provenance == "synthetic_simulated"

    meta = loader.get_metadata()
    assert meta["source_type"] == "synthetic_simulated"
    assert "seed" in meta


def test_data_cleaner_resampling_and_imputation():
    """Test that DataCleaner enforces 1h frequency and imputes missing values."""
    dates = pd.date_range("2024-01-01 00:00", "2024-01-02 23:00", freq="1h")
    np.random.seed(42)
    demand = np.random.normal(5000, 200, len(dates))
    demand[5] = np.nan
    demand[6] = np.nan
    demand[20] = 999999.0  # extreme outlier

    df_raw = pd.DataFrame({
        TARGET_COL: demand,
        TEMP_COL: np.random.uniform(15, 30, len(dates)),
        HUMIDITY_COL: np.random.uniform(30, 80, len(dates)),
    }, index=dates)

    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    assert not df_clean[TARGET_COL].isnull().any()
    assert len(df_clean) == 48
    # Verify extreme outlier was clipped
    assert df_clean[TARGET_COL].max() < 10000.0


def test_csv_grid_loader(tmp_path):
    """Test that CSVGridLoader correctly standardizes external CSV schema."""
    csv_file = tmp_path / "custom_grid.csv"
    dates = pd.date_range("2024-01-01", periods=24, freq="1h")
    df_dummy = pd.DataFrame({
        "timestamp": dates,
        "load_mw": np.random.uniform(4000, 6000, 24),
        "temperature": np.random.uniform(20, 25, 24),
    })
    df_dummy.to_csv(csv_file, index=False)

    loader = CSVGridLoader(
        filepath=csv_file,
        date_col="timestamp",
        demand_col="load_mw",
        temp_col="temperature",
    )
    df_loaded = loader.load()

    assert isinstance(df_loaded.index, pd.DatetimeIndex)
    assert TARGET_COL in df_loaded.columns
    assert TEMP_COL in df_loaded.columns
    assert loader.data_provenance == "real_recorded"
