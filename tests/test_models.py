"""
Unit tests for model evaluation metrics, baselines, and ModelRegistry.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.evaluate import calculate_metrics, evaluate_horizon_segments
from src.models.baselines import SeasonalNaiveModel, RidgeLinearBaseline
from src.models.registry import ModelRegistry


def test_calculate_metrics():
    """Test evaluation metric calculations."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    m = calculate_metrics(y_true, y_pred)
    assert "mape" in m
    assert "mae" in m
    assert "rmse" in m
    assert "r2" in m
    assert "wape" in m

    assert m["mae"] == 10.0
    assert m["rmse"] == 10.0
    assert m["wape"] == 40.0 / 1000.0
    assert m["r2"] > 0.98


def test_seasonal_naive_baseline():
    """Test SeasonalNaiveModel predictions using lag column."""
    df_feat = pd.DataFrame({
        "lag_168h": [4500.0, 4600.0, 4700.0],
        "other_feat": [1, 2, 3],
    })
    model = SeasonalNaiveModel(seasonal_lag=168, fallback_col="lag_168h")
    model.fit(df_feat)
    preds = model.predict(df_feat)

    np.testing.assert_array_equal(preds, np.array([4500.0, 4600.0, 4700.0]))


def test_model_registry_save_and_load(tmp_path):
    """Test ModelRegistry serialization and metadata persistence."""
    from sklearn.linear_model import Ridge

    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]})
    y = pd.Series([10.0, 20.0, 30.0])

    model = Ridge().fit(X, y)
    saved_path = ModelRegistry.save_model(
        model=model,
        feature_cols=["feat1", "feat2"],
        metrics={"mape": 0.02},
        model_name="test_ridge",
        destination_dir=tmp_path,
    )

    assert saved_path.exists()

    loaded_model, meta = ModelRegistry.load_model(saved_path)
    assert meta["model_name"] == "test_ridge"
    assert meta["feature_columns"] == ["feat1", "feat2"]
    assert meta["metrics"]["mape"] == 0.02

    preds = loaded_model.predict(X)
    assert len(preds) == 3
