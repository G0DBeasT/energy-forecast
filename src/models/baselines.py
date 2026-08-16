"""
Baseline Models for Energy Grid Forecasting.

Provides transparent reference benchmarks against which gradient boosted models
must be evaluated.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, LinearRegression


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):
    """
    Seasonal Naive Benchmark Model.
    Predicts demand using the same hour from 1 week ago (lag 168h) or 1 day ago (lag 24h).
    """

    def __init__(self, seasonal_lag: int = 168, fallback_col: str = "lag_168h"):
        self.seasonal_lag = seasonal_lag
        self.fallback_col = fallback_col
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.fallback_col in X.columns:
            return X[self.fallback_col].values
        if "lag_24h" in X.columns:
            return X["lag_24h"].values
        raise ValueError(f"Neither '{self.fallback_col}' nor 'lag_24h' found in feature columns.")


class RidgeLinearBaseline(BaseEstimator, RegressorMixin):
    """
    Regularized Ridge Linear Regression Baseline.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
