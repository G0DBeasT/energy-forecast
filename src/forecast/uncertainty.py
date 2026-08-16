"""
Dynamic Horizon-Dependent Uncertainty Estimation.

Computes statistically sound prediction intervals where forecast uncertainty
widens non-linearly with lead time:
sigma(h) = sigma_base * sqrt(1 + alpha * (h / 24)^gamma)
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


class DynamicUncertaintyEstimator:
    """
    Estimator for multi-step forecast prediction intervals.

    Accounts for compounding auto-regressive error and long-range meteorological
    entropy over extended lead times (up to 720h).
    """

    def __init__(
        self,
        base_sigma_mw: float = 145.0,
        lead_time_growth_alpha: float = 0.45,
        growth_power_gamma: float = 0.65,
    ):
        self.base_sigma_mw = base_sigma_mw
        self.alpha = lead_time_growth_alpha
        self.gamma = growth_power_gamma

    def calculate_sigma_at_horizon(self, horizon_step: int) -> float:
        """
        Calculate expected forecast error standard deviation at step h (hours).
        """
        lead_days = max(horizon_step, 1) / 24.0
        scale_factor = np.sqrt(1.0 + self.alpha * (lead_days ** self.gamma))
        return float(self.base_sigma_mw * scale_factor)

    def attach_prediction_intervals(
        self,
        forecast_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Add 80% (z=1.282), 90% (z=1.645), and 95% (z=1.960) prediction bounds to forecast series.

        Args:
            forecast_series: pd.Series of point forecasts with DatetimeIndex.

        Returns:
            pd.DataFrame with point forecast and confidence interval bounds.
        """
        n_steps = len(forecast_series)
        sigmas = np.array([self.calculate_sigma_at_horizon(h + 1) for h in range(n_steps)])

        predictions = forecast_series.values

        df_bounds = pd.DataFrame(
            {
                "forecast_mw": np.round(predictions, 2),
                "lower_bound_80": np.round(np.maximum(predictions - 1.282 * sigmas, 0), 2),
                "upper_bound_80": np.round(predictions + 1.282 * sigmas, 2),
                "lower_bound_90": np.round(np.maximum(predictions - 1.645 * sigmas, 0), 2),
                "upper_bound_90": np.round(predictions + 1.645 * sigmas, 2),
                "lower_bound_95": np.round(np.maximum(predictions - 1.960 * sigmas, 0), 2),
                "upper_bound_95": np.round(predictions + 1.960 * sigmas, 2),
                "uncertainty_sigma_mw": np.round(sigmas, 2),
            },
            index=forecast_series.index,
        )

        return df_bounds
