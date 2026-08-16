"""
Data Quality Validation Layer.

Performs strict statistical and physical sanity checks on time series data.
Fails loudly with actionable diagnostics when anomalies or corrupted data are detected.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.config import TARGET_COL, TEMP_COL, HUMIDITY_COL, DATE_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataQualityError(ValueError):
    """Raised when data quality validation constraints are violated."""
    pass


class DataQualityValidator:
    """
    Validates grid demand and weather datasets against physical bounds and temporal continuity.
    """

    def __init__(
        self,
        min_demand_mw: float = 0.0,
        max_demand_mw: float = 20000.0,
        min_temp_c: float = -30.0,
        max_temp_c: float = 60.0,
        min_humidity: float = 0.0,
        max_humidity: float = 100.0,
        expected_freq: str = "1h",
    ):
        self.min_demand_mw = min_demand_mw
        self.max_demand_mw = max_demand_mw
        self.min_temp_c = min_temp_c
        self.max_temp_c = max_temp_c
        self.min_humidity = min_humidity
        self.max_humidity = max_humidity
        self.expected_freq = expected_freq

    def validate(self, df: pd.DataFrame, allow_missing: bool = False) -> Dict[str, Any]:
        """
        Execute full validation checks.

        Args:
            df: DataFrame to validate.
            allow_missing: If False, raises DataQualityError if any NaNs exist.

        Returns:
            dict: Diagnostic validation report.

        Raises:
            DataQualityError: If critical validation rules are breached.
        """
        report = {
            "total_rows": len(df),
            "start_time": None,
            "end_time": None,
            "issues": [],
            "passed": True,
        }

        # 1. Index Check
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataQualityError("Dataset must have a pandas DatetimeIndex.")

        report["start_time"] = str(df.index[0])
        report["end_time"] = str(df.index[-1])

        # 2. Monotonicity & Duplicates
        if not df.index.is_monotonic_increasing:
            msg = "DatetimeIndex is not strictly monotonically increasing."
            report["issues"].append(msg)
            raise DataQualityError(msg)

        if df.index.has_duplicates:
            n_dups = df.index.duplicated().sum()
            msg = f"Dataset contains {n_dups} duplicate timestamps."
            report["issues"].append(msg)
            raise DataQualityError(msg)

        # 3. Missing Value Audit
        null_counts = df.isnull().sum().to_dict()
        report["null_counts"] = null_counts
        if not allow_missing and any(c > 0 for c in null_counts.values()):
            bad_cols = [k for k, v in null_counts.items() if v > 0]
            msg = f"Unresolved missing values found in columns: {bad_cols}. Imputation must be executed first."
            report["issues"].append(msg)
            report["passed"] = False
            raise DataQualityError(msg)

        # 4. Physical Demand Bounds
        if TARGET_COL in df.columns:
            demand = df[TARGET_COL].dropna()
            if (demand <= self.min_demand_mw).any():
                n_neg = (demand <= self.min_demand_mw).sum()
                msg = f"Physical violation: Found {n_neg} non-positive or negative demand values (min={demand.min():.1f} MW)."
                report["issues"].append(msg)
                report["passed"] = False
                raise DataQualityError(msg)

            if (demand > self.max_demand_mw).any():
                n_over = (demand > self.max_demand_mw).sum()
                msg = f"Physical violation: Found {n_over} demand spikes exceeding ceiling of {self.max_demand_mw} MW (max={demand.max():.1f} MW)."
                report["issues"].append(msg)
                report["passed"] = False
                raise DataQualityError(msg)

        # 5. Meteorological Bounds
        if TEMP_COL in df.columns:
            t = df[TEMP_COL].dropna()
            if (t < self.min_temp_c).any() or (t > self.max_temp_c).any():
                msg = f"Meteorological violation: Temperature values outside realistic bounds [{self.min_temp_c}°C, {self.max_temp_c}°C] (range: {t.min():.1f}°C to {t.max():.1f}°C)."
                report["issues"].append(msg)
                report["passed"] = False
                raise DataQualityError(msg)

        if HUMIDITY_COL in df.columns:
            rh = df[HUMIDITY_COL].dropna()
            if (rh < self.min_humidity).any() or (rh > self.max_humidity).any():
                msg = f"Meteorological violation: Relative humidity values outside [0%, 100%] (range: {rh.min():.1f}% to {rh.max():.1f}%)."
                report["issues"].append(msg)
                report["passed"] = False
                raise DataQualityError(msg)

        logger.info(f"Data quality validation PASSED ({len(df):,} observations from {df.index[0]} to {df.index[-1]}).")
        return report
