"""
Data Cleaning & Preprocessing Pipeline.

Performs robust temporal alignment, frequency standardization (1H),
missing value imputation, and sensor anomaly filtering.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from src.config import DATA_PROC, TARGET_COL, TEMP_COL, HUMIDITY_COL, FREQ


class DataCleaner:
    """
    Production data cleaner for time-series energy and weather streams.

    Cleaning Rules:
    1. Resample to continuous hourly index ('1h') with mean aggregation.
    2. Outlier Clipping: Clip extreme sensor spikes/troughs to [0.1th, 99.9th] quantile bounds.
    3. Missing Imputation:
       - Short gaps (<= 4h): Forward fill (preserves autoregressive continuity)
       - Medium gaps: Linear interpolation
       - Any trailing edge gaps: Backward fill / historical mean
    """

    def __init__(
        self,
        freq: str = FREQ,
        clip_lower_quantile: float = 0.001,
        clip_upper_quantile: float = 0.999,
        ffill_limit: int = 4,
    ):
        self.freq = freq
        self.clip_lower_quantile = clip_lower_quantile
        self.clip_upper_quantile = clip_upper_quantile
        self.ffill_limit = ffill_limit

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataframe and return clean continuous 1h series.

        Args:
            df: Raw DataFrame with DatetimeIndex.

        Returns:
            pd.DataFrame: Cleaned DataFrame with complete hourly sequence and 0 NaNs.
        """
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input dataframe must have a pandas DatetimeIndex.")

        # 1. Resample to strict frequency
        df = df.resample(self.freq).mean()

        # 2. Outlier clipping on demand using robust IQR bounds
        if TARGET_COL in df.columns:
            s = df[TARGET_COL].dropna()
            if len(s) > 10:
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                iqr = q75 - q25
                lo = max(0.0, q25 - 3.5 * iqr)
                hi = q75 + 3.5 * iqr
                df[TARGET_COL] = df[TARGET_COL].clip(lo, hi)
            else:
                lo = df[TARGET_COL].quantile(self.clip_lower_quantile)
                hi = df[TARGET_COL].quantile(self.clip_upper_quantile)
                df[TARGET_COL] = df[TARGET_COL].clip(lo, hi)

            # Impute target
            df[TARGET_COL] = (
                df[TARGET_COL]
                .ffill(limit=self.ffill_limit)
                .interpolate(method="linear")
                .bfill()
            )

        # 3. Impute weather features if present
        for col in [TEMP_COL, HUMIDITY_COL]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .ffill(limit=self.ffill_limit)
                    .interpolate(method="linear")
                    .bfill()
                )

        # Ensure no residual NaNs remain
        if df.isnull().any().any():
            df = df.bfill().ffill()

        return df

    def save_processed(self, df: pd.DataFrame, filename: str = "hourly_demand.parquet") -> Path:
        """Save cleaned dataset to parquet."""
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        out_path = DATA_PROC / filename
        df.to_parquet(out_path)
        return out_path
