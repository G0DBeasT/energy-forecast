"""
Unified Feature Engineering Pipeline.

Provides high-dimensional, leakage-safe feature transformations for both:
1. Hourly time series (Next-Day 24h and Next-Week hourly views)
2. Daily aggregated time series (Next-Week and Next-Month 30d views)

Engineers temporal, calendar, holiday proximity, autoregressive lag,
rolling window statistics, and thermodynamic weather interaction features.
"""

from datetime import timedelta
from typing import List, Optional, Tuple
import holidays as hol
import numpy as np
import pandas as pd

from src.config import (
    BASE_TEMP_C,
    COUNTRY_HOLIDAY_CODE,
    DATA_PROC,
    HUMIDITY_COL,
    TARGET_COL,
    TEMP_COL,
)
from src.weather.client import WeatherService


class FeatureEngineer:
    """
    Production-grade feature engineer for hourly and daily energy grid forecasting.
    """

    def __init__(self, country: str = COUNTRY_HOLIDAY_CODE):
        self.country = country
        try:
            self.holiday_calendar = hol.country_holidays(self.country)
        except Exception:
            self.holiday_calendar = hol.country_holidays("US")

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add hour, day, week, month, quarter, year, and operational dispatch flags.
        """
        df = df.copy()
        idx = df.index

        if isinstance(idx, pd.DatetimeIndex):
            # Hour-level (only if hourly series)
            if hasattr(idx, "hour") and len(idx.hour.unique()) > 1:
                df["hour"] = idx.hour
                df["is_peak_hour"] = (((idx.hour >= 9) & (idx.hour <= 11)) | ((idx.hour >= 18) & (idx.hour <= 22))).astype(int)
                df["is_night_valley"] = ((idx.hour >= 1) & (idx.hour <= 5)).astype(int)

            # Day-level
            df["day_of_week"] = idx.dayofweek  # 0=Mon, 6=Sun
            df["day_of_month"] = idx.day
            df["day_of_year"] = idx.dayofyear
            df["week_of_year"] = idx.isocalendar().week.astype(int)

            # Month, Quarter, Year, Season
            df["month"] = idx.month
            df["quarter"] = idx.quarter
            df["year"] = idx.year

            # Season indicator: 1=Winter(Dec-Feb), 2=Spring(Mar-May), 3=Summer(Jun-Aug), 4=Autumn(Sep-Nov)
            df["season"] = idx.month % 12 // 3 + 1
            df["annual_position"] = idx.dayofyear / 365.25

        return df

    def add_calendar_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weekend, working day, holiday flags, and proximity distance features
        (days_until_holiday, days_since_holiday).
        """
        df = df.copy()
        dates = df.index.normalize()

        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        df["is_holiday"] = dates.isin(self.holiday_calendar).astype(int)
        df["is_pre_holiday"] = (dates + timedelta(days=1)).isin(self.holiday_calendar).astype(int)
        df["is_post_holiday"] = (dates - timedelta(days=1)).isin(self.holiday_calendar).astype(int)
        df["is_working_day"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

        # Proximity to nearest public holiday (up to 7 days)
        unique_dates = pd.Series(dates.unique())
        holiday_dates = pd.to_datetime(list(self.holiday_calendar.keys()))

        def get_holiday_distance(d):
            diffs = (holiday_dates - d).total_seconds() / 86400.0
            future_diffs = diffs[diffs >= 0]
            past_diffs = diffs[diffs <= 0]
            days_until = float(np.min(future_diffs)) if len(future_diffs) > 0 else 30.0
            days_since = float(np.abs(np.max(past_diffs))) if len(past_diffs) > 0 else 30.0
            return min(days_until, 30.0), min(days_since, 30.0)

        dist_map = {d: get_holiday_distance(d) for d in unique_dates}
        df["days_until_holiday"] = dates.map(lambda d: dist_map[d][0]).astype(float)
        df["days_since_holiday"] = dates.map(lambda d: dist_map[d][1]).astype(float)

        return df

    def add_fourier_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical temporal properties continuously using Sine/Cosine pairs.
        """
        df = df.copy()
        idx = df.index

        if isinstance(idx, pd.DatetimeIndex):
            if hasattr(idx, "hour") and len(idx.hour.unique()) > 1:
                df["sin_hour"] = np.sin(2 * np.pi * idx.hour / 24.0)
                df["cos_hour"] = np.cos(2 * np.pi * idx.hour / 24.0)

            df["sin_week"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
            df["cos_week"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

            df["sin_year"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
            df["cos_year"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

        return df

    def add_hourly_demand_lags(self, df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
        """
        Add hourly autoregressive demand lags:
        1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 168h (1 week), 336h (2 weeks), 720h (30 days), 8760h (1 year).
        """
        df = df.copy()
        if col not in df.columns:
            return df

        # Intra-day lags
        df["lag_1h"] = df[col].shift(1)
        df["lag_2h"] = df[col].shift(2)
        df["lag_3h"] = df[col].shift(3)
        df["lag_6h"] = df[col].shift(6)
        df["lag_12h"] = df[col].shift(12)

        # Daily lags
        df["lag_24h"] = df[col].shift(24)
        df["lag_48h"] = df[col].shift(48)
        df["lag_72h"] = df[col].shift(72)

        # Weekly & Monthly lags
        df["lag_168h"] = df[col].shift(168)
        df["lag_336h"] = df[col].shift(336)
        df["lag_720h"] = df[col].shift(720)

        # Annual lag if sufficient data exists
        if len(df) >= 8760 + 100:
            df["lag_8760h"] = df[col].shift(8760)

        return df

    def add_hourly_rolling_aggregations(self, df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
        """
        Compute leakage-safe rolling statistics shifted by 1 hour:
        - 24h window: mean, std, min, max
        - 7d (168h) window: mean, std, min, max
        - 30d (720h) window: mean, std, min, max
        """
        df = df.copy()
        if col not in df.columns:
            return df

        shifted = df[col].shift(1)

        # 24-hour window
        df["rolling_mean_24h"] = shifted.rolling(24).mean()
        df["rolling_std_24h"] = shifted.rolling(24).std()
        df["rolling_min_24h"] = shifted.rolling(24).min()
        df["rolling_max_24h"] = shifted.rolling(24).max()

        # 7-day (168h) window
        df["rolling_mean_168h"] = shifted.rolling(168).mean()
        df["rolling_std_168h"] = shifted.rolling(168).std()
        df["rolling_min_168h"] = shifted.rolling(168).min()
        df["rolling_max_168h"] = shifted.rolling(168).max()

        # 30-day (720h) window
        df["rolling_mean_720h"] = shifted.rolling(720).mean()
        df["rolling_std_720h"] = shifted.rolling(720).std()
        df["rolling_min_720h"] = shifted.rolling(720).min()
        df["rolling_max_720h"] = shifted.rolling(720).max()

        # Ratio of previous step to rolling mean
        df["demand_ratio_to_24h_mean"] = shifted / (df["rolling_mean_24h"] + 1e-5)

        return df

    def add_daily_demand_lags(self, df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
        """
        Add daily autoregressive demand lags for daily models:
        lag_1d, lag_2d, lag_3d, lag_7d (same day last week), lag_14d, lag_30d, lag_365d.
        """
        df = df.copy()
        if col not in df.columns:
            return df

        df["lag_1d"] = df[col].shift(1)
        df["lag_2d"] = df[col].shift(2)
        df["lag_3d"] = df[col].shift(3)
        df["lag_7d"] = df[col].shift(7)
        df["lag_14d"] = df[col].shift(14)
        df["lag_30d"] = df[col].shift(30)

        if len(df) >= 365 + 30:
            df["lag_365d"] = df[col].shift(365)

        return df

    def add_daily_rolling_aggregations(self, df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
        """
        Compute leakage-safe rolling statistics for daily models shifted by 1 day:
        - 7d rolling: mean, std, min, max
        - 14d rolling: mean, std
        - 30d rolling: mean, std, min, max
        """
        df = df.copy()
        if col not in df.columns:
            return df

        shifted = df[col].shift(1)

        # 7-day rolling
        df["rolling_mean_7d"] = shifted.rolling(7).mean()
        df["rolling_std_7d"] = shifted.rolling(7).std()
        df["rolling_min_7d"] = shifted.rolling(7).min()
        df["rolling_max_7d"] = shifted.rolling(7).max()

        # 14-day rolling
        df["rolling_mean_14d"] = shifted.rolling(14).mean()
        df["rolling_std_14d"] = shifted.rolling(14).std()

        # 30-day rolling
        df["rolling_mean_30d"] = shifted.rolling(30).mean()
        df["rolling_std_30d"] = shifted.rolling(30).std()
        df["rolling_min_30d"] = shifted.rolling(30).min()
        df["rolling_max_30d"] = shifted.rolling(30).max()

        df["demand_ratio_to_7d_mean"] = shifted / (df["rolling_mean_7d"] + 1e-5)

        return df

    def add_weather_features(self, df: pd.DataFrame, is_daily: bool = False) -> pd.DataFrame:
        """
        Add meteorological features, degree days, and temperature rolling statistics.
        """
        df = df.copy()
        df = WeatherService.calculate_derived_weather(df)

        if TEMP_COL in df.columns:
            t = df[TEMP_COL]
            t_shifted = t.shift(1)

            if is_daily:
                df["temp_lag_1d"] = t.shift(1)
                df["temp_lag_7d"] = t.shift(7)
                df["temp_rolling_mean_7d"] = t_shifted.rolling(7).mean()
                df["temp_rolling_mean_30d"] = t_shifted.rolling(30).mean()
            else:
                df["temp_lag_24h"] = t.shift(24)
                df["temp_rolling_mean_24h"] = t_shifted.rolling(24).mean()
                df["temp_rolling_mean_7d"] = t_shifted.rolling(168).mean()

        if HUMIDITY_COL in df.columns:
            rh_shifted = df[HUMIDITY_COL].shift(1)
            window = 7 if is_daily else 24
            df["humidity_rolling_mean"] = rh_shifted.rolling(window).mean()

        return df

    def transform_hourly(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Execute full hourly feature transformation pipeline."""
        df_feat = df.copy()
        df_feat = self.add_temporal_features(df_feat)
        df_feat = self.add_calendar_holiday_features(df_feat)
        df_feat = self.add_fourier_cyclical_features(df_feat)
        df_feat = self.add_hourly_demand_lags(df_feat)
        df_feat = self.add_hourly_rolling_aggregations(df_feat)
        df_feat = self.add_weather_features(df_feat, is_daily=False)

        if drop_na:
            if "lag_8760h" in df_feat.columns and df_feat["lag_8760h"].isnull().sum() > len(df_feat) * 0.5:
                df_feat = df_feat.drop(columns=["lag_8760h"])
            df_feat = df_feat.dropna()

        return df_feat

    def transform_daily(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Execute full daily feature transformation pipeline."""
        df_feat = df.copy()
        df_feat = self.add_temporal_features(df_feat)
        df_feat = self.add_calendar_holiday_features(df_feat)
        df_feat = self.add_fourier_cyclical_features(df_feat)
        df_feat = self.add_daily_demand_lags(df_feat)
        df_feat = self.add_daily_rolling_aggregations(df_feat)
        df_feat = self.add_weather_features(df_feat, is_daily=True)

        if drop_na:
            if "lag_365d" in df_feat.columns and df_feat["lag_365d"].isnull().sum() > len(df_feat) * 0.5:
                df_feat = df_feat.drop(columns=["lag_365d"])
            df_feat = df_feat.dropna()

        return df_feat

    def extract_step_features(
        self,
        full_df: pd.DataFrame,
        current_dt: pd.Timestamp,
        feature_cols: List[str],
        is_daily: bool = False,
    ) -> pd.DataFrame:
        """
        Incremental feature computation for a single future timestep in recursive forecasting.
        """
        if is_daily:
            window_start = current_dt - pd.Timedelta(days=400)
            recent_df = full_df.loc[window_start:current_dt].copy()
            transformed_df = self.transform_daily(recent_df, drop_na=False)
        else:
            window_start = current_dt - pd.Timedelta(hours=800)
            recent_df = full_df.loc[window_start:current_dt].copy()
            transformed_df = self.transform_hourly(recent_df, drop_na=False)

        return transformed_df.loc[[current_dt], [c for c in feature_cols if c in transformed_df.columns]]


def build_and_save_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save both hourly and daily feature parquet matrices."""
    parquet_path = DATA_PROC / "hourly_demand.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cleaned dataset {parquet_path} not found. Run data pipeline first.")

    df_hourly = pd.read_parquet(parquet_path)
    engineer = FeatureEngineer()

    # 1. Hourly Features
    df_feat_hourly = engineer.transform_hourly(df_hourly, drop_na=True)
    out_hourly = DATA_PROC / "features.parquet"
    df_feat_hourly.to_parquet(out_hourly)
    print(f"Hourly feature matrix saved to {out_hourly} ({df_feat_hourly.shape[0]:,} rows x {df_feat_hourly.shape[1]} cols)")

    # 2. Daily Aggregated Features
    # Aggregate hourly demand into daily mean demand (MW) and daily weather averages
    daily_agg = pd.DataFrame({
        TARGET_COL: df_hourly[TARGET_COL].resample("1D").mean(),
        TEMP_COL: df_hourly[TEMP_COL].resample("1D").mean(),
        HUMIDITY_COL: df_hourly[HUMIDITY_COL].resample("1D").mean(),
    })
    df_feat_daily = engineer.transform_daily(daily_agg, drop_na=True)
    out_daily = DATA_PROC / "features_daily.parquet"
    df_feat_daily.to_parquet(out_daily)
    print(f"Daily feature matrix saved to {out_daily} ({df_feat_daily.shape[0]:,} rows x {df_feat_daily.shape[1]} cols)")

    return df_feat_hourly, df_feat_daily


if __name__ == "__main__":
    build_and_save_features()
