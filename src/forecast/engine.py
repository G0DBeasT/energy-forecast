"""
Multi-Horizon Energy Demand Forecasting Engine.

Generates three explicit, horizon-appropriate forecasting products:
1. NEXT-DAY FORECAST:
   - Horizon: Next 24 Hours
   - Resolution: Hourly
   - Model: Champion Hourly LightGBM
   - Variables: datetime, forecast_mw, lower_bound_90, upper_bound_90, temp_c, relative_humidity, day_of_week, hour

2. NEXT-WEEK FORECAST:
   - Horizon: Next 7 Days (168 Hours)
   - Resolution: Daily (Aggregated) + Hourly Profile
   - Model: Hourly LightGBM aggregated to daily peak/mean + Day-level analytics
   - Highlights: Highest demand day, Lowest demand day, Largest day-over-day shift

3. NEXT-MONTH FORECAST:
   - Horizon: Next 30 Days (720 Hours)
   - Resolution: Daily
   - Strategy: Dedicated Daily LightGBM Model (avoids 720-step hourly error compounding!)
   - Weather: Open-Meteo live NWP forecast (days 1-14) + Seasonal climatology (days 15-30)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import pickle
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROC,
    DATE_COL,
    HORIZON_DAY,
    HORIZON_MONTH,
    HORIZON_WEEK,
    HUMIDITY_COL,
    TARGET_COL,
    TEMP_COL,
    WEATHER_SOURCE_COL,
)
from src.features.pipeline import FeatureEngineer
from src.forecast.uncertainty import DynamicUncertaintyEstimator
from src.models.registry import ModelRegistry
from src.weather.client import WeatherService


class MultiHorizonForecastEngine:
    """Production Multi-Horizon Forecaster."""

    def __init__(
        self,
        hourly_model_path: Optional[Path] = None,
        daily_model_path: Optional[Path] = None,
    ):
        # 1. Load Hourly Model
        h_path = hourly_model_path or (DATA_PROC / "best_model.pkl")
        self.hourly_model, self.hourly_metadata = ModelRegistry.load_model(h_path)
        self.hourly_features = self.hourly_metadata.get("feature_columns", [])

        # 2. Load Daily Model (for Next-Month)
        d_path = daily_model_path or (DATA_PROC / "best_model_daily.pkl")
        if d_path.exists():
            with open(d_path, "rb") as f:
                self.daily_model = pickle.load(f)
            meta_d_path = DATA_PROC / "model_metadata_daily.json"
            self.daily_features = []
            if meta_d_path.exists():
                with open(meta_d_path, "r") as f:
                    self.daily_features = json.load(f).get("feature_columns", [])
        else:
            self.daily_model = None
            self.daily_features = []

        self.feature_engineer = FeatureEngineer()
        self.weather_service = WeatherService()
        self.uncertainty_hourly = DynamicUncertaintyEstimator(base_sigma_mw=130.0, lead_time_growth_alpha=0.35)
        self.uncertainty_daily = DynamicUncertaintyEstimator(base_sigma_mw=110.0, lead_time_growth_alpha=0.40)

    def forecast_next_day(self, temp_delta: float = 0.0) -> pd.DataFrame:
        """
        Generate 24-Hour Hourly Next-Day Forecast.
        """
        history_df = pd.read_parquet(DATA_PROC / "hourly_demand.parquet").copy()
        last_dt = history_df.index[-1]

        future_dates = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=HORIZON_DAY, freq="1h")
        future_weather = self.weather_service.get_weather_for_horizon(future_dates)
        if temp_delta != 0.0:
            future_weather[TEMP_COL] += temp_delta

        future_df = pd.DataFrame(
            {
                TARGET_COL: np.nan,
                TEMP_COL: future_weather[TEMP_COL].values,
                HUMIDITY_COL: future_weather[HUMIDITY_COL].values,
                WEATHER_SOURCE_COL: future_weather[WEATHER_SOURCE_COL].values,
            },
            index=future_dates,
        )

        working_df = pd.concat([history_df.tail(8800), future_df])
        predictions = []

        for dt in future_dates:
            feat_row = self.feature_engineer.extract_step_features(working_df, dt, self.hourly_features, is_daily=False)
            for c in self.hourly_features:
                if c not in feat_row.columns:
                    feat_row[c] = 0.0
            pred = float(self.hourly_model.predict(feat_row[self.hourly_features])[0])
            pred = max(pred, 0.0)
            working_df.loc[dt, TARGET_COL] = pred
            predictions.append(pred)

        pred_series = pd.Series(predictions, index=future_dates, name="forecast_mw")
        df_out = self.uncertainty_hourly.attach_prediction_intervals(pred_series)
        df_out[DATE_COL] = future_dates
        df_out["hour"] = future_dates.hour
        df_out["day_of_week"] = future_dates.dayofweek
        df_out[TEMP_COL] = future_weather[TEMP_COL].values
        df_out[HUMIDITY_COL] = future_weather[HUMIDITY_COL].values
        df_out[WEATHER_SOURCE_COL] = future_weather[WEATHER_SOURCE_COL].values

        return df_out

    def forecast_next_week(self, temp_delta: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate 7-Day (168h) Next-Week Forecast.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (Hourly Profile DataFrame, Daily Aggregated Summary DataFrame)
        """
        history_df = pd.read_parquet(DATA_PROC / "hourly_demand.parquet").copy()
        last_dt = history_df.index[-1]

        future_dates = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=HORIZON_WEEK, freq="1h")
        future_weather = self.weather_service.get_weather_for_horizon(future_dates)
        if temp_delta != 0.0:
            future_weather[TEMP_COL] += temp_delta

        future_df = pd.DataFrame(
            {
                TARGET_COL: np.nan,
                TEMP_COL: future_weather[TEMP_COL].values,
                HUMIDITY_COL: future_weather[HUMIDITY_COL].values,
                WEATHER_SOURCE_COL: future_weather[WEATHER_SOURCE_COL].values,
            },
            index=future_dates,
        )

        working_df = pd.concat([history_df.tail(8800), future_df])
        predictions = []

        for dt in future_dates:
            feat_row = self.feature_engineer.extract_step_features(working_df, dt, self.hourly_features, is_daily=False)
            for c in self.hourly_features:
                if c not in feat_row.columns:
                    feat_row[c] = 0.0
            pred = float(self.hourly_model.predict(feat_row[self.hourly_features])[0])
            pred = max(pred, 0.0)
            working_df.loc[dt, TARGET_COL] = pred
            predictions.append(pred)

        pred_series = pd.Series(predictions, index=future_dates, name="forecast_mw")
        df_hourly = self.uncertainty_hourly.attach_prediction_intervals(pred_series)
        df_hourly[DATE_COL] = future_dates
        df_hourly[TEMP_COL] = future_weather[TEMP_COL].values
        df_hourly[HUMIDITY_COL] = future_weather[HUMIDITY_COL].values
        df_hourly[WEATHER_SOURCE_COL] = future_weather[WEATHER_SOURCE_COL].values

        # Daily aggregated summary
        daily_summary = df_hourly.resample("1D", on=DATE_COL).agg({
            "forecast_mw": ["mean", "max", "min", lambda s: s.sum() / 1000.0],  # Mean MW, Peak MW, Min MW, Total GWh
            TEMP_COL: "mean",
            HUMIDITY_COL: "mean",
            "lower_bound_90": "mean",
            "upper_bound_90": "mean",
        })
        daily_summary.columns = ["forecast_mean_mw", "peak_mw", "min_mw", "total_gwh", "temp_c", "relative_humidity", "lower_bound_mw", "upper_bound_mw"]
        daily_summary = daily_summary.reset_index()
        daily_summary["day_of_week"] = daily_summary[DATE_COL].dt.day_name()
        daily_summary["is_weekend"] = (daily_summary[DATE_COL].dt.dayofweek >= 5).astype(int)
        dates_norm = daily_summary[DATE_COL].dt.normalize()
        daily_summary["is_holiday"] = dates_norm.isin(self.feature_engineer.holiday_calendar).astype(int)
        daily_summary["day_over_day_change_mw"] = daily_summary["forecast_mean_mw"].diff().fillna(0.0).round(1)

        return df_hourly, daily_summary

    def forecast_next_month(self, temp_delta: float = 0.0) -> pd.DataFrame:
        """
        Generate 30-Day Next-Month Forecast using Dedicated Daily LightGBM Model.
        """
        history_df = pd.read_parquet(DATA_PROC / "hourly_demand.parquet").copy()
        daily_history = pd.DataFrame({
            TARGET_COL: history_df[TARGET_COL].resample("1D").mean(),
            TEMP_COL: history_df[TEMP_COL].resample("1D").mean(),
            HUMIDITY_COL: history_df[HUMIDITY_COL].resample("1D").mean(),
        })

        last_date = daily_history.index[-1]
        future_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="1D")

        # Future weather over 30 days
        future_hourly_dates = pd.date_range(start=future_days[0], periods=30 * 24, freq="1h")
        future_weather_hourly = self.weather_service.get_weather_for_horizon(future_hourly_dates)
        if temp_delta != 0.0:
            future_weather_hourly[TEMP_COL] += temp_delta

        future_weather_daily = pd.DataFrame({
            TEMP_COL: future_weather_hourly[TEMP_COL].resample("1D").mean(),
            HUMIDITY_COL: future_weather_hourly[HUMIDITY_COL].resample("1D").mean(),
            WEATHER_SOURCE_COL: future_weather_hourly[WEATHER_SOURCE_COL].resample("1D").first(),
        }, index=future_days)

        future_daily_container = pd.DataFrame({
            TARGET_COL: np.nan,
            TEMP_COL: future_weather_daily[TEMP_COL].values,
            HUMIDITY_COL: future_weather_daily[HUMIDITY_COL].values,
            WEATHER_SOURCE_COL: future_weather_daily[WEATHER_SOURCE_COL].values,
        }, index=future_days)

        working_daily = pd.concat([daily_history.tail(400), future_daily_container])
        predictions = []

        for d in future_days:
            feat_row = self.feature_engineer.extract_step_features(working_daily, d, self.daily_features, is_daily=True)
            for c in self.daily_features:
                if c not in feat_row.columns:
                    feat_row[c] = 0.0
            pred = float(self.daily_model.predict(feat_row[self.daily_features])[0]) if self.daily_model is not None else 5000.0
            pred = max(pred, 0.0)
            working_daily.loc[d, TARGET_COL] = pred
            predictions.append(pred)

        pred_series = pd.Series(predictions, index=future_days, name="forecast_mw")
        df_month = self.uncertainty_daily.attach_prediction_intervals(pred_series)
        df_month[DATE_COL] = future_days
        df_month["day_of_week"] = future_days.day_name()
        df_month["is_weekend"] = (future_days.dayofweek >= 5).astype(int)
        df_month["is_holiday"] = future_days.normalize().isin(self.feature_engineer.holiday_calendar).astype(int)
        df_month[TEMP_COL] = future_weather_daily[TEMP_COL].values
        df_month[HUMIDITY_COL] = future_weather_daily[HUMIDITY_COL].values
        df_month[WEATHER_SOURCE_COL] = future_weather_daily[WEATHER_SOURCE_COL].values
        df_month["total_daily_gwh"] = np.round(df_month["forecast_mw"] * 24.0 / 1000.0, 2)

        return df_month

    def generate_all_horizons(self) -> Dict[str, pd.DataFrame]:
        """Generate and serialize all 3 forecasting products."""
        print("Generating Multi-Horizon Forecasts...")

        # 1. Next Day
        print(" -> Generating Next-Day Forecast (24 Hours)...")
        df_day = self.forecast_next_day()
        df_day.to_csv(DATA_PROC / "forecast_next_day.csv", index=False)

        # 2. Next Week
        print(" -> Generating Next-Week Forecast (7 Days / 168 Hours)...")
        df_week_hourly, df_week_daily = self.forecast_next_week()
        df_week_hourly.to_csv(DATA_PROC / "forecast_next_week.csv", index=False)
        df_week_daily.to_csv(DATA_PROC / "forecast_next_week_daily.csv", index=False)
        df_week_hourly.to_csv(DATA_PROC / "forecast.csv", index=False)  # Legacy compatibility

        # 3. Next Month
        print(" -> Generating Next-Month Forecast (30 Days)...")
        df_month = self.forecast_next_month()
        df_month.to_csv(DATA_PROC / "forecast_next_month.csv", index=False)

        print(f"All 3 multi-horizon forecast files saved to {DATA_PROC}")
        return {
            "next_day": df_day,
            "next_week_hourly": df_week_hourly,
            "next_week_daily": df_week_daily,
            "next_month": df_month,
        }


def run_forecast_pipeline():
    engine = MultiHorizonForecastEngine()
    results = engine.generate_all_horizons()
    print("\n--- NEXT-DAY PREVIEW ---")
    print(results["next_day"].head(6)[["datetime", "forecast_mw", "lower_bound_90", "upper_bound_90", "weather_source"]])
    print("\n--- NEXT-WEEK DAILY SUMMARY ---")
    print(results["next_week_daily"][["datetime", "day_of_week", "forecast_mean_mw", "peak_mw", "total_gwh"]])
    print("\n--- NEXT-MONTH PREVIEW (First 7 Days) ---")
    print(results["next_month"].head(7)[["datetime", "day_of_week", "forecast_mw", "total_daily_gwh", "weather_source"]])


if __name__ == "__main__":
    run_forecast_pipeline()
