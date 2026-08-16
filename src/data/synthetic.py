"""
Synthetic Grid Load Data Generator.

Generates realistic, physically-sound hourly energy grid demand (MW) modeled
from meteorological conditions, diurnal human activity patterns, annual trends,
and industrial calendar schedules.

IMPORTANT PROVENANCE NOTE:
This data source is SIMULATED / SYNTHETIC for development, testing, and benchmarking.
It does not represent uncalibrated empirical substation SCADA recordings.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.config import DATA_RAW, DATE_COL, TARGET_COL, TEMP_COL, HUMIDITY_COL, LATITUDE, LONGITUDE, RANDOM_SEED
from src.data.base import BaseDataLoader
from src.weather.client import WeatherService


class SyntheticGridLoader(BaseDataLoader):
    """
    Physically modeled synthetic power grid dataset generator.

    Combines meteorological observations (from Open-Meteo or physical fallback)
    with grid load physics:
    - Base load floor (MW)
    - Macroeconomic demand growth trend
    - Non-linear Cooling Degree Day (CDD) response (summer air conditioning)
    - Non-linear Heating Degree Day (HDD) response (winter heating)
    - Diurnal double-peak human activity pattern (morning & evening)
    - Weekly commercial/industrial load reduction on weekends
    - Gaussian stochastic grid fluctuation noise
    """

    def __init__(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2024-06-30",
        base_load: float = 4500.0,
        trend_growth: float = 800.0,
        noise_std: float = 120.0,
        seed: int = RANDOM_SEED,
        lat: float = LATITUDE,
        lon: float = LONGITUDE,
        inject_missing_rate: float = 0.002,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.base_load = base_load
        self.trend_growth = trend_growth
        self.noise_std = noise_std
        self.seed = seed
        self.lat = lat
        self.lon = lon
        self.inject_missing_rate = inject_missing_rate
        self.weather_service = WeatherService()

    @property
    def data_provenance(self) -> str:
        return "synthetic_simulated"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": self.data_provenance,
            "description": "Physically simulated hourly power grid demand combined with Open-Meteo weather.",
            "start_date": self.start_date,
            "end_date": self.end_date,
            "base_load_mw": self.base_load,
            "location_lat_lon": (self.lat, self.lon),
            "seed": self.seed,
            "provenance_note": (
                "Synthetic development dataset. Weather is sourced from Open-Meteo archive; "
                "grid demand is computed using non-linear thermodynamic load response functions."
            ),
        }

    def load(self, force_regenerate: bool = False) -> pd.DataFrame:
        """
        Load synthetic grid dataset. If cached on disk, loads from CSV; otherwise generates it.
        """
        raw_demand_path = DATA_RAW / "raw_hourly_grid_demand.csv"

        if raw_demand_path.exists() and not force_regenerate:
            df = pd.read_csv(raw_demand_path)
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df = df.set_index(DATE_COL).sort_index()
            self.validate_schema(df)
            return df

        df = self.generate()
        raw_demand_path.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(raw_demand_path, index=False)
        return df

    def generate(self) -> pd.DataFrame:
        """Generate full synthetic time series dataset."""
        np.random.seed(self.seed)

        # 1. Fetch historical weather
        df_weather = self.weather_service.fetch_historical(
            start_date=self.start_date,
            end_date=self.end_date,
            lat=self.lat,
            lon=self.lon,
        )

        dates = df_weather[DATE_COL]
        n_obs = len(dates)

        # 2. Demand components
        # Trend
        trend = np.linspace(0, self.trend_growth, n_obs)

        # Meteorological thermal demand response
        temp = df_weather[TEMP_COL].values
        cooling_effect = np.maximum(temp - 22.0, 0.0) ** 1.35 * 65.0
        heating_effect = np.maximum(16.0 - temp, 0.0) ** 1.2 * 35.0

        # Diurnal pattern (dual peak: mid-afternoon cooling + evening lighting/appliances)
        hour = dates.dt.hour.values
        daily_pattern = 300.0 * np.sin(2 * np.pi * (hour - 6) / 24.0) + 400.0 * np.sin(4 * np.pi * (hour - 12) / 24.0)

        # Weekly pattern (weekend commercial/industrial dip)
        day_of_week = dates.dt.dayofweek.values
        weekend_effect = np.where(day_of_week >= 5, -500.0, 0.0)

        # Stochastic noise
        noise = np.random.normal(0, self.noise_std, n_obs)

        # Total demand
        demand = self.base_load + trend + cooling_effect + heating_effect + daily_pattern + weekend_effect + noise

        df_grid = pd.DataFrame({
            DATE_COL: dates,
            TARGET_COL: np.round(demand, 2),
            TEMP_COL: df_weather[TEMP_COL].values,
            HUMIDITY_COL: df_weather[HUMIDITY_COL].values,
        })

        # Inject realistic sensor missingness / artifacts for cleaner to process
        if self.inject_missing_rate > 0:
            nan_mask = np.random.rand(len(df_grid)) < self.inject_missing_rate
            df_grid.loc[nan_mask, TARGET_COL] = np.nan

        df_grid[DATE_COL] = pd.to_datetime(df_grid[DATE_COL])
        df_grid = df_grid.set_index(DATE_COL).sort_index()
        self.validate_schema(df_grid)
        return df_grid
