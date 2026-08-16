"""
Climatological Weather Model & Seasonal Fallback.

Provides physically-grounded historical seasonal average weather profiles
for extended forecasting horizons (>14 days) and offline network fallback.
"""

from typing import Optional
import numpy as np
import pandas as pd
from src.config import DATE_COL, TEMP_COL, HUMIDITY_COL, BASE_TEMP_C


class ClimatologyWeatherModel:
    """
    Parametric meteorological climatology generator for regional grids.

    Models:
    - Annual solar insulation cycle (winter trough to summer peak)
    - Diurnal solar heating cycle (coolest at sunrise ~05:00, peak ~15:00)
    - Monsoon season humidity surge & evaporative cooling
    - ASHRAE degree-day thresholds
    """

    def __init__(
        self,
        annual_mean_temp: float = 26.5,
        annual_temp_amplitude: float = 11.5,
        daily_temp_amplitude: float = 4.5,
        summer_peak_day: int = 150,  # ~Late May / Early June peak in Northern hemisphere
        monsoon_start_day: int = 180, # July
        monsoon_end_day: int = 270,   # September
    ):
        self.annual_mean_temp = annual_mean_temp
        self.annual_temp_amplitude = annual_temp_amplitude
        self.daily_temp_amplitude = daily_temp_amplitude
        self.summer_peak_day = summer_peak_day
        self.monsoon_start_day = monsoon_start_day
        self.monsoon_end_day = monsoon_end_day

    def generate_profile(
        self,
        date_range: pd.DatetimeIndex,
        add_noise: bool = True,
        seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Generate continuous hourly temperature and humidity profile for given dates.

        Args:
            date_range: Hourly DatetimeIndex.
            add_noise: Whether to add slight stochastic weather turbulence.
            seed: Random seed.

        Returns:
            pd.DataFrame with DATE_COL, TEMP_COL, HUMIDITY_COL.
        """
        if seed is not None:
            np.random.seed(seed)

        day_of_year = date_range.dayofyear.values
        hour = date_range.hour.values
        n_obs = len(date_range)

        # Annual temperature cycle
        temp_annual = self.annual_mean_temp + self.annual_temp_amplitude * np.sin(
            2 * np.pi * (day_of_year - (self.summer_peak_day - 91)) / 365.25
        )

        # Diurnal temperature cycle (min ~06:00, max ~15:00)
        temp_daily = self.daily_temp_amplitude * np.sin(2 * np.pi * (hour - 9) / 24.0)

        temp_noise = np.random.normal(0, 0.8, n_obs) if add_noise else np.zeros(n_obs)
        temp = temp_annual + temp_daily + temp_noise

        # Humidity profile: inverse with temperature + monsoon effect
        is_monsoon = (day_of_year >= self.monsoon_start_day) & (day_of_year <= self.monsoon_end_day)
        monsoon_boost = np.where(is_monsoon, 22.0, 0.0)
        hum_noise = np.random.normal(0, 2.0, n_obs) if add_noise else np.zeros(n_obs)

        humidity = 82.0 - 1.25 * temp + monsoon_boost + hum_noise
        humidity = np.clip(humidity, 15.0, 98.0)

        return pd.DataFrame({
            DATE_COL: date_range,
            TEMP_COL: np.round(temp, 2),
            HUMIDITY_COL: np.round(humidity, 2),
        })
