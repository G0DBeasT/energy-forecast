"""
Weather Service & Open-Meteo Integration Client.

Handles:
1. Historical reanalysis weather fetching (Open-Meteo archive API)
2. Live numerical weather prediction (NWP) forecasting (Open-Meteo forecast API, up to 16 days)
3. Climatological seasonal blending for long horizons (16 to 30 days)
4. Derived meteorological features: CDD, HDD, Heat Index, and weather source tags.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd
import requests

from src.config import (
    DATE_COL,
    TEMP_COL,
    HUMIDITY_COL,
    WEATHER_SOURCE_COL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    BASE_TEMP_C,
)
from src.weather.climatology import ClimatologyWeatherModel

logger = logging.getLogger(__name__)


class WeatherService:
    """Production meteorological client with automatic NWP forecast, climatology, and fallback."""

    def __init__(
        self,
        lat: float = LATITUDE,
        lon: float = LONGITUDE,
        timezone: str = TIMEZONE,
        timeout: int = 15,
    ):
        self.lat = lat
        self.lon = lon
        self.timezone = timezone
        self.timeout = timeout
        self.climatology = ClimatologyWeatherModel()

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly historical temperature and relative humidity from Open-Meteo Archive API.
        Falls back to parametric climatology if offline.
        """
        lat = lat or self.lat
        lon = lon or self.lon
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m"
            f"&timezone={self.timezone.replace('/', '%2F')}"
        )

        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            df_weather = pd.DataFrame({
                DATE_COL: pd.to_datetime(data["hourly"]["time"]),
                TEMP_COL: data["hourly"]["temperature_2m"],
                HUMIDITY_COL: data["hourly"]["relative_humidity_2m"],
                WEATHER_SOURCE_COL: "historical_reanalysis",
            })
            return df_weather
        except Exception as e:
            logger.warning(f"Open-Meteo historical archive API unavailable ({e}). Using climatological fallback.")
            date_range = pd.date_range(start=start_date, end=end_date + " 23:00:00", freq="1h")
            df_weather = self.climatology.generate_profile(date_range, add_noise=True)
            df_weather[WEATHER_SOURCE_COL] = "climatological_fallback"
            return df_weather

    def fetch_live_forecast(
        self,
        forecast_days: int = 16,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Fetch up to 16 days of hourly numerical weather predictions from Open-Meteo Forecast API.
        """
        lat = lat or self.lat
        lon = lon or self.lon
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m"
            f"&forecast_days={min(forecast_days, 16)}"
            f"&timezone={self.timezone.replace('/', '%2F')}"
        )

        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            df_forecast = pd.DataFrame({
                DATE_COL: pd.to_datetime(data["hourly"]["time"]),
                TEMP_COL: data["hourly"]["temperature_2m"],
                HUMIDITY_COL: data["hourly"]["relative_humidity_2m"],
                WEATHER_SOURCE_COL: "nwp_forecast",
            })
            return df_forecast
        except Exception as e:
            logger.warning(f"Open-Meteo live forecast API unavailable ({e}). Using climatological model.")
            start_dt = pd.Timestamp.now().floor("h")
            date_range = pd.date_range(start=start_dt, periods=forecast_days * 24, freq="1h")
            df_forecast = self.climatology.generate_profile(date_range, add_noise=True)
            df_forecast[WEATHER_SOURCE_COL] = "climatological_fallback"
            return df_forecast

    def get_weather_for_horizon(
        self,
        future_dates: pd.DatetimeIndex,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate complete weather stream for future forecasting horizon:
        - Uses live NWP weather forecast for days 1 to 14.
        - Uses climatological seasonal profile for extended days (15 to 30).
        - Explicitly tags each row with weather source.
        """
        total_hours = len(future_dates)
        days_needed = int(np.ceil(total_hours / 24))

        # 1. Fetch live forecast up to 14 days
        live_df = self.fetch_live_forecast(forecast_days=min(days_needed, 14), lat=lat, lon=lon)
        live_df = live_df.set_index(DATE_COL)

        # 2. Build target container
        weather_rows = []
        for dt in future_dates:
            if dt in live_df.index and not np.isnan(live_df.loc[dt, TEMP_COL]):
                weather_rows.append({
                    DATE_COL: dt,
                    TEMP_COL: float(live_df.loc[dt, TEMP_COL]),
                    HUMIDITY_COL: float(live_df.loc[dt, HUMIDITY_COL]),
                    WEATHER_SOURCE_COL: "nwp_forecast",
                })
            else:
                # Climatology for dates beyond live forecast horizon
                single_clim = self.climatology.generate_profile(pd.DatetimeIndex([dt]), add_noise=True)
                weather_rows.append({
                    DATE_COL: dt,
                    TEMP_COL: float(single_clim[TEMP_COL].iloc[0]),
                    HUMIDITY_COL: float(single_clim[HUMIDITY_COL].iloc[0]),
                    WEATHER_SOURCE_COL: "climatology",
                })

        df_out = pd.DataFrame(weather_rows).set_index(DATE_COL)
        return df_out

    @staticmethod
    def calculate_derived_weather(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute meteorological derived indicators:
        - Cooling Degree Days (CDD): max(temp - base_temp, 0)
        - Heating Degree Days (HDD): max(base_temp - temp, 0)
        - Apparent Heat Index (°C) via Rothfusz regression
        """
        df = df.copy()
        if TEMP_COL in df.columns:
            t = df[TEMP_COL]
            df["cdd"] = np.maximum(t - BASE_TEMP_C, 0.0)
            df["hdd"] = np.maximum(BASE_TEMP_C - t, 0.0)
            df["cdd_sq"] = df["cdd"] ** 2

        if TEMP_COL in df.columns and HUMIDITY_COL in df.columns:
            t = df[TEMP_COL]
            rh = df[HUMIDITY_COL]
            # Convert to Fahrenheit for standard Rothfusz Heat Index
            t_f = t * 1.8 + 32.0
            hi_f = (
                -42.379
                + 2.04901523 * t_f
                + 10.14333127 * rh
                - 0.22475541 * t_f * rh
                - 0.00683783 * (t_f ** 2)
                - 0.05481717 * (rh ** 2)
                + 0.00122874 * (t_f ** 2) * rh
                + 0.00085282 * t_f * (rh ** 2)
                - 0.00000199 * (t_f ** 2) * (rh ** 2)
            )
            # Apparent temperature in Celsius
            df["heat_index_c"] = np.where(t_f >= 80.0, (hi_f - 32.0) / 1.8, t)
            df["temp_humidity_interaction"] = df[TEMP_COL] * (df[HUMIDITY_COL] / 100.0)

        return df
