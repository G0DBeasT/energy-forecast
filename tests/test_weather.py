"""
Unit tests for weather service, climatology fallback, and derived indicators.
"""

import pytest
import pandas as pd
import numpy as np
from src.weather.climatology import ClimatologyWeatherModel
from src.weather.client import WeatherService
from src.config import TEMP_COL, HUMIDITY_COL, BASE_TEMP_C


def test_climatology_profile_generation():
    """Test that ClimatologyWeatherModel generates physically reasonable temperature and humidity."""
    model = ClimatologyWeatherModel()
    dates = pd.date_range("2024-06-01", periods=48, freq="1h")
    df_clim = model.generate_profile(dates, add_noise=False)

    assert len(df_clim) == 48
    assert (df_clim[TEMP_COL] >= 10.0).all() and (df_clim[TEMP_COL] <= 55.0).all()
    assert (df_clim[HUMIDITY_COL] >= 10.0).all() and (df_clim[HUMIDITY_COL] <= 100.0).all()


def test_weather_service_derived_variables():
    """Test calculation of CDD, HDD, and Apparent Heat Index."""
    dates = pd.date_range("2024-07-01", periods=24, freq="1h")
    df = pd.DataFrame({
        TEMP_COL: [35.0] * 12 + [15.0] * 12,
        HUMIDITY_COL: [60.0] * 24,
    }, index=dates)

    df_derived = WeatherService.calculate_derived_weather(df)

    assert "cdd" in df_derived.columns
    assert "hdd" in df_derived.columns
    assert "heat_index_c" in df_derived.columns

    # When temp = 35C (> 18.33C), CDD > 0 and HDD == 0
    assert df_derived["cdd"].iloc[0] == pytest.approx(35.0 - BASE_TEMP_C, rel=1e-2)
    assert df_derived["hdd"].iloc[0] == 0.0

    # When temp = 15C (< 18.33C), HDD > 0 and CDD == 0
    assert df_derived["hdd"].iloc[15] == pytest.approx(BASE_TEMP_C - 15.0, rel=1e-2)
    assert df_derived["cdd"].iloc[15] == 0.0


def test_weather_service_horizon_tagging():
    """Test that get_weather_for_horizon attaches weather source tags."""
    service = WeatherService()
    dates = pd.date_range("2024-08-01", periods=48, freq="1h")
    df_w = service.get_weather_for_horizon(dates)

    assert "weather_source" in df_w.columns
    assert len(df_w) == 48
    assert not df_w["weather_source"].isnull().any()
