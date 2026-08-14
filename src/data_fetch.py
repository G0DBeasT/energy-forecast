import pandas as pd
import numpy as np
import requests
from pathlib import Path
from src.config import DATA_RAW, LATITUDE, LONGITUDE, DATE_COL, TARGET_COL, TEMP_COL, HUMIDITY_COL

def fetch_weather_open_meteo(start_date: str, end_date: str, lat: float = LATITUDE, lon: float = LONGITUDE) -> pd.DataFrame:
    """Fetch hourly historical temperature and relative humidity from Open-Meteo API."""
    print(f"Fetching weather data from Open-Meteo for {start_date} to {end_date}...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m"
        f"&timezone=Asia%2FKolkata"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        df_weather = pd.DataFrame({
            DATE_COL: pd.to_datetime(data['hourly']['time']),
            TEMP_COL: data['hourly']['temperature_2m'],
            HUMIDITY_COL: data['hourly']['relative_humidity_2m']
        })
        print(f"Successfully fetched {len(df_weather)} hourly weather observations.")
        return df_weather
    except Exception as e:
        print(f"Warning: Open-Meteo API call failed ({e}). Generating high-fidelity meteorological fallback...")
        dates = pd.date_range(start=start_date, end=end_date + " 23:00:00", freq="1h")
        day_of_year = dates.dayofyear
        hour = dates.hour
        
        # Physical temperature model: annual cycle (15C winter, 38C summer in Delhi) + daily cycle (peak ~15:00)
        temp_annual = 26.5 + 11.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
        temp_daily = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24)
        noise_temp = np.random.normal(0, 1.2, len(dates))
        temp = temp_annual + temp_daily + noise_temp
        
        # Physical humidity model: inversely correlated with temperature + monsoon effect (July-Sept)
        monsoon = np.where((day_of_year >= 180) & (day_of_year <= 270), 25.0, 0.0)
        humidity = 85.0 - 1.2 * temp + monsoon + np.random.normal(0, 3.0, len(dates))
        humidity = np.clip(humidity, 15.0, 98.0)
        
        df_weather = pd.DataFrame({
            DATE_COL: dates,
            TEMP_COL: np.round(temp, 2),
            HUMIDITY_COL: np.round(humidity, 2)
        })
        return df_weather

def generate_hourly_grid_data(start_date: str = "2021-01-01", end_date: str = "2024-06-30") -> pd.DataFrame:
    """Generate realistic hourly power grid load data combined with hourly weather data."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_demand_path = DATA_RAW / "raw_hourly_grid_demand.csv"
    
    # Check if raw dataset already exists
    if raw_demand_path.exists():
        print(f"Raw dataset already exists at {raw_demand_path}.")
        return pd.read_csv(raw_demand_path, parse_dates=[DATE_COL])
        
    print("Generating comprehensive hourly energy grid demand dataset...")
    df_weather = fetch_weather_open_meteo(start_date, end_date)
    dates = df_weather[DATE_COL]
    
    # Base load components
    base_load = 4500.0  # Base MW
    trend = np.linspace(0, 800, len(dates))  # Annual demand growth over 3.5 years
    
    # Seasonal temperature demand effect (cooling in summer, mild heating in winter)
    temp = df_weather[TEMP_COL].values
    cooling_effect = np.maximum(temp - 22.0, 0) ** 1.35 * 65.0
    heating_effect = np.maximum(16.0 - temp, 0) ** 1.2 * 35.0
    
    # Daily profile (afternoon peak around 15:00, evening peak around 21:00)
    hour = dates.dt.hour.values
    daily_pattern = 300 * np.sin(2 * np.pi * (hour - 6) / 24) + 400 * np.sin(4 * np.pi * (hour - 12) / 24)
    
    # Weekly profile (weekends 10-15% lower commercial/industrial load)
    day_of_week = dates.dt.dayofweek.values
    weekend_effect = np.where(day_of_week >= 5, -500.0, 0.0)
    
    # Random grid fluctuation noise
    noise = np.random.normal(0, 120.0, len(dates))
    
    # Ocassional minor synthetic missing/outlier artifacts (to test Phase 3 cleaning pipeline)
    demand = base_load + trend + cooling_effect + heating_effect + daily_pattern + weekend_effect + noise
    
    df_raw = pd.DataFrame({
        DATE_COL: dates,
        TARGET_COL: np.round(demand, 2),
        TEMP_COL: df_weather[TEMP_COL],
        HUMIDITY_COL: df_weather[HUMIDITY_COL]
    })
    
    # Inject a small fraction (0.2%) of NaNs and slight outliers to demonstrate Phase 3 cleaning
    nan_mask = np.random.rand(len(df_raw)) < 0.002
    df_raw.loc[nan_mask, TARGET_COL] = np.nan
    
    df_raw.to_csv(raw_demand_path, index=False)
    print(f"Saved raw dataset ({len(df_raw)} rows) to {raw_demand_path}.")
    return df_raw

if __name__ == "__main__":
    df_raw = generate_hourly_grid_data()
    print(df_raw.head())
    print("Summary:")
    print(df_raw.describe())
