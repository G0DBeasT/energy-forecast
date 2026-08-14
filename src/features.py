import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import holidays as hol
from datetime import timedelta
from src.config import DATA_PROC, TARGET_COL, TEMP_COL, HUMIDITY_COL

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal, calendar, day of week, day of month, month of year features."""
    df = df.copy()
    df['hour']         = df.index.hour
    df['day_of_week']  = df.index.dayofweek    # 0=Mon, 6=Sun
    df['day_of_month'] = df.index.day
    df['month']        = df.index.month
    df['quarter']      = df.index.quarter
    df['year']         = df.index.year
    df['day_of_year']  = df.index.dayofyear
    df['is_weekend']   = (df.index.dayofweek >= 5).astype(int)
    return df

def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cyclical temporal patterns continuously via sine/cosine pairs."""
    df = df.copy()
    # Daily cycle (24h)
    df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    # Weekly cycle (7d)
    df['sin_week'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df['cos_week'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    # Annual cycle (365.25d)
    df['sin_year'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    return df

def add_holiday_features(df: pd.DataFrame, country: str = 'IN') -> pd.DataFrame:
    """Mark public holidays and surrounding days."""
    df = df.copy()
    try:
        country_hols = hol.country_holidays(country)
    except Exception:
        country_hols = hol.country_holidays('US')
        
    dates = df.index.normalize()
    df['is_holiday']      = dates.isin(country_hols).astype(int)
    df['is_pre_holiday']  = (dates + timedelta(days=1)).isin(country_hols).astype(int)
    df['is_post_holiday'] = (dates - timedelta(days=1)).isin(country_hols).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Add historical demand lags: previous hours, previous day, week, month, year."""
    df = df.copy()
    df['lag_1h']    = df[col].shift(1)
    df['lag_2h']    = df[col].shift(2)
    df['lag_3h']    = df[col].shift(3)
    df['lag_24h']   = df[col].shift(24)     # previous day (same hour)
    df['lag_48h']   = df[col].shift(48)     # 2 days ago
    df['lag_168h']  = df[col].shift(168)    # previous week (same hour)
    df['lag_336h']  = df[col].shift(336)    # 2 weeks ago
    df['lag_720h']  = df[col].shift(720)    # previous month (30 days ago)
    df['lag_8760h'] = df[col].shift(8760)   # previous year (365 days ago)
    return df

def add_rolling_features(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Add rolling statistics shifted by 1 hour to prevent data leakage."""
    df = df.copy()
    shifted = df[col].shift(1)
    
    # 24-hour window
    df['rolling_mean_24h'] = shifted.rolling(24).mean()
    df['rolling_std_24h']  = shifted.rolling(24).std()
    df['rolling_min_24h']  = shifted.rolling(24).min()
    df['rolling_max_24h']  = shifted.rolling(24).max()
    
    # 168-hour (1 week) window
    df['rolling_mean_168h'] = shifted.rolling(168).mean()
    df['rolling_std_168h']  = shifted.rolling(168).std()
    df['rolling_min_168h']  = shifted.rolling(168).min()
    df['rolling_max_168h']  = shifted.rolling(168).max()
    
    # 720-hour (30 days) window
    df['rolling_mean_720h'] = shifted.rolling(720).mean()
    df['rolling_std_720h']  = shifted.rolling(720).std()
    
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather derived indicators (CDD, HDD, Heat Index, temperature lags/rolling)."""
    df = df.copy()
    if TEMP_COL in df.columns:
        t = df[TEMP_COL]
        # Cooling Degree Days & Heating Degree Days (Base 18.3°C / 65°F)
        df['cdd'] = np.maximum(t - 18.3, 0.0)
        df['hdd'] = np.maximum(18.3 - t, 0.0)
        
        # Temperature Lags & Rolling Stats
        t_shifted = t.shift(1)
        df['temp_lag_24h']          = t.shift(24)
        df['temp_rolling_mean_24h'] = t_shifted.rolling(24).mean()
        df['temp_rolling_mean_168h']= t_shifted.rolling(168).mean()
        
    if TEMP_COL in df.columns and HUMIDITY_COL in df.columns:
        t = df[TEMP_COL]
        rh = df[HUMIDITY_COL]
        # Heat Index / Apparent Temperature approximation (°C)
        # Rothfusz heat index regression simplified for Celsius
        t_f = t * 1.8 + 32.0
        hi_f = (-42.379 + 2.04901523*t_f + 10.14333127*rh - 0.22475541*t_f*rh - 
                0.00683783*t_f*t_f - 0.05481717*rh*rh + 0.00122874*t_f*t_f*rh + 
                0.00085282*t_f*rh*rh - 0.00000199*t_f*t_f*rh*rh)
        df['heat_index_c'] = np.where(t_f >= 80.0, (hi_f - 32.0) / 1.8, t)
        
        # Humidity rolling mean
        rh_shifted = rh.shift(1)
        df['humidity_rolling_mean_24h'] = rh_shifted.rolling(24).mean()
        
    return df

def build_features() -> pd.DataFrame:
    """Execute complete feature engineering pipeline."""
    parquet_path = DATA_PROC / 'hourly_demand.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed file {parquet_path} not found. Run Phase 3 first.")
        
    df = pd.read_parquet(parquet_path)
    df = add_calendar_features(df)
    df = add_fourier_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_features(df)
    
    # Drop initial rows with NaNs resulting from lag calculations
    # Drop lag_8760h if dataset length is shorter, otherwise dropna on all features
    if len(df) < 8760 + 100:
        df = df.drop(columns=['lag_8760h'])
        
    df = df.dropna()
    return df

def run_features_pipeline():
    df_feat = build_features()
    out_path = DATA_PROC / 'features.parquet'
    df_feat.to_parquet(out_path)
    print(f"Feature matrix built and saved to {out_path}.")
    print(f"Matrix shape: {df_feat.shape}")
    print(f"Total features created: {df_feat.shape[1] - 1}")
    print("Features list:", [c for c in df_feat.columns if c != TARGET_COL])

if __name__ == '__main__':
    run_features_pipeline()
