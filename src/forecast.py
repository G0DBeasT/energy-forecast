import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import holidays as hol

from src.config import DATA_PROC, TARGET_COL, FORECAST_HORIZON, TEMP_COL, HUMIDITY_COL
from src.features import build_features, add_calendar_features, add_fourier_features, add_holiday_features, add_weather_features

def run_forecast():
    best_model_path = DATA_PROC / 'best_model.pkl'
    feat_path = DATA_PROC / 'features.parquet'
    
    if not best_model_path.exists() or not feat_path.exists():
        raise FileNotFoundError("Prerequisite files missing. Please run Phase 4 (features) and Phase 5 (train) first.")
        
    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)
        
    df_feat = pd.read_parquet(feat_path)
    feature_cols = [c for c in df_feat.columns if c != TARGET_COL]
    
    print(f"Loaded trained model: {type(model).__name__}")
    print(f"Dataset last date: {df_feat.index[-1]}")
    
    # Generate 48-hour future timeline
    last_dt = df_feat.index[-1]
    future_dates = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=FORECAST_HORIZON, freq='1h')
    
    # We maintain a working DataFrame combining recent history and future rows to update lags dynamically
    working_df = pd.read_parquet(DATA_PROC / 'hourly_demand.parquet').copy()
    
    # Project future 48h weather (e.g. smooth diurnal continuation of recent weather)
    recent_temp = working_df[TEMP_COL].iloc[-24:].values
    recent_hum  = working_df[HUMIDITY_COL].iloc[-24:].values
    
    future_temp = np.tile(recent_temp, 2)[:FORECAST_HORIZON] + np.random.normal(0, 0.5, FORECAST_HORIZON)
    future_hum  = np.tile(recent_hum, 2)[:FORECAST_HORIZON] + np.random.normal(0, 1.0, FORECAST_HORIZON)
    
    future_df = pd.DataFrame({
        TARGET_COL: np.nan,
        TEMP_COL: np.round(future_temp, 2),
        HUMIDITY_COL: np.round(future_hum, 2)
    }, index=future_dates)
    
    full_df = pd.concat([working_df.tail(8800), future_df])
    
    # Recursive multi-step prediction loop for 48 hours
    predictions = []
    
    for dt in future_dates:
        # Re-compute features up to current dt
        temp_df = full_df.loc[:dt].copy()
        
        # Calendar & Fourier
        temp_df['hour']         = temp_df.index.hour
        temp_df['day_of_week']  = temp_df.index.dayofweek
        temp_df['day_of_month'] = temp_df.index.day
        temp_df['month']        = temp_df.index.month
        temp_df['quarter']      = temp_df.index.quarter
        temp_df['year']         = temp_df.index.year
        temp_df['day_of_year']  = temp_df.index.dayofyear
        temp_df['is_weekend']   = (temp_df.index.dayofweek >= 5).astype(int)
        
        temp_df['sin_hour'] = np.sin(2 * np.pi * temp_df.index.hour / 24.0)
        temp_df['cos_hour'] = np.cos(2 * np.pi * temp_df.index.hour / 24.0)
        temp_df['sin_week'] = np.sin(2 * np.pi * temp_df.index.dayofweek / 7.0)
        temp_df['cos_week'] = np.cos(2 * np.pi * temp_df.index.dayofweek / 7.0)
        temp_df['sin_year'] = np.sin(2 * np.pi * temp_df.index.dayofyear / 365.25)
        temp_df['cos_year'] = np.cos(2 * np.pi * temp_df.index.dayofyear / 365.25)
        
        # Holidays
        try:
            chols = hol.country_holidays('IN')
        except Exception:
            chols = hol.country_holidays('US')
        ndates = temp_df.index.normalize()
        temp_df['is_holiday']      = ndates.isin(chols).astype(int)
        temp_df['is_pre_holiday']  = (ndates + timedelta(days=1)).isin(chols).astype(int)
        temp_df['is_post_holiday'] = (ndates - timedelta(days=1)).isin(chols).astype(int)
        
        # Demand Lags
        col = TARGET_COL
        temp_df['lag_1h']    = temp_df[col].shift(1)
        temp_df['lag_2h']    = temp_df[col].shift(2)
        temp_df['lag_3h']    = temp_df[col].shift(3)
        temp_df['lag_24h']   = temp_df[col].shift(24)
        temp_df['lag_48h']   = temp_df[col].shift(48)
        temp_df['lag_168h']  = temp_df[col].shift(168)
        temp_df['lag_336h']  = temp_df[col].shift(336)
        temp_df['lag_720h']  = temp_df[col].shift(720)
        if len(temp_df) >= 8761 and 'lag_8760h' in feature_cols:
            temp_df['lag_8760h'] = temp_df[col].shift(8760)
            
        # Rolling stats
        shifted = temp_df[col].shift(1)
        temp_df['rolling_mean_24h']  = shifted.rolling(24).mean()
        temp_df['rolling_std_24h']   = shifted.rolling(24).std()
        temp_df['rolling_min_24h']   = shifted.rolling(24).min()
        temp_df['rolling_max_24h']   = shifted.rolling(24).max()
        temp_df['rolling_mean_168h'] = shifted.rolling(168).mean()
        temp_df['rolling_std_168h']  = shifted.rolling(168).std()
        temp_df['rolling_min_168h']  = shifted.rolling(168).min()
        temp_df['rolling_max_168h']  = shifted.rolling(168).max()
        temp_df['rolling_mean_720h'] = shifted.rolling(720).mean()
        temp_df['rolling_std_720h']  = shifted.rolling(720).std()
        
        # Weather features
        t = temp_df[TEMP_COL]
        rh = temp_df[HUMIDITY_COL]
        temp_df['cdd'] = np.maximum(t - 18.3, 0.0)
        temp_df['hdd'] = np.maximum(18.3 - t, 0.0)
        t_shifted = t.shift(1)
        temp_df['temp_lag_24h']          = t.shift(24)
        temp_df['temp_rolling_mean_24h'] = t_shifted.rolling(24).mean()
        temp_df['temp_rolling_mean_168h']= t_shifted.rolling(168).mean()
        
        t_f = t * 1.8 + 32.0
        hi_f = (-42.379 + 2.04901523*t_f + 10.14333127*rh - 0.22475541*t_f*rh - 
                0.00683783*t_f*t_f - 0.05481717*rh*rh + 0.00122874*t_f*t_f*rh + 
                0.00085282*t_f*rh*rh - 0.00000199*t_f*t_f*rh*rh)
        temp_df['heat_index_c'] = np.where(t_f >= 80.0, (hi_f - 32.0) / 1.8, t)
        temp_df['humidity_rolling_mean_24h'] = rh.shift(1).rolling(24).mean()
        
        # Extract row for current dt
        current_features = temp_df.loc[[dt], feature_cols]
        pred_val = float(model.predict(current_features)[0])
        
        # Update predicted value into full_df so subsequent lags use it
        full_df.loc[dt, TARGET_COL] = pred_val
        predictions.append(pred_val)

    # Add 10th and 90th percentile prediction interval bounds (using CV std residual ~ 165 MW)
    std_err = 165.0
    forecast_df = pd.DataFrame({
        'datetime': future_dates,
        'forecast_mw': np.round(predictions, 2),
        'lower_bound_mw': np.round(np.array(predictions) - 1.645 * std_err, 2),
        'upper_bound_mw': np.round(np.array(predictions) + 1.645 * std_err, 2)
    })
    
    out_csv = DATA_PROC / 'forecast.csv'
    forecast_df.to_csv(out_csv, index=False)
    print(f"Generated and saved {len(forecast_df)}-hour multi-step forecast to {out_csv}.")
    print(forecast_df.head(10))
    return forecast_df

if __name__ == '__main__':
    run_forecast()
