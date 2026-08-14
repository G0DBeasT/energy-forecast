import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_PROC, TARGET_COL, DATE_COL, TEMP_COL, HUMIDITY_COL
from src.data_fetch import generate_hourly_grid_data

def load_raw(filename: str = "raw_hourly_grid_demand.csv") -> pd.DataFrame:
    """Load raw demand and weather dataset, ensuring datetime index."""
    raw_path = DATA_RAW / filename
    if not raw_path.exists():
        print(f"Raw file {raw_path} not found. Fetching raw data...")
        generate_hourly_grid_data()
        
    df = pd.read_csv(raw_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Cleaning Strategy:
    1. Resample index to strict 1H resolution to ensure uninterrupted datetime sequence.
    2. Outlier Clipping: Clip target demand to [0.1th, 99.9th] percentiles to neutralize sensor errors.
    3. Missing Imputation: Forward fill short gaps (<= 4h) and linearly interpolate any remaining gaps.
    """
    df = df.resample('1h').mean()

    # Outlier clipping on demand
    lo = df[TARGET_COL].quantile(0.001)
    hi = df[TARGET_COL].quantile(0.999)
    df[TARGET_COL] = df[TARGET_COL].clip(lo, hi)

    # Impute missing values
    df[TARGET_COL] = df[TARGET_COL].ffill(limit=4).interpolate(method='linear')
    if TEMP_COL in df.columns:
        df[TEMP_COL] = df[TEMP_COL].ffill(limit=4).interpolate(method='linear')
    if HUMIDITY_COL in df.columns:
        df[HUMIDITY_COL] = df[HUMIDITY_COL].ffill(limit=4).interpolate(method='linear')

    return df

def save_processed(df: pd.DataFrame, filename: str = "hourly_demand.parquet") -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROC / filename
    df.to_parquet(out_path)
    print(f"Saved {len(df)} cleaned rows to {out_path}")

def run_data_pipeline():
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    save_processed(df_clean)
    print("Data cleaning pipeline finished cleanly.")
    print("Cleaned shape:", df_clean.shape)
    print("Null count:\n", df_clean.isnull().sum())
    return df_clean

if __name__ == "__main__":
    run_data_pipeline()
