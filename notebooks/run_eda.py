import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from src.config import DATA_PROC, REPORTS, TARGET_COL, TEMP_COL, HUMIDITY_COL

def run_eda():
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = (14, 5)

    parquet_path = DATA_PROC / "hourly_demand.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed file {parquet_path} does not exist. Run Phase 3 first.")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataset: {df.shape} from {df.index[0]} to {df.index[-1]}")

    # 1. Full Series Overview
    plt.figure()
    df[TARGET_COL].plot(title='Full Energy Grid Demand Series (MW)', color='#378ADD')
    plt.ylabel('Demand (MW)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '01_full_series.png', dpi=120)
    plt.close()

    # 2. One Week Zoom
    plt.figure()
    sample_start = df.index[-24 * 60]  # Representative recent week
    sample_end = sample_start + pd.Timedelta(days=7)
    df.loc[sample_start:sample_end, TARGET_COL].plot(title=f'One-Week Hourly Demand Pattern ({sample_start.date()} to {sample_end.date()})', color='#EF9F27')
    plt.ylabel('Demand (MW)')
    plt.xlabel('Date & Hour')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '02_one_week.png', dpi=120)
    plt.close()

    # 3. Average Demand by Hour of Day
    plt.figure()
    df.groupby(df.index.hour)[TARGET_COL].mean().plot(kind='bar', color='#4CAF50', title='Average Energy Demand by Hour of Day')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Avg Demand (MW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '03_hourly_pattern.png', dpi=120)
    plt.close()

    # 4. Average Demand by Day of Week
    plt.figure()
    day_means = df.groupby(df.index.dayofweek)[TARGET_COL].mean()
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.bar(day_labels, day_means.values, color='#9C27B0')
    plt.title('Average Energy Demand by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Avg Demand (MW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '04_weekly_pattern.png', dpi=120)
    plt.close()

    # 5. Average Demand by Month
    plt.figure()
    month_means = df.groupby(df.index.month)[TARGET_COL].mean()
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(month_labels[:len(month_means)], month_means.values, color='#FF5722')
    plt.title('Average Energy Demand by Month of Year')
    plt.xlabel('Month')
    plt.ylabel('Avg Demand (MW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '05_monthly_pattern.png', dpi=120)
    plt.close()

    # 6. STL Decomposition
    clean_series = df[TARGET_COL].dropna()
    stl = STL(clean_series, period=24)
    res = stl.fit()
    fig = res.plot()
    fig.set_size_inches(14, 8)
    plt.suptitle('STL Time Series Decomposition (Trend, Seasonal 24h, Residual)', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / '06_stl_decomp.png', dpi=120)
    plt.close()

    # 7. ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(clean_series, lags=48, ax=ax1, color='#00BCD4')
    plot_pacf(clean_series, lags=48, ax=ax2, color='#E91E63')
    ax1.set_title('Autocorrelation (ACF) - Lags up to 48 Hours')
    ax2.set_title('Partial Autocorrelation (PACF) - Lags up to 48 Hours')
    plt.tight_layout()
    plt.savefig(REPORTS / '07_acf_pacf.png', dpi=120)
    plt.close()

    # 8. Weather vs Demand Scatter Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(df[TEMP_COL], df[TARGET_COL], alpha=0.15, color='#FF9800', s=10)
    ax1.set_title('Demand (MW) vs Temperature (°C)')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Demand (MW)')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(df[HUMIDITY_COL], df[TARGET_COL], alpha=0.15, color='#03A9F4', s=10)
    ax2.set_title('Demand (MW) vs Relative Humidity (%)')
    ax2.set_xlabel('Relative Humidity (%)')
    ax2.set_ylabel('Demand (MW)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS / '08_weather_demand_scatter.png', dpi=120)
    plt.close()

    # 9. Compute Naive Baseline Benchmark (Same Hour Last Week: shift 168h)
    df['naive_pred'] = df[TARGET_COL].shift(168)
    test_df = df.dropna().tail(60 * 24)  # Holdout last 60 days
    naive_mape = mean_absolute_percentage_error(test_df[TARGET_COL], test_df['naive_pred'])
    naive_mae = mean_absolute_error(test_df[TARGET_COL], test_df['naive_pred'])

    print("\n" + "="*50)
    print("EDA SUMMARY & NAIVE BASELINE EVALUATION")
    print("="*50)
    print(f"Dataset Range      : {df.index[0]} to {df.index[-1]}")
    print(f"Total Hourly Rows  : {len(df)}")
    print(f"Target Demand Mean : {df[TARGET_COL].mean():.2f} MW (Std: {df[TARGET_COL].std():.2f})")
    print(f"Temperature Range  : {df[TEMP_COL].min():.1f}°C to {df[TEMP_COL].max():.1f}°C")
    print(f"Naive Baseline MAPE: {naive_mape:.2%}")
    print(f"Naive Baseline MAE : {naive_mae:.2f} MW")
    print("="*50)
    print(f"Saved all 8 diagnostic charts to {REPORTS}/")

if __name__ == "__main__":
    run_eda()
