# Energy Grid Load Forecasting — Project Board

> Use this as a long-running TODO. Check off as you go. Skip or reorder freely.

---

## Before You Start — Pick Your Data Mode

Three honest options. Everything from Phase 1 onward is identical regardless of choice.

| Mode  | Dataset                   | Effort                  | Realism               |
| ----- | ------------------------- | ----------------------- | --------------------- |
| **A** | MERIT India / data.gov.in | High (scraping/cleanup) | Most honest           |
| **B** | PJM (Rob Mulla, Kaggle)   | Low (clean, 10 years)   | Methodology identical |
| **C** | Some India Kaggle dataset | Medium                  | Middle ground         |

> If Indian data is too messy or unavailable after 2–3 days of searching, use PJM. Frame it in your report as: _"Pipeline designed for Indian grid; demonstrated on publicly available PJM dataset pending clean substation-level data access."_ Nobody will question this in a college minor project.

---

## Phase 0 · Setup

_Goal: project runs from a fresh clone by end of this phase._

### 0.1 Environment

- [x] `mkdir energy-forecast && cd energy-forecast`
- [x] `uv init` (or just `python -m venv .venv` if skipping uv)
- [ ] `source .venv/bin/activate`
- [ ] Create `requirements.txt`:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  lightgbm
  xgboost
  mlflow
  skforecast
  statsmodels
  streamlit
  plotly
  holidays
  python-dotenv
  pyarrow
  jupyter
  ipykernel
  ```
- [ ] `pip install -r requirements.txt`
- [ ] Confirm: `python -c "import lightgbm, mlflow, skforecast; print('ok')"` — no errors

### 0.2 Git

- [ ] `git init`
- [ ] Create `.gitignore`:
  ```
  .venv/
  data/raw/
  mlruns/
  .env
  __pycache__/
  *.pyc
  *.pyo
  .ipynb_checkpoints/
  *.parquet
  ```
- [ ] `git add . && git commit -m "chore: init project"`
- [ ] Create repo on GitHub, push: `git remote add origin ... && git push -u origin main`

### 0.3 Project Structure

- [ ] `mkdir -p data/raw data/processed notebooks src reports`
- [ ] `touch src/__init__.py src/config.py src/data.py src/features.py src/train.py src/evaluate.py src/forecast.py`
- [ ] `touch Makefile README.md app.py`

### 0.4 Config file

- [ ] Fill in `src/config.py`:

  ```python
  from pathlib import Path

  ROOT = Path(__file__).parent.parent
  DATA_RAW     = ROOT / "data" / "raw"
  DATA_PROC    = ROOT / "data" / "processed"
  REPORTS      = ROOT / "reports"

  TARGET_COL       = "demand_mw"
  DATE_COL         = "datetime"
  FREQ             = "1h"
  FORECAST_HORIZON = 48
  MLFLOW_URI       = "http://localhost:5000"
  EXPERIMENT_NAME  = "energy-forecast"
  ```

### 0.5 Makefile skeleton

- [ ] Add to `Makefile`:

  ```makefile
  .PHONY: data features train forecast dashboard

  data:
  	python src/data.py

  features:
  	python src/features.py

  train:
  	python src/train.py

  forecast:
  	python src/forecast.py

  dashboard:
  	streamlit run app.py
  ```

  > Note: indentation in Makefiles must be a tab, not spaces. Neovim will handle this correctly with `:set ft=make`.

- [ ] Commit: `"chore: project structure and config"`

---

## Phase 1 · Data Acquisition

_Goal: a raw file in `data/raw/` that you understand._

### 1.1 Find India data (try in order, give each 1–2 days max)

**Option A — data.gov.in**

- [ ] Go to `data.gov.in`, search: `"electricity consumption"` or `"power demand hourly"`
- [ ] Filter by: format = CSV/XLS, organization = Ministry of Power or POSOCO
- [ ] Download anything that looks hourly. Check the date range and resolution before committing.

**Option B — MERIT India (merit.posoco.in)**

- [ ] Go to `merit.posoco.in` → Reports / Historical Data
- [ ] This has hourly all-India and regional generation data
- [ ] Download 1–2 years manually if no bulk download exists
- [ ] Write a short Python script to merge monthly CSVs into one file if needed:
  ```python
  import glob, pandas as pd
  files = glob.glob("data/raw/merit_*.csv")
  df = pd.concat([pd.read_csv(f) for f in files])
  df.to_csv("data/raw/merit_combined.csv", index=False)
  ```

**Option C — Kaggle India electricity datasets**

- [ ] Search Kaggle: `"India electricity demand hourly"` or `"India load forecasting"`
- [ ] Vet the dataset: does it have hourly resolution? At least 2 years of data? A clear source cited?
- [ ] Download via Kaggle CLI: `kaggle datasets download <slug> -p data/raw/`

**Option D — PJM fallback (Rob Mulla)**

- [ ] `kaggle datasets download robikscube/hourly-energy-consumption -p data/raw/`
- [ ] Unzip: `unzip data/raw/hourly-energy-consumption.zip -d data/raw/`
- [ ] Simplest file to start with: `AEP_hourly.csv` (single region, 17 years)

### 1.2 First look — answer these questions before touching anything else

- [ ] Open the raw file in Python/Jupyter:
  ```python
  import pandas as pd
  df = pd.read_csv("data/raw/your_file.csv")
  print(df.shape)
  print(df.dtypes)
  print(df.head(20))
  print(df.isnull().sum())
  print(df.describe())
  ```
- [ ] What is the datetime column called? What format is it in?
- [ ] What is the demand column called? Units — MW, MWh, kWh?
- [ ] Temporal resolution — hourly, 15-min, daily?
- [ ] Date range — how many years?
- [ ] Approximate missing value count
- [ ] Write the answers as comments in a notebook cell or a `data/README.md`

> **India data reality:** Indian datasets often have: inconsistent column names, mixed date formats (DD-MM-YYYY and MM/DD/YYYY in the same file), missing hours, and duplicate rows. This is expected. The cleaning phase handles it. Messy data = more realistic experience.

- [ ] Commit raw data source info (not the file itself): `git commit -m "data: add raw data source notes"`

---

## Phase 2 · Exploratory Data Analysis

_Goal: you understand the data. You have a baseline number to beat._

### 2.1 Create notebook

- [ ] `notebooks/01_eda.ipynb` — kernel = your `.venv`
- [ ] Set plot style at top:
  ```python
  import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np
  plt.style.use('dark_background')
  plt.rcParams['figure.figsize'] = (14, 4)
  ```

### 2.2 Load and parse datetime

- [ ] Load and parse:
  ```python
  df = pd.read_csv("data/raw/your_file.csv")
  df['datetime'] = pd.to_datetime(df['your_datetime_col'])
  df = df.set_index('datetime').sort_index()
  df = df[['demand_mw']]  # keep only what you need
  ```
- [ ] Indian data: check timezone. If timestamps are in IST, localize:
  ```python
  df.index = df.index.tz_localize('Asia/Kolkata')
  # or convert to UTC if needed for consistency
  ```
- [ ] Check for duplicates: `df.index.duplicated().sum()` — should be 0
- [ ] Check time gaps: `df.index.to_series().diff().value_counts().head(5)` — dominant gap should be 1H

### 2.3 Time series plots (save everything to `reports/`)

- [ ] Full series overview:
  ```python
  df['demand_mw'].plot(title='Full demand series')
  plt.tight_layout()
  plt.savefig('reports/01_full_series.png', dpi=100)
  ```
- [ ] Zoom into one week (pick a representative non-holiday week):
  ```python
  df['2023-01-09':'2023-01-15']['demand_mw'].plot(title='One week zoom')
  plt.savefig('reports/02_one_week.png', dpi=100)
  ```
- [ ] Average demand by hour-of-day:
  ```python
  df.groupby(df.index.hour)['demand_mw'].mean().plot(kind='bar', title='Avg demand by hour')
  plt.savefig('reports/03_hourly_pattern.png', dpi=100)
  ```
- [ ] Average demand by day-of-week:
  ```python
  df.groupby(df.index.dayofweek)['demand_mw'].mean().plot(kind='bar',
      title='Avg demand by day (0=Mon)', xticks=range(7),
      xticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
  plt.savefig('reports/04_weekly_pattern.png', dpi=100)
  ```
- [ ] Average demand by month:
  ```python
  df.groupby(df.index.month)['demand_mw'].mean().plot(kind='bar', title='Avg demand by month')
  plt.savefig('reports/05_monthly_pattern.png', dpi=100)
  ```

> **India-specific observations to look for:**
>
> - Two daily peaks: afternoon (~14:00–16:00) and evening (~19:00–22:00)
> - Summer months (Apr–Jun) significantly higher than winter
> - Weekdays higher than weekends
> - If you see this pattern, your data is good. If not, dig into why.

### 2.4 Missing values and outliers

- [ ] Count missing per month:
  ```python
  df.resample('ME')['demand_mw'].apply(lambda x: x.isnull().sum()).plot(kind='bar')
  ```
- [ ] Check for physical impossibilities (zeros or near-zero during daytime):
  ```python
  print(df[df['demand_mw'] < df['demand_mw'].quantile(0.001)])
  print(df[df['demand_mw'] > df['demand_mw'].quantile(0.999)])
  ```
- [ ] Note: are there long stretches of missing data (>24h)? Which months?

### 2.5 STL decomposition

- [ ] Run STL on a clean chunk of your series:
  ```python
  from statsmodels.tsa.seasonal import STL
  clean = df['demand_mw'].dropna()
  stl = STL(clean, period=24)  # period=24 for hourly data (daily seasonality)
  result = stl.fit()
  fig = result.plot()
  fig.set_size_inches(14, 8)
  plt.tight_layout()
  plt.savefig('reports/06_stl_decomp.png', dpi=100)
  plt.show()
  ```
- [ ] Interpret each component:
  - **Trend**: going up over time? (India: likely yes)
  - **Seasonal**: strong regular pattern? (should be)
  - **Residual**: random-looking? (good) or structured? (something not captured)

### 2.6 ACF and PACF

- [ ] Plot autocorrelation up to 2 days:
  ```python
  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
  plot_acf(df['demand_mw'].dropna(), lags=48, ax=ax1)
  plot_pacf(df['demand_mw'].dropna(), lags=48, ax=ax2)
  plt.tight_layout()
  plt.savefig('reports/07_acf_pacf.png', dpi=100)
  ```
- [ ] Key thing to notice: large spikes at lags 24 and 48 confirm daily seasonality
- [ ] This justifies your lag feature choices in Phase 4

### 2.7 Naive baseline — your benchmark

- [ ] Compute it now. This number is your target for the rest of the project.

  ```python
  from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

  df['naive_pred'] = df['demand_mw'].shift(24 * 7)  # same hour last week

  # Evaluate only on last 60 days (simulate a test set)
  test = df.dropna().tail(60 * 24)
  naive_mape = mean_absolute_percentage_error(test['demand_mw'], test['naive_pred'])
  naive_mae  = mean_absolute_error(test['demand_mw'], test['naive_pred'])

  print(f"Naive MAPE : {naive_mape:.2%}")
  print(f"Naive MAE  : {naive_mae:.0f} MW")
  ```

- [ ] Write these numbers somewhere you'll see them throughout the project

### 2.8 EDA summary

- [ ] Add a markdown cell at the TOP of `01_eda.ipynb`:
  ```
  ## Summary
  - Data: [source], [date range], [resolution], [total rows]
  - Target column: [name], units: [unit]
  - Missing values: ~[X]% total, worst month: [month]
  - Naive baseline MAPE: [X.X]%
  - Key patterns: [what you saw]
  - Concerns: [data quality issues]
  ```
- [ ] Commit: `"notebook: complete EDA, naive MAPE = X%"`

---

## Phase 3 · Data Cleaning Pipeline

_Goal: `data/processed/hourly_demand.parquet` — clean, consistent, ready._

### 3.1 Decide on cleaning strategy (document your choices)

- [ ] For missing values — pick one:
  - Short gaps (<4h): forward fill (`ffill`)
  - Longer gaps: interpolate linearly
  - Very long gaps (>24h): leave as NaN, they'll be excluded when you call `dropna()` later
- [ ] For outliers — pick one:
  - Clip: values below 5th percentile or above 99.5th percentile → replace with rolling 24h mean
  - Flag: add an `is_outlier` boolean column, don't modify the value
- [ ] Write a 3-line comment in `src/data.py` explaining why you chose these strategies

### 3.2 Write `src/data.py`

```python
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROC, TARGET_COL, DATE_COL

def load_raw(filename: str) -> pd.DataFrame:
    path = DATA_RAW / filename
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    df = df[[TARGET_COL]]
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Resample to 1H (handles 15-min data or gaps in index)
    df = df.resample('1h').mean()

    # Handle outliers: clip to [1st, 99.5th] percentile
    lo = df[TARGET_COL].quantile(0.01)
    hi = df[TARGET_COL].quantile(0.995)
    df[TARGET_COL] = df[TARGET_COL].clip(lo, hi)

    # Fill short gaps (<=4h)
    df[TARGET_COL] = df[TARGET_COL].fillna(method='ffill', limit=4)

    return df

def save_processed(df: pd.DataFrame, filename: str = 'hourly_demand.parquet') -> None:
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROC / filename)
    print(f"Saved {len(df)} rows to {DATA_PROC / filename}")

if __name__ == '__main__':
    df = load_raw('your_raw_file.csv')   # <- change this
    df = clean(df)
    save_processed(df)
    print(df.info())
    print(df.isnull().sum())
```

- [ ] Fill in the actual filename
- [ ] Run: `python src/data.py` — does it complete without errors?
- [ ] Run: `python -c "import pandas as pd; print(pd.read_parquet('data/processed/hourly_demand.parquet').head())"` — does it load?
- [ ] Run: `make data` — should do the same
- [ ] Commit: `"feat: data cleaning pipeline"`

> **Parquet over CSV:** Preserves dtypes (including datetime index), compressed (~4x smaller), loads much faster. Standard in production ML pipelines.

---

## Phase 4 · Feature Engineering

_Goal: `data/processed/features.parquet` — ML-ready feature matrix._

This phase is where the project stops being generic. The quality of your features determines 70% of model quality.

### 4.1 Calendar features

- [ ] In `src/features.py`:
  ```python
  def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
      df = df.copy()
      df['hour']        = df.index.hour
      df['day_of_week'] = df.index.dayofweek   # 0=Mon, 6=Sun
      df['month']       = df.index.month
      df['quarter']     = df.index.quarter
      df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
      df['day_of_year'] = df.index.dayofyear
      df['year']        = df.index.year
      return df
  ```

### 4.2 Lag features

- [ ] Add:
  ```python
  def add_lag_features(df: pd.DataFrame, col: str = 'demand_mw') -> pd.DataFrame:
      df = df.copy()
      df['lag_1h']   = df[col].shift(1)    # last hour
      df['lag_24h']  = df[col].shift(24)   # same hour yesterday
      df['lag_48h']  = df[col].shift(48)   # same hour 2 days ago
      df['lag_168h'] = df[col].shift(168)  # same hour last week  (= naive baseline)
      df['lag_336h'] = df[col].shift(336)  # same hour 2 weeks ago
      return df
  ```
- [ ] Note: `lag_168h` is exactly the naive baseline predictor. If LightGBM can't beat just using this one feature, something is wrong.

### 4.3 Rolling statistics

- [ ] Add:
  ```python
  def add_rolling_features(df: pd.DataFrame, col: str = 'demand_mw') -> pd.DataFrame:
      df = df.copy()
      shifted = df[col].shift(1)  # shift by 1 to prevent current-hour leakage
      df['rolling_mean_24h']  = shifted.rolling(24).mean()
      df['rolling_std_24h']   = shifted.rolling(24).std()
      df['rolling_mean_168h'] = shifted.rolling(168).mean()
      df['rolling_max_24h']   = shifted.rolling(24).max()
      df['rolling_min_24h']   = shifted.rolling(24).min()
      return df
  ```
- [ ] The `.shift(1)` before `.rolling()` is important — without it you leak current-hour info into the model

### 4.4 Fourier features (encode cyclical patterns properly)

- [ ] Add:

  ```python
  import numpy as np

  def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
      df = df.copy()
      # Daily cycle
      df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
      df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
      # Weekly cycle
      df['sin_week'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
      df['cos_week'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
      # Annual cycle
      df['sin_year'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
      df['cos_year'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
      return df
  ```

- [ ] Why this and not just `hour`? Because `hour` is an integer 0–23: the model sees no relationship between 23 and 0. Sin/cos encodes that 11pm is geometrically close to midnight. Same for weeks and months.

### 4.5 India holiday features

- [ ] `pip install holidays` (already in requirements.txt)
- [ ] Add:

  ```python
  import holidays as hol

  def add_holiday_features(df: pd.DataFrame, country: str = 'IN') -> pd.DataFrame:
      df = df.copy()
      india_hols = hol.country_holidays(country)
      dates = df.index.normalize()  # strip time, keep date

      df['is_holiday'] = dates.isin(india_hols).astype(int)

      # Day before and after holiday (demand changes around holidays too)
      from datetime import timedelta
      df['is_pre_holiday']  = (dates + timedelta(days=1)).isin(india_hols).astype(int)
      df['is_post_holiday'] = (dates - timedelta(days=1)).isin(india_hols).astype(int)

      return df
  ```

- [ ] Note: `holidays` library covers national Indian holidays. Diwali, Holi, Independence Day etc. all show up here.
- [ ] If you're using PJM data, use `country='US'` and the logic is identical

### 4.6 Optional: temperature as a feature

- [ ] Open-Meteo API: free, no API key, historical hourly temperature

  ```python
  import requests

  def fetch_temperature(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
      url = (
          f"https://archive-api.open-meteo.com/v1/archive?"
          f"latitude={lat}&longitude={lon}"
          f"&start_date={start}&end_date={end}"
          f"&hourly=temperature_2m&timezone=Asia%2FKolkata"
      )
      r = requests.get(url)
      data = r.json()
      df = pd.DataFrame({'datetime': data['hourly']['time'],
                         'temp_c': data['hourly']['temperature_2m']})
      df['datetime'] = pd.to_datetime(df['datetime'])
      return df.set_index('datetime')
  ```

- [ ] If you go this route: fetch for your data's date range, merge on datetime index, add `lag_temp_24h` as well
- [ ] Skip if it's causing delays — it's additive, not required

### 4.7 Assemble everything

- [ ] Write `build_features()` at the bottom of `src/features.py`:

  ```python
  from src.config import DATA_PROC, TARGET_COL

  def build_features() -> pd.DataFrame:
      df = pd.read_parquet(DATA_PROC / 'hourly_demand.parquet')
      df = add_calendar_features(df)
      df = add_lag_features(df)
      df = add_rolling_features(df)
      df = add_fourier_features(df)
      df = add_holiday_features(df)
      df = df.dropna()
      return df

  if __name__ == '__main__':
      df = build_features()
      df.to_parquet(DATA_PROC / 'features.parquet')
      print(f"Feature matrix: {df.shape}")
      print(df.columns.tolist())
  ```

- [ ] Run: `python src/features.py`
- [ ] Verify: `print(df.shape)` — how many rows dropped by `dropna()`? Should be ~336 (2 weeks of lags)
- [ ] Run `make features` — same result
- [ ] Commit: `"feat: feature engineering pipeline"`

### 4.8 Quick sanity check — feature importance

- [ ] Do a rough LightGBM fit just to check features make sense:

  ```python
  import lightgbm as lgb
  import pandas as pd

  df = pd.read_parquet('data/processed/features.parquet')
  TARGET = 'demand_mw'
  feature_cols = [c for c in df.columns if c != TARGET]

  # Simple train/val split (no CV yet, just for checking)
  split = int(len(df) * 0.8)
  X_tr, y_tr = df[feature_cols].iloc[:split], df[TARGET].iloc[:split]
  X_val, y_val = df[feature_cols].iloc[split:], df[TARGET].iloc[split:]

  m = lgb.LGBMRegressor(n_estimators=200, verbose=-1).fit(X_tr, y_tr)
  lgb.plot_importance(m, max_num_features=15, figsize=(8, 6))
  ```

- [ ] Top features should be: `lag_168h`, `lag_24h`, `rolling_mean_24h`, `lag_1h`
- [ ] If calendar features dominate over lag features, something is wrong with the lags

---

## Phase 5 · Modeling with MLflow

_Goal: 3+ logged runs in MLflow. LightGBM visibly beats the naive baseline._

### 5.1 Start MLflow

- [ ] In a separate terminal (keep it running):
  ```bash
  source .venv/bin/activate
  mlflow ui --port 5000
  ```
- [ ] Open `http://localhost:5000` — you should see the MLflow UI
- [ ] At the top of your training script:
  ```python
  import mlflow
  mlflow.set_tracking_uri("http://localhost:5000")
  mlflow.set_experiment("energy-forecast")
  ```

### 5.2 Understand TimeSeriesSplit (read, don't skip)

- [ ] Understand why this is different:

  ```python
  from sklearn.model_selection import TimeSeriesSplit
  from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

  tss = TimeSeriesSplit(n_splits=5, test_size=24 * 30)  # 5 folds, 30-day test window each
  ```

- [ ] The key: each fold's validation data is always strictly _after_ the training data. No shuffling. No future data leaking into training.
- [ ] Visualize what the folds look like:

  ```python
  df = pd.read_parquet('data/processed/features.parquet')
  TARGET = 'demand_mw'
  feature_cols = [c for c in df.columns if c != TARGET]
  X, y = df[feature_cols], df[TARGET]

  for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
      print(f"Fold {fold}: train={df.index[tr_idx[0]].date()}→{df.index[tr_idx[-1]].date()} | val={df.index[val_idx[0]].date()}→{df.index[val_idx[-1]].date()}")
  ```

### 5.3 Write a reusable eval helper

- [ ] In `src/evaluate.py`:

  ```python
  import numpy as np
  from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

  def regression_metrics(y_true, y_pred) -> dict:
      return {
          "mape": mean_absolute_percentage_error(y_true, y_pred),
          "mae":  mean_absolute_error(y_true, y_pred),
          "rmse": np.sqrt(((y_true - y_pred) ** 2).mean()),
      }
  ```

### 5.4 Write `src/train.py`

- [ ] Build the full training script with 3 models:

```python
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from src.config import DATA_PROC, MLFLOW_URI, EXPERIMENT_NAME, TARGET_COL
from src.evaluate import regression_metrics

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_parquet(DATA_PROC / 'features.parquet')
feature_cols = [c for c in df.columns if c != TARGET_COL]
X, y = df[feature_cols], df[TARGET_COL]

tss = TimeSeriesSplit(n_splits=3, test_size=24 * 30)

# --- Run 1: Naive baseline ---
with mlflow.start_run(run_name="naive-baseline"):
    mlflow.log_param("model", "naive_lag_168h")
    fold_metrics = []
    for tr_idx, val_idx in tss.split(X):
        y_val = y.iloc[val_idx]
        naive_pred = X.iloc[val_idx]['lag_168h']
        fold_metrics.append(regression_metrics(y_val, naive_pred))
    avg = {k: sum(d[k] for d in fold_metrics) / len(fold_metrics) for k in fold_metrics[0]}
    mlflow.log_metrics({f"cv_{k}": v for k, v in avg.items()})
    print(f"[Naive] MAPE={avg['mape']:.2%}  MAE={avg['mae']:.0f}")

# --- Run 2: Linear Regression ---
with mlflow.start_run(run_name="linear-regression"):
    mlflow.log_param("model", "linear_regression")
    fold_metrics = []
    for tr_idx, val_idx in tss.split(X):
        m = LinearRegression().fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = m.predict(X.iloc[val_idx])
        fold_metrics.append(regression_metrics(y.iloc[val_idx], pred))
    avg = {k: sum(d[k] for d in fold_metrics) / len(fold_metrics) for k in fold_metrics[0]}
    mlflow.log_metrics({f"cv_{k}": v for k, v in avg.items()})
    print(f"[LinReg] MAPE={avg['mape']:.2%}  MAE={avg['mae']:.0f}")

# --- Run 3: LightGBM ---
lgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
}

with mlflow.start_run(run_name="lightgbm-v1"):
    mlflow.log_params(lgb_params)
    fold_metrics = []
    model = None
    for tr_idx, val_idx in tss.split(X):
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        )
        pred = model.predict(X.iloc[val_idx])
        fold_metrics.append(regression_metrics(y.iloc[val_idx], pred))
    avg = {k: sum(d[k] for d in fold_metrics) / len(fold_metrics) for k in fold_metrics[0]}
    mlflow.log_metrics({f"cv_{k}": v for k, v in avg.items()})
    mlflow.lightgbm.log_model(model, "lgbm_model")
    print(f"[LightGBM] MAPE={avg['mape']:.2%}  MAE={avg['mae']:.0f}")
```

- [ ] Run: `python src/train.py`
- [ ] Open `http://localhost:5000` — you should see 3 runs in the `energy-forecast` experiment
- [ ] LightGBM should beat naive. If it doesn't, your features need more work.
- [ ] Run `make train` — same result
- [ ] Commit: `"feat: training pipeline with MLflow, 3 models"`

### 5.5 Optional: XGBoost comparison run

- [ ] Add a 4th run with XGBoost, same structure as LightGBM run:
  ```python
  import xgboost as xgb
  # params: same shape as lightgbm, just swap the model class
  model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, verbosity=0)
  ```
- [ ] Log it identically. Compare in MLflow UI.

### 5.6 Hyperparameter experiment (log each change as a new run)

- [ ] Try: `num_leaves=63` → new run `lightgbm-v2`
- [ ] Try: adding `colsample_bytree=0.8, subsample=0.8` → new run `lightgbm-v3`
- [ ] Don't over-tune. The goal is to show the experimentation workflow, not perfect accuracy.
- [ ] Or: enable autolog: `mlflow.lightgbm.autolog()` before training — logs everything automatically

---

## Phase 6 · 48-Hour Forecast Pipeline

_Goal: `make forecast` produces a 48h prediction._

### 6.1 Understand multi-step forecasting

There are two strategies:

- **Recursive:** one model, feeds its own predictions as future lags. Simpler, errors accumulate over horizon.
- **Direct (MIMO):** one model per step (48 models). More accurate for longer horizons, more expensive.

For this project: start recursive. It's enough.

### 6.2 Set up skforecast

- [ ] Read the skforecast quickstart (30 min): `skforecast.org/docs/`
- [ ] Create `notebooks/02_skforecast_test.ipynb` to experiment before putting in `src/`

### 6.3 Recursive forecaster

- [ ] In the notebook first:

  ```python
  import pandas as pd
  from skforecast.ForecasterAutoreg import ForecasterAutoreg
  from lightgbm import LGBMRegressor
  from src.config import DATA_PROC, TARGET_COL, FORECAST_HORIZON

  df = pd.read_parquet(DATA_PROC / 'hourly_demand.parquet')

  forecaster = ForecasterAutoreg(
      regressor=LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1),
      lags=[1, 24, 48, 168, 336]
  )

  # Train on all but last 60 days
  train_end = df.index[-1] - pd.Timedelta(days=60)
  forecaster.fit(y=df.loc[:train_end, TARGET_COL])

  # Predict next 48 hours
  predictions = forecaster.predict(steps=FORECAST_HORIZON)
  print(predictions)
  ```

### 6.4 Backtesting (proper evaluation)

- [ ] This is the right way to evaluate multi-step forecasting:

  ```python
  from skforecast.model_selection import backtesting_forecaster
  from sklearn.metrics import mean_absolute_percentage_error

  metric, predictions_df = backtesting_forecaster(
      forecaster=forecaster,
      y=df[TARGET_COL],
      initial_train_size=len(df) - 60 * 24,  # hold out last 60 days
      steps=FORECAST_HORIZON,
      refit=False,
      metric='mean_absolute_percentage_error',
      verbose=True,
  )

  print(f"Backtesting MAPE (48h horizon): {metric:.2%}")
  print(f"Naive baseline MAPE was: X.X%")   # fill in your Phase 2 number
  ```

- [ ] This is your **headline result**. Write it down.

### 6.5 Write `src/forecast.py`

- [ ] Script that produces a forecast file:

  ```python
  import pandas as pd
  from skforecast.ForecasterAutoreg import ForecasterAutoreg
  from lightgbm import LGBMRegressor
  from src.config import DATA_PROC, TARGET_COL, FORECAST_HORIZON

  def run_forecast():
      df = pd.read_parquet(DATA_PROC / 'hourly_demand.parquet')
      forecaster = ForecasterAutoreg(
          regressor=LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1),
          lags=[1, 24, 48, 168, 336]
      )
      forecaster.fit(y=df[TARGET_COL])
      predictions = forecaster.predict(steps=FORECAST_HORIZON)

      forecast_df = pd.DataFrame({
          'datetime': predictions.index,
          'forecast_mw': predictions.values
      })
      forecast_df.to_csv(DATA_PROC / 'forecast.csv', index=False)
      print(f"Saved {len(forecast_df)}-step forecast to data/processed/forecast.csv")
      return forecast_df

  if __name__ == '__main__':
      run_forecast()
  ```

- [ ] Run: `python src/forecast.py` — produces `data/processed/forecast.csv`
- [ ] Run: `make forecast` — same
- [ ] Commit: `"feat: 48h multi-step forecast pipeline"`

---

## Phase 7 · Error Analysis

_Goal: understand when and why the model is wrong. This is what separates real work from a notebook dump._

### 7.1 Create `notebooks/03_error_analysis.ipynb`

- [ ] Load predictions from backtesting and actual values, compute residuals:
  ```python
  residuals = predictions_df['pred'] - predictions_df['y']
  ```
- [ ] Plot: actual vs predicted for one representative week
- [ ] Plot: residuals over time — do errors cluster at certain periods?
- [ ] Plot: MAPE by hour-of-day — which hours are hardest?
  ```python
  hourly_mape = predictions_df.groupby(predictions_df.index.hour).apply(
      lambda g: mean_absolute_percentage_error(g['y'], g['pred'])
  )
  hourly_mape.plot(kind='bar', title='MAPE by hour of day')
  ```
- [ ] Plot: MAPE for holidays vs non-holidays — is the model worse on holidays?

### 7.2 Record findings

- [ ] Write 3–5 bullet observations in the notebook. Example things to look for:
  - "Model struggles most during early morning ramp-up (4am–7am) — MAPE 12% vs 5% average"
  - "Holiday predictions are 2× worse than regular days"
  - "Errors higher in summer (Jun–Aug) — likely due to temperature variance not captured"
- [ ] These observations become your "Future Work" section in the college report

---

## Phase 8 · Results Table

Fill this in as you complete runs. This is what you show everyone.

| Model                | CV MAPE | CV MAE (MW) | Notes              |
| -------------------- | ------- | ----------- | ------------------ |
| Naive (lag_168h)     |         |             | Baseline           |
| Linear Regression    |         |             |                    |
| LightGBM v1          |         |             | 500 trees, default |
| LightGBM v2          |         |             | Tuned leaves       |
| XGBoost              |         |             | Optional           |
| **skforecast (48h)** |         |             | Final pipeline     |

> Industry reference: commercial energy forecasts run 2–4% MAPE. A student project at 5–8% is solid. Above 15% means something went wrong in features or CV.

---

## Phase 9 · Streamlit Dashboard

_Goal: something you can demo. One screen, one story._

### 9.1 Create `app.py`

- [ ] Minimal but complete:

  ```python
  import streamlit as st
  import pandas as pd
  import plotly.express as px
  from src.config import DATA_PROC, TARGET_COL

  st.set_page_config(page_title="Energy Demand Forecast", layout="wide")
  st.title("48-Hour Load Forecast")
  st.caption("LightGBM model — trained on historical hourly demand")

  # Load data
  actual   = pd.read_parquet(DATA_PROC / 'hourly_demand.parquet')
  forecast = pd.read_csv(DATA_PROC / 'forecast.csv', parse_dates=['datetime'], index_col='datetime')

  # Plot last 7 days actual + 48h forecast
  recent = actual.tail(7 * 24)
  fig = px.line(title='Recent demand + 48h forecast')
  fig.add_scatter(x=recent.index, y=recent[TARGET_COL], name='Actual', line_color='#378ADD')
  fig.add_scatter(x=forecast.index, y=forecast['forecast_mw'], name='Forecast',
                  line=dict(color='#EF9F27', dash='dash'))
  st.plotly_chart(fig, use_container_width=True)

  # Metrics
  col1, col2, col3 = st.columns(3)
  col1.metric("Model", "LightGBM")
  col2.metric("48h MAPE", "X.X%")    # fill in your number
  col3.metric("Naive baseline", "X.X%")

  st.markdown("---")
  st.markdown("**Data source:** [your source]  |  **Forecast horizon:** 48 hours")
  ```

- [ ] Run: `streamlit run app.py` — does it start without errors?
- [ ] Test: refresh, does it reload correctly?

### 9.2 Polish

- [ ] Replace any hardcoded numbers (MAPE etc.) with computed values
- [ ] Add a feature importance chart as a second section (optional)
- [ ] Make sure no absolute paths are hardcoded (`/home/jitendra/...`)
- [ ] Run: `make dashboard` — should launch correctly

---

## Phase 10 · Final Cleanup & Report

_Goal: repo is presentable, report is honest and short._

### 10.1 Code review

- [ ] Run the full pipeline from scratch on your machine:
  ```bash
  make data
  make features
  make train
  make forecast
  ```
- [ ] Does it work without errors? Fix anything that breaks.
- [ ] Delete: dead notebook cells, `test_asdf.py`, any file you don't understand why it's there
- [ ] Check: no absolute paths in any `src/` file
- [ ] Check: `data/raw/` is in `.gitignore` and not committed

### 10.2 Git history

- [ ] `git log --oneline` — do commits tell a story?
- [ ] Tag the final version: `git tag v1.0 && git push origin v1.0`

### 10.3 README.md

- [ ] Structure:

  ```markdown
  # Energy Demand Forecasting

  48-hour load forecast using LightGBM. Trained on [dataset]. Backtested MAPE: X.X%.

  ## Setup

  pip install -r requirements.txt

  ## Run

  make data # clean raw data
  make features # build feature matrix
  make train # train + log to MLflow (start: mlflow ui --port 5000)
  make forecast # generate 48h forecast
  make dashboard # launch Streamlit app

  ## Results

  [paste your results table here]

  ## Data

  [source, date range, resolution, any notes]
  ```

- [ ] Add one screenshot of the Streamlit app
- [ ] Add one screenshot of the MLflow experiment page

### 10.4 College report

Suggested structure (8–10 pages max):

- [ ] **Section 1 — Introduction** (1 page): why load forecasting matters, India's grid context, problem statement
- [ ] **Section 2 — Data** (1 page): source, resolution, date range, preprocessing decisions and why, key patterns found in EDA
- [ ] **Section 3 — Methodology** (2–3 pages):
  - Feature engineering: explain lag features, Fourier encoding, holiday flags in plain language
  - Why `TimeSeriesSplit` over random split
  - Model selection rationale
  - MLflow for experiment tracking
- [ ] **Section 4 — Results** (1–2 pages): results table, one actual-vs-predicted plot, error analysis observations
- [ ] **Section 5 — Conclusion** (0.5 page): what MAPE you achieved, what would improve it (temperature data, ensemble, regional granularity)
- [ ] Keep it honest. A 7% MAPE with a clear methodology is better than a claimed 2% with a data-leaking random split.

---

## Optional Extensions

_Only if the core project is fully done and working. These change the tier of the project._

- [ ] **Weather data**: pull historical temperature from Open-Meteo (free, no key) and add as an exogenous feature. Expect 1–2% MAPE improvement.
- [ ] **Probabilistic forecasting**: instead of a point forecast, output 10th/90th percentile bounds. skforecast supports this natively. Useful for grid operators.
- [ ] **ARIMA comparison**: train a SARIMA model via `statsmodels`, compare to LightGBM. Shows you understand the tradeoff. Usually ~2× worse MAPE than LightGBM.
- [ ] **FastAPI endpoint**: `POST /forecast` → returns 48h prediction as JSON. Simple wrapper around `src/forecast.py`. Makes the project an actual service.
- [ ] **Regional breakdown**: if your dataset has multiple regions or states, compare forecasting difficulty across them. India's five grid regions have very different demand patterns.
- [ ] **GitHub Actions**: CI that runs `make features && make train` on push. Shows DevOps awareness. 30 min to set up.

---

## Checkpoints (verify these, don't skip)

| After... | Verify                                                                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 0  | `make data` runs, repo is on GitHub                                                                                                                     |
| Phase 1  | Raw file in `data/raw/`, you know its schema                                                                                                            |
| Phase 2  | You know the naive MAPE. You've seen the daily seasonal pattern in a plot.                                                                              |
| Phase 3  | `make data` produces `data/processed/hourly_demand.parquet` cleanly                                                                                     |
| Phase 4  | `make features` produces feature matrix. Top LightGBM importance features are lag/rolling, not calendar.                                                |
| Phase 5  | MLflow has 3+ runs. LightGBM MAPE < naive MAPE. You can explain why `TimeSeriesSplit`.                                                                  |
| Phase 6  | `make forecast` outputs `data/processed/forecast.csv`                                                                                                   |
| End      | `git clone → pip install -r requirements.txt → make data → make features → make train → make forecast → streamlit run app.py` all work on a fresh clone |
