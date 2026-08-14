# Energy Grid Load Forecasting System

A complete, production-grade 48-Hour Energy Load Forecasting pipeline built with Python, LightGBM, XGBoost, MLflow, skforecast, and Streamlit.

---

## 📌 Project Overview

This project forecasts 48-hour electricity demand (MW) using 3.5 years of hourly grid load data (30,648 observations) combined with historical & forecast weather data (Temperature °C and Relative Humidity %). 

### 🏆 Key Validation Results (Phase 8 Benchmark)

| Model | CV MAPE | CV MAE (MW) | RMSE (MW) | $R^2$ Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (`lag_168h`)** | **6.28%** | 415.5 MW | 527.4 MW | 0.7248 | Benchmark |
| **Linear Regression** | **2.39%** | 145.3 MW | 182.7 MW | 0.9507 | Baseline Linear Model |
| **LightGBM v1 (Default)** | **2.10%** | 131.2 MW | 166.9 MW | 0.9623 | Gradient Boosting |
| **LightGBM v2 (Tuned)** | **2.09%** | **130.4 MW** | **165.9 MW** | **0.9630** | **Best Production Model** |
| **XGBoost Regressor** | **2.11%** | 131.7 MW | 167.4 MW | 0.9618 | Gradient Boosting |

- **Holdout Test Set Accuracy**: **1.56% MAPE** (MAE = 109.9 MW).
- **Baseline Improvement**: LightGBM achieves over **3.0x reduction in forecast error** compared to the naive baseline (same hour last week).

---

## 🛠️ Feature Engineering Strategy

The feature matrix includes **45 high-impact features**:

1. **Temporal & Calendar Features**:
   - `hour` (0–23), `day_of_week` (0–6), `day_of_month` (1–31), `month` (1–12), `quarter` (1–4), `year`, `day_of_year`, `is_weekend`.
2. **Cyclical Fourier Encodings**:
   - Daily 24h: `sin_hour`, `cos_hour`
   - Weekly 7d: `sin_week`, `cos_week`
   - Annual 365.25d: `sin_year`, `cos_year`
3. **Public Holiday Indicators**:
   - `is_holiday`, `is_pre_holiday`, `is_post_holiday` (using `holidays` library).
4. **Historical Demand Lags**:
   - Short Lags: `lag_1h`, `lag_2h`, `lag_3h`
   - Daily Lags: `lag_24h` (previous day), `lag_48h`
   - Weekly Lags: `lag_168h` (previous week), `lag_336h`
   - Monthly Lags: `lag_720h` (30 days ago)
   - Annual Lags: `lag_8760h` (365 days ago)
5. **Rolling Demand Aggregations** (Shifted to prevent data leakage):
   - 24-hour: `rolling_mean_24h`, `rolling_std_24h`, `rolling_min_24h`, `rolling_max_24h`
   - 168-hour (1 week): `rolling_mean_168h`, `rolling_std_168h`, `rolling_min_168h`, `rolling_max_168h`
   - 720-hour (30 days): `rolling_mean_720h`, `rolling_std_720h`
6. **Meteorological & Weather Features**:
   - Raw weather: `temp_c` (Temperature °C), `relative_humidity` (%)
   - Cooling Degree Days (`cdd`): $\max(\text{temp} - 18.3, 0)$
   - Heating Degree Days (`hdd`): $\max(18.3 - \text{temp}, 0)$
   - Apparent Heat Index (`heat_index_c`)
   - Weather Lags & Rolling: `temp_lag_24h`, `temp_rolling_mean_24h`, `temp_rolling_mean_168h`, `humidity_rolling_mean_24h`.

---

## 📂 Project Architecture

```
energy-forecast/
├── app.py                     # Streamlit Interactive Web Dashboard
├── Makefile                   # Automation Makefile target runners
├── pyproject.toml             # Project dependencies configuration
├── requirements.txt           # Python environment requirements
├── todo.md / todo_broad.md    # Master Phase & Sub-phase project checklist
├── data/
│   ├── raw/                   # Raw hourly demand & Open-Meteo weather CSVs
│   └── processed/             # Parquet datasets, trained model pkl, 48h forecast CSV
├── notebooks/
│   ├── 01_eda.ipynb           # Phase 2 Exploratory Data Analysis Notebook
│   ├── run_eda.py             # Phase 2 Script generating report diagnostic plots
│   ├── 03_error_analysis.ipynb# Phase 7 Error Analysis Notebook
│   └── run_error_analysis.py  # Phase 7 Script evaluating residual diagnostics
├── reports/                   # 12 Generated high-resolution diagnostic charts
└── src/
    ├── __init__.py
    ├── config.py              # Centralized paths and hyperparameters
    ├── data_fetch.py          # Data acquisition & Open-Meteo API weather fetcher
    ├── data.py                # Phase 3 Data cleaning & missing value imputation
    ├── features.py            # Phase 4 Feature engineering pipeline
    ├── train.py               # Phase 5 MLflow training & TimeSeriesSplit CV
    ├── evaluate.py            # Metric calculation helper (MAPE, MAE, RMSE, R2)
    └── forecast.py            # Phase 6 48-Hour multi-step forecast generator
```

---

## 🚀 Quickstart & Pipeline Commands

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Pipeline via Makefile

```bash
make data       # Clean raw demand & weather data -> data/processed/hourly_demand.parquet
make features   # Build 45 ML features -> data/processed/features.parquet
make train      # Train 5 models with TimeSeriesSplit CV & MLflow logging
make forecast   # Generate 48-hour forecast -> data/processed/forecast.csv
make dashboard  # Launch interactive Streamlit Web App
```

---

## 🖥️ Streamlit Web Dashboard Features

Run `make dashboard` or `streamlit run app.py` to open the web app:

1. **Interactive 48h Forecast Plot**: Includes past demand actuals + 48h forecast line with 90% confidence bounds.
2. **KPI Header Cards**: Displays production model status, MAPE, MAE, and baseline comparison.
3. **Weather Drivers Timeline**: Synchronized view of temperature, humidity, and load fluctuations.
4. **Feature Importance Chart**: Top 12 features driving the LightGBM predictions.
5. **CSV Export**: One-click download button for 48-hour forecast values.
