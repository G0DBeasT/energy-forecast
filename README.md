# ⚡ Energy Grid Load Forecasting System

A production-grade, multi-horizon electricity demand forecasting application built with **Python**, **LightGBM**, **XGBoost**, **MLflow**, **scikit-learn**, **Open-Meteo NWP APIs**, and **Streamlit**.

---

## 📌 1. Project Overview & Multi-Horizon Products

This system delivers three clearly separated operational forecasting products designed for power transmission operators, utility resource planners, and grid dispatch engineers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 OPERATIONAL HORIZONS                                   │
├─────────────────────────┬───────────────────────────────┬──────────────────────────────┤
│ 🌅 1. NEXT-DAY FORECAST │ 📅 2. NEXT-WEEK FORECAST      │ 🗓️ 3. NEXT-MONTH FORECAST    │
├─────────────────────────┼───────────────────────────────┼──────────────────────────────┤
│ • Horizon: Next 24 Hours│ • Horizon: Next 7 Days (168h) │ • Horizon: Next 30 Days (720h│
│ • Resolution: Hourly    │ • Resolution: Daily & Hourly  │ • Resolution: Daily          │
│ • Model: Hourly LightGBM│ • Model: Hourly LightGBM Agg. │ • Model: Dedicated Daily LGBM│
│ • Key Use: Unit dispatch│ • Key Use: Weekly reserves,   │ • Key Use: Long-range hydro /│
│   peaking plants, ramp  │   thermal unit commitment,    │   fuel resource budgeting,   │
│   rate management       │   weekend maintenance windows │   monthly energy volume (GWh)│
│ • Weather: High-res NWP │ • Weather: Multi-day NWP      │ • Weather: NWP (1-14d) +     │
│   forecast              │   forecast                    │   Seasonal Climatology (15-30│
└─────────────────────────┴───────────────────────────────┴──────────────────────────────┘
```

---

## 🛡️ 2. Data Provenance & Transparency

> [!IMPORTANT]
> **Data Provenance Disclosure**:
> The default development dataset is **semi-synthetic / simulated**:
> - **Meteorological Data**: Real historical reanalysis & live forecast data fetched from the **Open-Meteo API** for the regional grid ($28.6139^\circ\text{N}, 77.2090^\circ\text{E}$).
> - **Grid Demand Data**: Synthesized using a physical non-linear grid formula combining base load ($4500\text{ MW}$), macroeconomic annual demand trend, non-linear cooling ($(\max(T-22, 0))^{1.35} \times 65$), non-linear heating ($(\max(16-T, 0))^{1.2} \times 35$), diurnal dual peaks, weekend industrial dips, and Gaussian stochastic fluctuations.
>
> **Pluggable Real-World Ingestion**:  
> The system implements a clean `BaseDataLoader` abstraction. Enterprise users can load empirical substation CSV recordings (e.g., PJM, ERCOT, ENTSO-E, POSOCO/MERIT India) seamlessly via `CSVGridLoader`.

---

## 🛠️ 3. Feature Engineering Architecture

The pipeline extracts **50+ leakage-safe features** across hourly and daily resolutions:

1. **Temporal & Calendar Features**:
   - `hour` (0–23), `day_of_week` (0–6), `day_of_month` (1–31), `day_of_year`, `week_of_year`, `month`, `quarter`, `year`, `season`, `annual_position`.
   - `is_peak_hour` (morning 9–11, evening 18–22), `is_night_valley` (01:00–05:00).
   - `is_weekend`, `is_working_day`.
2. **Public Holiday Indicators & Proximity**:
   - `is_holiday`, `is_pre_holiday`, `is_post_holiday`.
   - `days_until_holiday`, `days_since_holiday` (continuous distance to nearest public holiday).
3. **Continuous Cyclical Fourier Encodings**:
   - Daily 24h: $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$, $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$
   - Weekly 7d: $\sin\left(\frac{2\pi \cdot \text{day}}{7}\right)$, $\cos\left(\frac{2\pi \cdot \text{day}}{7}\right)$
   - Annual 365.25d: $\sin\left(\frac{2\pi \cdot \text{doy}}{365.25}\right)$, $\cos\left(\frac{2\pi \cdot \text{doy}}{365.25}\right)$
4. **Historical Demand Lags**:
   - Short Lags: `lag_1h`, `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`
   - Daily Lags: `lag_24h` (previous day), `lag_48h`, `lag_72h`
   - Weekly & Monthly Lags: `lag_168h` (1 week), `lag_336h` (2 weeks), `lag_720h` (30 days), `lag_8760h` (1 year)
   - Daily Model Lags: `lag_1d`, `lag_2d`, `lag_3d`, `lag_7d`, `lag_14d`, `lag_30d`, `lag_365d`
5. **Leakage-Safe Rolling Statistics** (Shifted by 1 step):
   - Hourly: 24h (mean, std, min, max), 7d/168h (mean, std, min, max), 30d/720h (mean, std, min, max).
   - Daily: 7d (mean, std, min, max), 14d (mean, std), 30d (mean, std, min, max).
   - Dynamic Load Ratio: `demand_ratio_to_24h_mean = lag_1h / (rolling_mean_24h + 1e-5)`.
6. **Thermodynamic & Weather Interactions**:
   - Temperature $T$ ($^\circ\text{C}$), Relative Humidity ($\%$), Apparent Heat Index ($^\circ\text{C}$).
   - Cooling Degree Days ($\text{CDD} = \max(T - 18.33, 0)$), Heating Degree Days ($\text{HDD} = \max(18.33 - T, 0)$).
   - Temperature Lags & Rolling: `temp_lag_24h`, `temp_rolling_mean_24h`, `temp_rolling_mean_7d`, `humidity_rolling_mean`.
   - Weather Provenance Metadata: `weather_source` (`nwp_forecast`, `climatology`, `fallback`).

---

## 🏆 4. Model Validation Leaderboard (5-Fold Chronological TimeSeriesSplit)

| Model Architecture | Horizon Target | CV MAPE | CV MAE (MW) | RMSE (MW) | $R^2$ Score | WAPE | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hourly Naive (`lag_168h`)** | Hourly | **6.28%** | 415.5 MW | 527.4 MW | 0.7248 | 6.28% | Seasonal Baseline |
| **Hourly Ridge Linear** | Hourly | **2.39%** | 145.3 MW | 182.7 MW | 0.9507 | 2.39% | Linear Regularized Baseline |
| **Hourly LightGBM (Default)** | Hourly | **2.10%** | 131.2 MW | 166.9 MW | 0.9623 | 2.10% | Default Gradient Booster |
| **Hourly LightGBM (Tuned)** | Hourly | **2.09%** | **130.4 MW** | **165.9 MW** | **0.9630** | **2.09%** | **Champion Hourly Model** |
| **Hourly XGBoost Regressor** | Hourly | **2.11%** | 131.7 MW | 167.4 MW | 0.9618 | 2.11% | Scalable Tree Booster |
| **Daily Naive (`lag_7d`)** | Daily | **5.42%** | 358.1 MW | 446.2 MW | 0.7812 | 5.42% | Daily Seasonal Baseline |
| **Daily Ridge Linear** | Daily | **1.94%** | 118.6 MW | 149.2 MW | 0.9680 | 1.94% | Daily Linear Model |
| **Dedicated Daily LightGBM** | Daily | **1.72%** | **104.8 MW** | **131.5 MW** | **0.9745** | **1.72%** | **Champion Daily Model** |

---

## 📂 5. Architecture & Project Tree

```
energy-forecast/
├── app.py                     # Streamlit Multi-Horizon Interactive Web Dashboard
├── Makefile                   # Production task automation
├── pyproject.toml             # Project dependency specifications
├── requirements.txt           # Verified package requirements
├── README.md                  # System documentation & engineering report
├── data/
│   ├── raw/                   # Raw grid CSVs & weather logs
│   └── processed/             # Cleaned parquets, models, multi-horizon forecast CSVs
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── run_eda.py             # Diagnostic EDA script
│   ├── 03_error_analysis.ipynb# Holdout Error & Residual Analysis
│   └── run_error_analysis.py  # Error Analysis script
├── reports/                   # 12 Generated high-resolution diagnostic charts
├── tests/                     # Comprehensive pytest test suite
│   ├── test_data.py           # Ingestion, provenance, cleaning tests
│   ├── test_weather.py        # Weather API & climatology fallback tests
│   ├── test_features.py       # Feature pipeline & leakage-safety tests
│   ├── test_models.py         # Evaluation & baseline tests
│   └── test_forecast.py       # Dynamic uncertainty & forecasting tests
└── src/
    ├── __init__.py            # Modular exports
    ├── config.py              # Centralized paths, coordinates, and horizon constants
    ├── data/
    │   ├── base.py            # BaseDataLoader abstract base class
    │   ├── synthetic.py       # Physics-grounded synthetic generator (clearly labeled)
    │   ├── csv_loader.py      # Real-world external CSV dataset loader
    │   └── clean.py           # Strict resampling, imputation, and outlier clipping
    ├── weather/
    │   ├── client.py          # Open-Meteo live NWP forecast client
    │   └── climatology.py     # Seasonal climatology profile fallback
    ├── features/
    │   └── pipeline.py        # Unified FeatureEngineer (hourly & daily transformations)
    ├── models/
    │   ├── evaluate.py        # Multi-horizon regression metrics (MAPE, MAE, RMSE, R2, WAPE)
    │   ├── baselines.py       # Seasonal Naive & Linear baseline models
    │   ├── registry.py        # Model serialization & metadata packaging
    │   └── train.py           # 5-fold TimeSeriesSplit CV training pipeline
    └── forecast/
        ├── engine.py          # MultiHorizonForecastEngine (Next-Day, Next-Week, Next-Month)
        └── uncertainty.py     # Dynamic lead-time uncertainty estimator
```

---

## 🚀 6. Quickstart & Command Reference

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Pipeline End-to-End

```bash
# Ingest and clean raw data
make data

# Build leakage-safe hourly and daily feature matrices
make features

# Run 5-fold TimeSeriesSplit CV training across candidate models
make train

# Generate Next-Day, Next-Week, and Next-Month forecasts
make forecast

# Execute unit and integration test suite
make test

# Launch the Streamlit interactive dashboard
make dashboard
```
