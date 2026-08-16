"""
Centralized Configuration for Energy Grid Load Forecasting System.

Defines project paths, column identifiers, forecasting horizon constants,
location coordinates, and default hyperparameters.
"""

from pathlib import Path

# Base Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROC = DATA_DIR / "processed"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROC, REPORTS, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Column Identifiers
TARGET_COL = "demand_mw"
DATE_COL = "datetime"
TEMP_COL = "temp_c"
HUMIDITY_COL = "relative_humidity"
WEATHER_SOURCE_COL = "weather_source"

# Time Frequency
FREQ = "1h"

# Multi-Horizon Forecasting Constants (in hours)
HORIZON_DAY = 24       # Next-Day (24h hourly dispatch view)
HORIZON_WEEK = 168     # Next-Week (7 days / 168h weekly planning view)
HORIZON_MONTH = 720    # Next-Month (30 days / 720h resource budgeting view)

HORIZONS = {
    "next_day": HORIZON_DAY,
    "next_week": HORIZON_WEEK,
    "next_month": HORIZON_MONTH,
}

# Legacy default forecast horizon for backwards compatibility
FORECAST_HORIZON = HORIZON_WEEK

# Meteorological & Location Coordinates (Default: New Delhi, India)
LATITUDE = 28.6139
LONGITUDE = 77.2090
TIMEZONE = "Asia/Kolkata"
COUNTRY_HOLIDAY_CODE = "IN"

# Base Temperature for Degree Days calculation (18.33°C / 65°F standard ASHRAE base)
BASE_TEMP_C = 18.33

# MLflow Experiment Tracking Configuration
MLFLOW_DB_PATH = ROOT / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "energy-forecast"

# Random Seed for Reproducibility
RANDOM_SEED = 42
