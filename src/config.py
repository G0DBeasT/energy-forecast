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
