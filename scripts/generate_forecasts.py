"""
Multi-Horizon Forecast Generator Runner.

Generates Next-Day (24h), Next-Week (168h), and Next-Month (30d) forecasts.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecast.engine import MultiHorizonForecastEngine, run_forecast_pipeline


def main():
    print("=" * 60)
    print("GENERATING MULTI-HORIZON OPERATIONAL FORECASTS")
    print("=" * 60)
    run_forecast_pipeline()


if __name__ == "__main__":
    main()
