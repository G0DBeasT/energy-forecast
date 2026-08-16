"""
Model Training & Selection Runner.

Executes TimeSeries cross-validation across all models and saves champion models.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train import run_training_pipeline


def main():
    print("=" * 60)
    print("EXECUTING MODEL TRAINING & VALIDATION PIPELINE")
    print("=" * 60)
    run_training_pipeline()


if __name__ == "__main__":
    main()
