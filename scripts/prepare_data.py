"""
Data Ingestion & Feature Engineering Runner.

Executes data generation, cleaning, and extraction of hourly & daily feature matrices.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import SyntheticGridLoader
from src.data.clean import DataCleaner
from src.features.pipeline import build_and_save_features


def main():
    print("=" * 60)
    print("1. INGESTING & CLEANING DATA")
    print("=" * 60)
    loader = SyntheticGridLoader()
    df_raw = loader.load(force_regenerate=False)
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
    cleaner.save_processed(df_clean, filename="hourly_demand.parquet")
    print(f"Data cleaned successfully ({len(df_clean):,} rows). Provenance: {loader.data_provenance}")

    print("\n" + "=" * 60)
    print("2. BUILDING FEATURE MATRICES (HOURLY & DAILY)")
    print("=" * 60)
    build_and_save_features()
    print("Feature matrices ready for model training.")


if __name__ == "__main__":
    main()
