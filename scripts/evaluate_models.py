"""
Model Evaluation & Selection Metrics Report Generator.

Loads validation metrics across all candidate models and persists structured JSON
and markdown performance artifacts under reports/metrics/.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import DATA_PROC, REPORTS


def main():
    metrics_path = DATA_PROC / "metrics_summary.parquet"
    if not metrics_path.exists():
        print(f"Metrics summary not found at {metrics_path}. Run training pipeline first.")
        return

    df_metrics = pd.read_parquet(metrics_path)
    print("=" * 70)
    print("STRUCTURED MODEL VALIDATION BENCHMARK (ALL HORIZONS)")
    print("=" * 70)
    print(df_metrics.to_string())

    out_metrics_dir = REPORTS / "metrics"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save structured JSON
    metrics_dict = df_metrics.to_dict(orient="index")
    json_path = out_metrics_dir / "model_selection_summary.json"
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save formatted markdown report
    md_path = out_metrics_dir / "model_benchmark_report.md"
    with open(md_path, "w") as f:
        f.write("# Model Selection & Cross-Validation Benchmark Report\n\n")
        f.write("| Model Architecture | CV MAPE | CV MAE (MW) | RMSE (MW) | R² Score | WAPE |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, row in df_metrics.iterrows():
            f.write(f"| **{idx}** | {row['mape']:.2%} | {row['mae']:.1f} MW | {row['rmse']:.1f} MW | {row['r2']:.4f} | {row['wape']:.2%} |\n")
        f.write("\n*Generated automatically by evaluation pipeline.*\n")

    print(f"\nStructured metrics saved to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
