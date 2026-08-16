"""
Multi-Horizon Model Training & TimeSeries Cross-Validation Pipeline.

Trains and validates multiple candidate architectures for:
1. Hourly Energy Forecasting (Next-Day 24h and Next-Week 168h)
   - Seasonal Naive Benchmark (lag 168h)
   - Ridge Linear Regression Baseline
   - LightGBM v1 (Default Gradient Booster)
   - LightGBM v2 (Tuned Gradient Booster)
   - XGBoost Regressor
2. Dedicated Daily Energy Forecasting (Next-Month 30-Day Planning)
   - Daily Seasonal Naive Benchmark (lag 7d)
   - Daily Ridge Linear Baseline
   - Dedicated Daily LightGBM Regressor (Tuned)

Persists champion models to the Model Registry with full CV diagnostics and MLflow tracking.
"""

from typing import Any, Dict, List, Tuple
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

from src.config import (
    DATA_PROC,
    EXPERIMENT_NAME,
    MLFLOW_DB_PATH,
    MLFLOW_URI,
    RANDOM_SEED,
    TARGET_COL,
)
from src.models.baselines import SeasonalNaiveModel
from src.models.evaluate import calculate_metrics
from src.models.registry import ModelRegistry

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Configure local MLflow tracking database."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except Exception as e:
        logger.warning(f"MLflow setup warning ({e}). Proceeding with local metrics logging.")
        return None


def train_hourly_models(X: pd.DataFrame, y: pd.Series, mlflow=None) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Train and evaluate candidate models for hourly forecasting."""
    print("\n" + "=" * 70)
    print("STAGE 1 · HOURLY MODEL EVALUATION (NEXT-DAY & NEXT-WEEK HORIZONS)")
    print(f"Observations: {len(X):,} | Features: {X.shape[1]} | 5-Fold TimeSeriesSplit (30-day blocks)")
    print("=" * 70)

    tss = TimeSeriesSplit(n_splits=5, test_size=24 * 30)

    candidate_models = {
        "Hourly Naive (Lag 168h)": {
            "model_creator": lambda: SeasonalNaiveModel(seasonal_lag=168, fallback_col="lag_168h"),
            "params": {"lag": 168},
        },
        "Hourly Ridge Regression": {
            "model_creator": lambda: Ridge(alpha=10.0, random_state=RANDOM_SEED),
            "params": {"alpha": 10.0},
        },
        "Hourly LightGBM (Default)": {
            "model_creator": lambda: lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
        },
        "Hourly LightGBM (Tuned)": {
            "model_creator": lambda: lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.04,
                num_leaves=45,
                colsample_bytree=0.8,
                subsample=0.8,
                min_child_samples=15,
                verbose=-1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            "params": {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 45, "colsample_bytree": 0.8},
        },
        "Hourly XGBoost": {
            "model_creator": lambda: xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.04, "max_depth": 5},
        },
    }

    results = {}
    best_model_name = ""
    best_mape = float("inf")

    for name, config in candidate_models.items():
        fold_metrics = []
        run_ctx = mlflow.start_run(run_name=f"Hourly_{name}") if mlflow else None

        try:
            if mlflow and run_ctx:
                mlflow.log_params(config["params"])

            for fold_idx, (tr_idx, val_idx) in enumerate(tss.split(X)):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                model = config["model_creator"]()
                if "LightGBM" in name:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
                elif "XGBoost" in name:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_tr, y_tr)

                pred = model.predict(X_val)
                fold_metrics.append(calculate_metrics(y_val.values, pred))

            avg_m = {
                "mape": float(np.mean([m["mape"] for m in fold_metrics])),
                "mae": float(np.mean([m["mae"] for m in fold_metrics])),
                "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
                "r2": float(np.mean([m["r2"] for m in fold_metrics])),
                "wape": float(np.mean([m["wape"] for m in fold_metrics])),
            }
            results[name] = avg_m

            if mlflow and run_ctx:
                mlflow.log_metrics({f"cv_{k}": v for k, v in avg_m.items()})

            print(f"[{name:<26}] CV MAPE: {avg_m['mape']:.2%} | MAE: {avg_m['mae']:.1f} MW | RMSE: {avg_m['rmse']:.1f} MW | R²: {avg_m['r2']:.4f}")

            if avg_m["mape"] < best_mape:
                best_mape = avg_m["mape"]
                best_model_name = name

        finally:
            if mlflow and run_ctx:
                mlflow.end_run()

    # Refit champion hourly model on full dataset
    print(f"\nRefitting Champion Hourly Model: '{best_model_name}' on all {len(X):,} observations...")
    champion_hourly = candidate_models[best_model_name]["model_creator"]()
    champion_hourly.fit(X, y)

    res_df = pd.DataFrame(results).T
    return champion_hourly, results, res_df


def train_daily_models(X: pd.DataFrame, y: pd.Series, mlflow=None) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Train and evaluate dedicated daily models for long-range Next-Month forecasting."""
    print("\n" + "=" * 70)
    print("STAGE 2 · DEDICATED DAILY MODEL EVALUATION (NEXT-MONTH HORIZON)")
    print(f"Observations: {len(X):,} Days | Features: {X.shape[1]} | 5-Fold TimeSeriesSplit (60-day blocks)")
    print("=" * 70)

    tss = TimeSeriesSplit(n_splits=5, test_size=60)

    candidate_models = {
        "Daily Naive (Lag 7d)": {
            "model_creator": lambda: SeasonalNaiveModel(seasonal_lag=7, fallback_col="lag_7d"),
            "params": {"lag": 7},
        },
        "Daily Ridge Regression": {
            "model_creator": lambda: Ridge(alpha=5.0, random_state=RANDOM_SEED),
            "params": {"alpha": 5.0},
        },
        "Daily LightGBM (Dedicated)": {
            "model_creator": lambda: lgb.LGBMRegressor(
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=25,
                min_child_samples=10,
                subsample=0.85,
                colsample_bytree=0.85,
                verbose=-1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            "params": {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 25},
        },
    }

    results = {}
    best_model_name = ""
    best_mape = float("inf")

    for name, config in candidate_models.items():
        fold_metrics = []
        run_ctx = mlflow.start_run(run_name=f"Daily_{name}") if mlflow else None

        try:
            if mlflow and run_ctx:
                mlflow.log_params(config["params"])

            for fold_idx, (tr_idx, val_idx) in enumerate(tss.split(X)):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                model = config["model_creator"]()
                if "LightGBM" in name:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])
                else:
                    model.fit(X_tr, y_tr)

                pred = model.predict(X_val)
                fold_metrics.append(calculate_metrics(y_val.values, pred))

            avg_m = {
                "mape": float(np.mean([m["mape"] for m in fold_metrics])),
                "mae": float(np.mean([m["mae"] for m in fold_metrics])),
                "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
                "r2": float(np.mean([m["r2"] for m in fold_metrics])),
                "wape": float(np.mean([m["wape"] for m in fold_metrics])),
            }
            results[name] = avg_m

            if mlflow and run_ctx:
                mlflow.log_metrics({f"cv_{k}": v for k, v in avg_m.items()})

            print(f"[{name:<26}] CV MAPE: {avg_m['mape']:.2%} | MAE: {avg_m['mae']:.1f} MW | RMSE: {avg_m['rmse']:.1f} MW | R²: {avg_m['r2']:.4f}")

            if avg_m["mape"] < best_mape:
                best_mape = avg_m["mape"]
                best_model_name = name

        finally:
            if mlflow and run_ctx:
                mlflow.end_run()

    print(f"\nRefitting Champion Daily Model: '{best_model_name}' on all {len(X):,} days...")
    champion_daily = candidate_models[best_model_name]["model_creator"]()
    champion_daily.fit(X, y)

    res_df = pd.DataFrame(results).T
    return champion_daily, results, res_df


def run_training_pipeline():
    """Execute complete multi-horizon training pipeline."""
    mlflow = setup_mlflow()

    # 1. Hourly Training
    hourly_path = DATA_PROC / "features.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Feature matrix {hourly_path} missing. Run feature pipeline first.")

    df_hourly = pd.read_parquet(hourly_path)
    feature_cols_hourly = [c for c in df_hourly.columns if c != TARGET_COL]
    X_h, y_h = df_hourly[feature_cols_hourly], df_hourly[TARGET_COL]

    champion_hourly, hourly_results, df_res_h = train_hourly_models(X_h, y_h, mlflow=mlflow)

    ModelRegistry.save_model(
        model=champion_hourly,
        feature_cols=feature_cols_hourly,
        metrics=hourly_results,
        model_name="Hourly LightGBM (Tuned)",
        provenance="synthetic_simulated",
        destination_dir=DATA_PROC,
    )

    # 2. Daily Training for Next-Month
    daily_path = DATA_PROC / "features_daily.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily feature matrix {daily_path} missing. Run feature pipeline first.")

    df_daily = pd.read_parquet(daily_path)
    feature_cols_daily = [c for c in df_daily.columns if c != TARGET_COL]
    X_d, y_d = df_daily[feature_cols_daily], df_daily[TARGET_COL]

    champion_daily, daily_results, df_res_d = train_daily_models(X_d, y_d, mlflow=mlflow)

    import pickle
    daily_model_path = DATA_PROC / "best_model_daily.pkl"
    with open(daily_model_path, "wb") as f:
        pickle.dump(champion_daily, f)

    import json
    with open(DATA_PROC / "model_metadata_daily.json", "w") as f:
        json.dump({
            "model_name": "Daily LightGBM (Dedicated)",
            "feature_columns": feature_cols_daily,
            "metrics": daily_results,
        }, f, indent=2)

    # Combined summary table
    combined_res = pd.concat([df_res_h, df_res_d])
    combined_res.to_parquet(DATA_PROC / "metrics_summary.parquet")

    print("\n" + "=" * 70)
    print("ALL MODELS CROSS-VALIDATION SUMMARY (PHASE 8 & 11)")
    print("=" * 70)
    disp = combined_res.copy()
    disp["mape"] = disp["mape"].apply(lambda x: f"{x:.2%}")
    disp["mae"] = disp["mae"].apply(lambda x: f"{x:.1f} MW")
    disp["rmse"] = disp["rmse"].apply(lambda x: f"{x:.1f} MW")
    disp["r2"] = disp["r2"].apply(lambda x: f"{x:.4f}")
    disp["wape"] = disp["wape"].apply(lambda x: f"{x:.2%}")
    print(disp.to_string())
    print("=" * 70)

    return champion_hourly, champion_daily


if __name__ == "__main__":
    run_training_pipeline()
