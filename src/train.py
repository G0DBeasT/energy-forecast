import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config import DATA_PROC, EXPERIMENT_NAME, TARGET_COL
from src.evaluate import regression_metrics

def setup_mlflow():
    """Configure MLflow tracking for local execution."""
    db_path = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(EXPERIMENT_NAME)

def run_training_pipeline():
    setup_mlflow()
    
    parquet_path = DATA_PROC / 'features.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(f"Feature matrix {parquet_path} does not exist. Run Phase 4 first.")
        
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X, y = df[feature_cols], df[TARGET_COL]
    
    print(f"Features: {len(feature_cols)} columns, Total observations: {len(X)}")
    
    # 5-fold TimeSeriesSplit with 30-day (720h) validation windows
    tss = TimeSeriesSplit(n_splits=5, test_size=24 * 30)
    
    results = {}
    best_model_obj = None
    best_mape = float('inf')
    best_model_name = ""

    # ==========================================
    # Run 1: Naive Baseline (lag_168h)
    # ==========================================
    with mlflow.start_run(run_name="01_naive_baseline"):
        mlflow.log_param("model_type", "Naive_Lag168h")
        fold_metrics = []
        for tr_idx, val_idx in tss.split(X):
            y_val = y.iloc[val_idx]
            pred = X.iloc[val_idx]['lag_168h']
            fold_metrics.append(regression_metrics(y_val, pred))
            
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg_metrics.items()})
        results["Naive Baseline"] = avg_metrics
        print(f"Run 1 - Naive Baseline : CV MAPE = {avg_metrics['mape']:.2%}, MAE = {avg_metrics['mae']:.1f} MW")

    # ==========================================
    # Run 2: Linear Regression
    # ==========================================
    with mlflow.start_run(run_name="02_linear_regression"):
        mlflow.log_param("model_type", "LinearRegression")
        fold_metrics = []
        for tr_idx, val_idx in tss.split(X):
            lr = LinearRegression().fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = lr.predict(X.iloc[val_idx])
            fold_metrics.append(regression_metrics(y.iloc[val_idx], pred))
            
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg_metrics.items()})
        results["Linear Regression"] = avg_metrics
        print(f"Run 2 - Linear Regr    : CV MAPE = {avg_metrics['mape']:.2%}, MAE = {avg_metrics['mae']:.1f} MW")

    # ==========================================
    # Run 3: LightGBM v1 (Default)
    # ==========================================
    lgb_params_v1 = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42
    }
    with mlflow.start_run(run_name="03_lightgbm_v1_default"):
        mlflow.log_params(lgb_params_v1)
        fold_metrics = []
        lgb_model = None
        for tr_idx, val_idx in tss.split(X):
            lgb_model = lgb.LGBMRegressor(**lgb_params_v1)
            lgb_model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            pred = lgb_model.predict(X.iloc[val_idx])
            fold_metrics.append(regression_metrics(y.iloc[val_idx], pred))
            
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg_metrics.items()})
        results["LightGBM v1 (Default)"] = avg_metrics
        print(f"Run 3 - LightGBM v1    : CV MAPE = {avg_metrics['mape']:.2%}, MAE = {avg_metrics['mae']:.1f} MW")
        
        if avg_metrics['mape'] < best_mape:
            best_mape = avg_metrics['mape']
            best_model_obj = lgb_model
            best_model_name = "LightGBM v1"

    # ==========================================
    # Run 4: LightGBM v2 (Tuned Hyperparameters)
    # ==========================================
    lgb_params_v2 = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 45,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "min_child_samples": 15,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42
    }
    with mlflow.start_run(run_name="04_lightgbm_v2_tuned"):
        mlflow.log_params(lgb_params_v2)
        fold_metrics = []
        lgb_v2_model = None
        for tr_idx, val_idx in tss.split(X):
            lgb_v2_model = lgb.LGBMRegressor(**lgb_params_v2)
            lgb_v2_model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            pred = lgb_v2_model.predict(X.iloc[val_idx])
            fold_metrics.append(regression_metrics(y.iloc[val_idx], pred))
            
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg_metrics.items()})
        results["LightGBM v2 (Tuned)"] = avg_metrics
        print(f"Run 4 - LightGBM v2    : CV MAPE = {avg_metrics['mape']:.2%}, MAE = {avg_metrics['mae']:.1f} MW")
        
        if avg_metrics['mape'] < best_mape:
            best_mape = avg_metrics['mape']
            best_model_obj = lgb_v2_model
            best_model_name = "LightGBM v2 (Tuned)"

    # ==========================================
    # Run 5: XGBoost Regressor
    # ==========================================
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42
    }
    with mlflow.start_run(run_name="05_xgboost_regressor"):
        mlflow.log_params(xgb_params)
        fold_metrics = []
        xgb_model = None
        for tr_idx, val_idx in tss.split(X):
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False
            )
            pred = xgb_model.predict(X.iloc[val_idx])
            fold_metrics.append(regression_metrics(y.iloc[val_idx], pred))
            
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg_metrics.items()})
        results["XGBoost"] = avg_metrics
        print(f"Run 5 - XGBoost        : CV MAPE = {avg_metrics['mape']:.2%}, MAE = {avg_metrics['mae']:.1f} MW")
        
        if avg_metrics['mape'] < best_mape:
            best_mape = avg_metrics['mape']
            best_model_obj = xgb_model
            best_model_name = "XGBoost"

    # Save best model pickle
    if best_model_obj is not None:
        best_model_path = DATA_PROC / 'best_model.pkl'
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model_obj, f)
        print(f"\nSaved best model ({best_model_name}, CV MAPE={best_mape:.2%}) to {best_model_path}")
        
        summary_df = pd.DataFrame(results).T
        summary_df.to_parquet(DATA_PROC / 'metrics_summary.parquet')

    print("\n" + "="*70)
    print("PHASE 8 · MODEL RESULTS COMPARISON TABLE")
    print("="*70)
    res_df = pd.DataFrame(results).T
    res_df['mape'] = res_df['mape'].apply(lambda x: f"{x:.2%}")
    res_df['mae']  = res_df['mae'].apply(lambda x: f"{x:.1f} MW")
    res_df['rmse'] = res_df['rmse'].apply(lambda x: f"{x:.1f} MW")
    res_df['r2']   = res_df['r2'].apply(lambda x: f"{x:.4f}")
    print(res_df.to_string())
    print("="*70)

if __name__ == "__main__":
    run_training_pipeline()
