import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from src.config import DATA_PROC, REPORTS, TARGET_COL

def run_error_analysis():
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = (14, 5)

    feat_path = DATA_PROC / 'features.parquet'
    model_path = DATA_PROC / 'best_model.pkl'
    
    if not feat_path.exists() or not model_path.exists():
        raise FileNotFoundError("Features or best model pickle missing.")
        
    df = pd.read_parquet(feat_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    
    # Holdout last 60 days for deep error analysis
    test_df = df.tail(60 * 24).copy()
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    
    test_df['pred'] = model.predict(X_test)
    test_df['residual'] = test_df['pred'] - test_df[TARGET_COL]
    test_df['abs_pct_error'] = np.abs(test_df['residual']) / test_df[TARGET_COL] * 100.0
    
    overall_mape = mean_absolute_percentage_error(y_test, test_df['pred'])
    overall_mae  = mean_absolute_error(y_test, test_df['pred'])
    
    # 1. Actual vs Predicted for 1 representative week in test set
    plt.figure()
    sample = test_df.tail(7 * 24)
    plt.plot(sample.index, sample[TARGET_COL], label='Actual Demand (MW)', color='#378ADD', linewidth=2)
    plt.plot(sample.index, sample['pred'], label='LightGBM Forecast (MW)', color='#EF9F27', linestyle='--', linewidth=2)
    plt.title('Actual vs Predicted Demand (Holdout Test Week)')
    plt.xlabel('Datetime')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '09_actual_vs_predicted.png', dpi=120)
    plt.close()

    # 2. Residual Distribution Histogram
    plt.figure()
    plt.hist(test_df['residual'], bins=40, color='#4CAF50', edgecolor='black', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title(f'Residual Error Distribution (Mean={test_df["residual"].mean():.1f} MW, Std={test_df["residual"].std():.1f} MW)')
    plt.xlabel('Residual Error (Predicted - Actual MW)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '10_residual_distribution.png', dpi=120)
    plt.close()

    # 3. MAPE by Hour of Day
    plt.figure()
    hourly_mape = test_df.groupby(test_df.index.hour)['abs_pct_error'].mean()
    hourly_mape.plot(kind='bar', color='#00BCD4')
    plt.axhline(overall_mape * 100, color='yellow', linestyle='--', label=f'Overall MAPE ({overall_mape:.2%})')
    plt.title('Mean Absolute Percentage Error (MAPE %) by Hour of Day')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('MAPE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '11_mape_by_hour.png', dpi=120)
    plt.close()

    # 4. Holiday vs Non-Holiday Error Comparison
    plt.figure()
    holiday_comparison = test_df.groupby('is_holiday')['abs_pct_error'].mean()
    labels = ['Regular Day', 'Public Holiday']
    vals = [holiday_comparison.get(0, 0), holiday_comparison.get(1, 0)]
    plt.bar(labels, vals, color=['#8BC34A', '#E91E63'], width=0.4)
    plt.title('Forecast Error Comparison: Regular Days vs Holidays')
    plt.ylabel('MAPE (%)')
    for i, v in enumerate(vals):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / '12_holiday_error_comparison.png', dpi=120)
    plt.close()

    print("\n" + "="*60)
    print("PHASE 7 · ERROR ANALYSIS FINDINGS")
    print("="*60)
    print(f"Overall Holdout Test MAPE : {overall_mape:.2%}")
    print(f"Overall Holdout Test MAE  : {overall_mae:.2f} MW")
    print(f"Worst Hour of Day         : Hour {hourly_mape.idxmax()} (MAPE = {hourly_mape.max():.2f}%)")
    print(f"Best Hour of Day          : Hour {hourly_mape.idxmin()} (MAPE = {hourly_mape.min():.2f}%)")
    print(f"Regular Day MAPE          : {vals[0]:.2f}%")
    print(f"Public Holiday MAPE       : {vals[1]:.2f}%")
    print("="*60)
    print(f"Saved error analysis figures (09-12) to {REPORTS}/")

if __name__ == '__main__':
    run_error_analysis()
