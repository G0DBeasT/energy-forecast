# Model Selection & Cross-Validation Benchmark Report

| Model Architecture | CV MAPE | CV MAE (MW) | RMSE (MW) | R² Score | WAPE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hourly Naive (Lag 168h)** | 6.24% | 412.7 MW | 523.2 MW | 0.7320 | 6.30% |
| **Hourly Ridge Regression** | 1.97% | 120.0 MW | 152.0 MW | 0.9654 | 1.94% |
| **Hourly LightGBM (Default)** | 2.08% | 130.5 MW | 167.7 MW | 0.9624 | 2.09% |
| **Hourly LightGBM (Tuned)** | 2.09% | 130.3 MW | 166.2 MW | 0.9624 | 2.09% |
| **Hourly XGBoost** | 2.13% | 132.8 MW | 169.5 MW | 0.9613 | 2.13% |
| **Daily Naive (Lag 7d)** | 3.53% | 220.5 MW | 282.9 MW | 0.5314 | 3.50% |
| **Daily Ridge Regression** | 0.67% | 38.6 MW | 48.6 MW | 0.9776 | 0.67% |
| **Daily LightGBM (Dedicated)** | 2.63% | 168.3 MW | 195.6 MW | 0.7816 | 2.68% |

*Generated automatically by evaluation pipeline.*
