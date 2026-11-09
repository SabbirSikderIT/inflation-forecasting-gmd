# Medium-Horizon Inflation Forecasting with GMD

## Overview
Leakage-free benchmark for 5-year-ahead inflation forecasting using the 
Global Macro Database (GMD), comparing 6 models with 2-fold temporal cross-validation.

## Key Finding
**Persistence dominates.** The simple "last 5-year average" benchmark achieves 
mean R² = 0.66, while tree-based ML models achieve R² = 0.49-0.51 with high 
variance. Deep learning (CNN-LSTM) is sample-size sensitive: R² = 0.12 
(775 samples) vs 0.47 (1,015 samples).

## Data
- Source: Global Macro Database (2025 release)
- Countries: 52 (100+ years of inflation history)
- Period: 1970-2025 (55 years)
- Features: 8 macroeconomic variables
- Target: 5-year forward average inflation

## Methodology
- Window: 10 years
- Horizon: 5 years
- Split: Train ≤ 2000, Test 2006-2015 / 2016-2020
- CV: 2 temporal folds

## Results

| Model | Fold 1 | Fold 2 | Mean R² |
|-------|--------|--------|---------|
| Persistence | 0.547 | 0.764 | **0.6555** |
| XGBoost | 0.560 | 0.469 | 0.5145 |
| Random Forest | 0.599 | 0.406 | 0.5025 |
| Gradient Boosting | 0.531 | 0.452 | 0.4915 |
| CNN-LSTM | 0.116 | 0.470 | 0.2931 |
| Naive | -0.320 | -0.007 | -0.1635 |

## Key Findings
1. Persistence is the most robust benchmark
2. Tree-based ML methods are period-dependent
3. Deep learning requires >1,000 training samples
4. Linear models fail catastrophically

## Reproducibility
- Python 3.10, PyTorch 2.x, scikit-learn, XGBoost
- Random seed: 42
- Hardware: Google Colab T4 GPU

## Files
- final_results_paper.csv — Final CV results
- final_abstract.txt — Abstract
- key_findings_final.txt — Summary
- final_project_dashboard.png — Visualization
- df_long_backup.csv — Preprocessed data
