"""Evaluation metrics and baselines."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, name: str = "Set") -> dict:
    """Compute RMSE, MAE, R² and print."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:+.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def inverse_pipeline(pred_scaled, scaler_y, shift):
    """Inverse: scaled -> log -> shifted -> real."""
    log_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    shifted_pred = np.exp(log_pred)
    return shifted_pred - shift


def persistence_baseline(
    df_long: pd.DataFrame,
    meta_test: pd.DataFrame,
    lookback: int = 5,
    clip_lower: float = None,
    clip_upper: float = None
) -> np.ndarray:
    """
    Persistence baseline: predict using last `lookback` years' average.
    """
    preds = []
    for _, row in meta_test.iterrows():
        iso = row['ISO3']
        py = row['pred_year']
        
        prev = df_long[
            (df_long['ISO3'] == iso) &
            (df_long['year'] >= py - lookback) &
            (df_long['year'] < py)
        ]['infl'].mean()
        preds.append(prev)
    
    preds = np.array(preds)
    if clip_lower is not None and clip_upper is not None:
        preds = np.clip(preds, clip_lower, clip_upper)
    return preds