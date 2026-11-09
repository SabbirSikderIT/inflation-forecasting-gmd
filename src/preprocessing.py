"""Feature engineering, target construction, and scaling."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Tuple, List


FEATURES_DEFAULT = [
    'infl', 'rGDP_pc', 'unemp', 'govdebt_GDP',
    'CA_GDP', 'USDfx', 'REER', 'pop'
]


def fill_small_gaps(
    df: pd.DataFrame, 
    features: List[str], 
    limit: int = 2
) -> pd.DataFrame:
    """Forward/backward fill for gaps up to `limit` years per country."""
    df = df.sort_values(['ISO3', 'year']).copy()
    for feat in features:
        if feat in df.columns:
            df[feat] = df.groupby('ISO3')[feat].ffill(limit=limit)
            df[feat] = df.groupby('ISO3')[feat].bfill(limit=limit)
    return df


def build_target(
    df: pd.DataFrame,
    horizon: int = 5
) -> pd.DataFrame:
    """
    Build the target: horizon-year forward average inflation.
    
    target(t) = mean(infl(t+1), ..., infl(t+horizon))
    """
    df = df.sort_values(['ISO3', 'year']).reset_index(drop=True)
    df[f'target_{horizon}y'] = df.groupby('ISO3')['infl'].transform(
        lambda s: s.rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    )
    return df


def clip_and_log_target(
    df: pd.DataFrame,
    target_col: str,
    lower_q: float = 0.05,
    upper_q: float = 0.95
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Clip target and apply log transform.
    
    Returns:
        (df, lower_clip, upper_clip, shift)
    """
    valid = df[target_col].dropna()
    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    print(f"Clipping target to [{lower:.2f}, {upper:.2f}]")
    
    df['target_clipped'] = df[target_col].clip(lower, upper)
    shift = -df['target_clipped'].min() + 1
    df['target_log'] = np.log(df['target_clipped'] + shift)
    print(f"Shift value: {shift:.4f}")
    
    return df, lower, upper, shift


def scale_features_target(
    df: pd.DataFrame,
    features: List[str],
    train_mask: pd.Series
) -> Tuple[pd.DataFrame, RobustScaler, StandardScaler]:
    """
    Fit scalers on training data, apply to all rows.
    
    Returns:
        (df_with_scaled_cols, scaler_X, scaler_y)
    """
    scaler_X = RobustScaler()
    scaler_y = StandardScaler()
    
    train_valid = train_mask & df['target_log'].notna()
    scaler_X.fit(df.loc[train_valid, features])
    scaler_y.fit(df.loc[train_valid, ['target_log']])
    
    # Apply to all valid rows
    all_valid = df['target_log'].notna()
    df.loc[all_valid, features] = scaler_X.transform(df.loc[all_valid, features])
    df.loc[all_valid, 'target_scaled'] = scaler_y.transform(
        df.loc[all_valid, ['target_log']]
    )
    
    return df, scaler_X, scaler_y


def make_windows(
    df: pd.DataFrame,
    features: List[str],
    window_size: int
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create sliding windows.
    
    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,)
        meta: DataFrame with ISO3, pred_year, target_clipped
    """
    X_list, y_list, meta = [], [], []
    
    for iso, group in df.groupby('ISO3'):
        group = group.sort_values('year').reset_index(drop=True)
        n = len(group)
        
        for i in range(window_size, n):
            window = group.iloc[i-window_size:i]
            target_row = group.iloc[i]
            
            if pd.isna(target_row['target_scaled']):
                continue
            if window[features].isna().any().any():
                continue
            
            X_list.append(window[features].values)
            y_list.append(target_row['target_scaled'])
            meta.append({
                'ISO3': iso,
                'pred_year': int(target_row['year']),
                'target_real': target_row['target_clipped']
            })
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta)
    
    return X, y, meta