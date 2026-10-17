"""Data loading and initial filtering for GMD."""

import pandas as pd
from pathlib import Path
from typing import List

def load_gmd(path: str = "data/gmd.csv") -> pd.DataFrame:
    """Load raw GMD CSV."""
    df = pd.read_csv(path)
    print(f"Loaded GMD: {df.shape}, years {df['year'].min()}-{df['year'].max()}")
    return df


def filter_long_history_countries(
    df: pd.DataFrame, 
    min_years: int = 100
) -> pd.DataFrame:
    """
    Filter to countries with at least `min_years` of inflation data.
    
    Returns:
        DataFrame with only long-history countries.
    """
    # Count non-missing infl years per country
    counts = (
        df[df['infl'].notna()]
        .groupby('ISO3')['year']
        .count()
        .reset_index(name='n_years')
    )
    
    long_iso3 = counts[counts['n_years'] >= min_years]['ISO3'].tolist()
    print(f"Countries with ≥{min_years} years of infl data: {len(long_iso3)}")
    
    return df[df['ISO3'].isin(long_iso3)].copy()


def filter_year_range(
    df: pd.DataFrame, 
    start_year: int, 
    end_year: int = 2025
) -> pd.DataFrame:
    """Filter to a specific year range."""
    out = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    print(f"Filtered to {start_year}-{end_year}: {out.shape}")
    return out


def load_and_prepare(
    path: str = "data/gmd.csv",
    start_year: int = 1970,
    min_history_years: int = 100
) -> pd.DataFrame:
    """
    Complete data loading pipeline.
    
    Returns:
        Preprocessed DataFrame ready for feature engineering.
    """
    df = load_gmd(path)
    df = df[df['year'] <= 2025].copy()
    df = filter_long_history_countries(df, min_history_years)
    df = filter_year_range(df, start_year)
    return df.reset_index(drop=True)