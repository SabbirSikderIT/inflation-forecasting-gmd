"""Input validation, preprocessing and prediction.

Mirrors the training pipeline (notebooks/Untitled0.ipynb):
  1. forward/backward fill gaps of up to 2 years
  2. RobustScaler on the 8 features (parameters stored in artifacts/preprocessing.json)
  3. flatten the 10-year x 8-feature window (time-major) into 80 inputs
  4. model output is a standardised log target:
        real = exp(y * y_scale + y_mean) - shift
  5. Persistence = mean inflation of the last 5 years, clipped to the training range
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
ART = json.loads((BASE / "artifacts" / "preprocessing.json").read_text(encoding="utf-8"))

FEATURES: List[str] = ART["features"]
WINDOW: int = ART["window"]
HORIZON: int = ART["horizon"]
X_CENTER = np.array(ART["x_center"], dtype=np.float64)
X_SCALE = np.array(ART["x_scale"], dtype=np.float64)
Y_MEAN, Y_SCALE, SHIFT = ART["y_mean"], ART["y_scale"], ART["shift"]
CLIP_LOWER, CLIP_UPPER = ART["clip_lower"], ART["clip_upper"]

MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "Persistence": "Persistence Baseline",
}

models: Dict[str, object] = {}
load_errors: Dict[str, str] = {}


class InputError(ValueError):
    """The uploaded data cannot be used; message is safe to show to the user."""


def load_models() -> None:
    """Load the trained models once at start-up."""
    sources = {
        "RandomForest": "random_forest_fold1.pkl",
        "XGBoost": "xgboost_fold1.json",
    }
    for name, fname in sources.items():
        path = MODELS_DIR / fname
        try:
            if path.suffix == ".json":
                import xgboost as xgb

                m = xgb.XGBRegressor()
                m.load_model(str(path))
                models[name] = m
            else:
                with open(path, "rb") as fh:
                    models[name] = pickle.load(fh)  # trusted, repo-owned files only
        except Exception as exc:  # keep the API up; report the model as unavailable
            load_errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"[models] {name} failed to load: {load_errors[name]}")


def available_models() -> List[str]:
    return [*models.keys(), "Persistence"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the upload and return the last WINDOW consecutive years, filled."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in ["year", *FEATURES] if c not in df.columns]
    if missing:
        raise InputError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: year, {', '.join(FEATURES)}."
        )
    df = df[["year", *FEATURES]].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df["year"].isna().any():
        raise InputError("The 'year' column contains empty or non-numeric values.")
    if (df["year"] % 1 != 0).any():
        raise InputError("The 'year' column must contain whole years (e.g. 2015).")
    df["year"] = df["year"].astype(int)
    if df["year"].duplicated().any():
        dups = sorted(df.loc[df["year"].duplicated(), "year"].unique().tolist())
        raise InputError(f"Duplicate year(s) in the file: {dups}.")
    df = df.sort_values("year").reset_index(drop=True)

    if len(df) < WINDOW:
        raise InputError(f"At least {WINDOW} years of data are required; the file has {len(df)}.")

    # Same small-gap filling as training (limit = 2 years)
    for f in FEATURES:
        df[f] = df[f].ffill(limit=2).bfill(limit=2)

    window = df.tail(WINDOW).reset_index(drop=True)
    years = window["year"].tolist()
    if years != list(range(years[0], years[0] + WINDOW)):
        raise InputError(
            f"The last {WINDOW} rows must be {WINDOW} consecutive years; found {years}."
        )
    bad = [f for f in FEATURES if window[f].isna().any()]
    if bad:
        raise InputError(
            f"Missing values in the last {WINDOW} years that could not be filled "
            f"(gaps longer than 2 years): {', '.join(bad)}."
        )
    return window


def predict_all(window: pd.DataFrame) -> Dict[str, float]:
    """Return the 5-year-ahead average inflation (%) for every available model."""
    out: Dict[str, float] = {}

    persistence = float(window["infl"].tail(5).mean())
    out["Persistence"] = float(np.clip(persistence, CLIP_LOWER, CLIP_UPPER))

    if models:
        scaled = (window[FEATURES].to_numpy(dtype=np.float64) - X_CENTER) / X_SCALE
        flat = scaled.astype(np.float32).reshape(1, -1)  # time-major, like training
        for name, model in models.items():
            y_scaled = float(np.asarray(model.predict(flat)).ravel()[0])
            out[name] = float(np.exp(y_scaled * Y_SCALE + Y_MEAN) - SHIFT)
    return out
