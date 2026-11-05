"""Input validation, preprocessing and prediction."""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"

# --- Load preprocessing parameters from JSON (for Tree Models) ---
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
    "GradientBoosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "CNN-LSTM": "CNN-LSTM (Aug)",
    "Persistence": "Persistence Baseline",
}

models: Dict[str, object] = {}
load_errors: Dict[str, str] = {}
TEMPORAL_PREPROCESSORS = None

# --- PyTorch Model Architecture ---
class Small_CNN_LSTM(nn.Module):
    def __init__(self, n_features, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 16, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.4) 
        self.lstm = nn.LSTM(16, 16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)


class InputError(ValueError):
    """The uploaded data cannot be used; message is safe to show to the user."""


def _lazy_load_lstm():
    """Lazily load the LSTM models and preprocessors only when needed."""
    global TEMPORAL_PREPROCESSORS
    
    if "CNN-LSTM" in models and TEMPORAL_PREPROCESSORS is not None:
        return True
    if "CNN-LSTM" in load_errors:
        return False
        
    try:
        device = torch.device('cpu')
        model_state = torch.load(MODELS_DIR / 'temporal_lstm_model.pth', map_location=device)
        m = Small_CNN_LSTM(len(FEATURES), WINDOW).to(device)
        m.load_state_dict(model_state)
        m.eval()
        models["CNN-LSTM"] = [m] 
        
        with open(MODELS_DIR / "temporal_preprocessors.pkl", "rb") as f:
            TEMPORAL_PREPROCESSORS = pickle.load(f)
            
        print("✅ CNN-LSTM (Temporal) loaded successfully")
        return True
    except Exception as exc:
        load_errors["CNN-LSTM"] = f"{type(exc).__name__}: {exc}"
        print(f"[models] CNN-LSTM failed to load: {load_errors['CNN-LSTM']}")
        return False


def load_models() -> None:
    """Load the trained tree-based models once at start-up."""
    sources = {
        "RandomForest": "random_forest_fold1.pkl",
        "GradientBoosting": "gradient_boosting_fold1.pkl",
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
                    models[name] = pickle.load(fh)
        except Exception as exc:
            load_errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"[models] {name} failed to load: {load_errors[name]}")


def available_models() -> List[str]:
    available = [*models.keys(), "Persistence"]
    if "CNN-LSTM" not in available and "CNN-LSTM" not in load_errors:
        available.append("CNN-LSTM")
    return available


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

    if "RandomForest" in models or "XGBoost" in models or "GradientBoosting" in models:
        scaled = (window[FEATURES].to_numpy(dtype=np.float64) - X_CENTER) / X_SCALE
        flat = scaled.astype(np.float32).reshape(1, -1)
        for name in ["RandomForest", "XGBoost", "GradientBoosting"]:
            if name in models:
                y_scaled = float(np.asarray(models[name].predict(flat)).ravel()[0])
                out[name] = float(np.exp(y_scaled * Y_SCALE + Y_MEAN) - SHIFT)

    if "CNN-LSTM" in available_models() and _lazy_load_lstm():
        scaler_X_new = TEMPORAL_PREPROCESSORS['scaler_X']
        scaler_y_new = TEMPORAL_PREPROCESSORS['scaler_y']
        shift_new = TEMPORAL_PREPROCESSORS['shift']
        
        X_raw = window[FEATURES].to_numpy(dtype=np.float64)
        X_scaled_new = scaler_X_new.transform(X_raw)
        
        X_scaled_new = X_scaled_new.reshape(1, WINDOW, len(FEATURES))
        X_tensor = torch.tensor(X_scaled_new, dtype=torch.float32)
        
        preds = []
        with torch.no_grad():
            for m in models["CNN-LSTM"]:
                preds.append(m(X_tensor).item())
        
        avg_pred_scaled = np.mean(preds)
        
        log_pred = scaler_y_new.inverse_transform([[avg_pred_scaled]])[0][0]
        real_pred = np.exp(log_pred) - shift_new
        
        real_pred = np.clip(real_pred, 0, 100)
        out["CNN-LSTM"] = float(real_pred)

    return out
