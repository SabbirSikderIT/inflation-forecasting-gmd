"""Rebuild the preprocessing artifacts the dashboard needs and verify them.

The saved models (Models/*.pkl) were trained on *scaled* data, but the scalers
were never saved. This script re-runs the exact notebook pipeline on
data/gmd.csv, stores the scaler parameters as JSON, and then proves they are
correct by reproducing the published Fold-1 R2 of both models.

Usage (from the repo root, with the backend venv):
    app/backend/venv/Scripts/python app/scripts/export_artifacts.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parents[1]
GMD = ROOT / "data" / "gmd.csv"
OUT = APP / "backend" / "artifacts" / "preprocessing.json"
MODELS = APP / "backend" / "models"

FEATURES = ["infl", "rGDP_pc", "unemp", "govdebt_GDP", "CA_GDP", "USDfx", "REER", "pop"]
WINDOW, HORIZON = 10, 5
TRAIN_END, TEST_START, TEST_END = 2000, 2006, 2015

# Published Fold-1 R2 (results/final_config.json)
EXPECTED_R2 = {"RandomForest": 0.599, "XGBoost": 0.560, "Persistence": 0.547}


def main() -> int:
    df = pd.read_csv(GMD)
    df = df[df["year"] <= 2025].copy()
    counts = df[df["infl"].notna()].groupby("ISO3")["year"].count()
    long_iso = counts[counts >= 100].index
    df_long = df[df["ISO3"].isin(long_iso)].copy()
    df_model = df_long[df_long["year"] >= 1970].copy()
    df_model = df_model.sort_values(["ISO3", "year"]).reset_index(drop=True)

    for f in FEATURES:
        df_model[f] = df_model.groupby("ISO3")[f].ffill(limit=2)
        df_model[f] = df_model.groupby("ISO3")[f].bfill(limit=2)

    df_model["target_5y"] = df_model.groupby("ISO3")["infl"].transform(
        lambda s: s.rolling(HORIZON, min_periods=HORIZON).mean().shift(-HORIZON)
    )
    valid = df_model["target_5y"].dropna()
    t_lower, t_upper = float(valid.quantile(0.05)), float(valid.quantile(0.95))
    df_model["target_clipped"] = df_model["target_5y"].clip(t_lower, t_upper)
    shift = float(-df_model["target_clipped"].min() + 1)
    df_model["target_log"] = np.log(df_model["target_clipped"] + shift)

    mask = df_model["target_log"].notna()
    sx, sy = RobustScaler(), StandardScaler()
    sx.fit(df_model.loc[mask, FEATURES])
    sy.fit(df_model.loc[mask, ["target_log"]])
    print(f"scaler_y mean={sy.mean_[0]:.4f} scale={sy.scale_[0]:.4f} (notebook: 1.8383 / 1.1639)")
    print(f"clip=[{t_lower:.4f}, {t_upper:.4f}] shift={shift:.4f}")

    raw = df_model.copy()  # unscaled copy, used for the verification below
    df_model.loc[mask, FEATURES] = sx.transform(df_model.loc[mask, FEATURES])
    df_model.loc[mask, "target_scaled"] = sy.transform(df_model.loc[mask, ["target_log"]])

    # --- rebuild windows exactly like the notebook ---
    Xs, ys, meta = [], [], []
    for iso, g in df_model.groupby("ISO3"):
        g = g.sort_values("year").reset_index(drop=True)
        for i in range(WINDOW, len(g)):
            w, t = g.iloc[i - WINDOW:i], g.iloc[i]
            if pd.isna(t["target_scaled"]) or w[FEATURES].isna().any().any():
                continue
            Xs.append(w[FEATURES].values)
            ys.append(t["target_scaled"])
            meta.append((iso, int(t["year"]), t["target_clipped"]))
    X = np.array(Xs, dtype=np.float32)
    meta = pd.DataFrame(meta, columns=["ISO3", "pred_year", "target_real"])
    print(f"windows: {X.shape} (notebook n_samples: 1757)")

    test = ((meta["pred_year"] >= TEST_START) & (meta["pred_year"] <= TEST_END)).values
    X_test = X[test].reshape(test.sum(), -1)
    y_real = meta.loc[test, "target_real"].values

    def to_real(p):
        return np.exp(sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()) - shift

    ok = X.shape[0] == 1757
    train = (meta["pred_year"] <= TRAIN_END).values
    X_flat = X.reshape(len(X), -1)
    y_all = np.array(ys, dtype=np.float32)

    def check(name, preds, expected, tol):
        nonlocal ok
        r2 = r2_score(y_real, preds)
        good = abs(r2 - expected) < tol
        ok &= good
        print(f"{name:34s} R2={r2:.4f} expected {expected:.3f} -> {'OK' if good else 'MISMATCH'}")

    # Persistence baseline
    m = meta[test].reset_index(drop=True)
    pers = []
    for _, r in m.iterrows():
        pers.append(df_long[(df_long["ISO3"] == r["ISO3"]) &
                            (df_long["year"] >= r["pred_year"] - 5) &
                            (df_long["year"] < r["pred_year"])]["infl"].mean())
    check("Persistence", np.clip(np.array(pers), t_lower, t_upper), EXPECTED_R2["Persistence"], 0.0015)

    # Random Forest: the saved pickle reproduces the published Fold-1 score.
    with open(MODELS / "random_forest_fold1.pkl", "rb") as fh:
        rf = pickle.load(fh)
    check("Random Forest (saved pickle)", to_real(rf.predict(X_test)), EXPECTED_R2["RandomForest"], 0.0015)

    # XGBoost: the saved pickle scores 0.651 on this test fold instead of the published
    # 0.560, so it was NOT trained on the Fold-1 training set only (it saw later data).
    # Re-train it exactly as in the notebook (train <= 2000, same hyper-parameters)
    # and verify that this reproduces the published number.
    import xgboost as xgb
    try:
        with open(MODELS / "xgboost_fold1.pkl", "rb") as fh:
            legacy = pickle.load(fh)
        print(f"{'XGBoost (saved pickle, NOT used)':34s} R2={r2_score(y_real, to_real(legacy.predict(X_test))):.4f}")
    except Exception as exc:  # old pickle may not load on new xgboost
        print(f"XGBoost pickle not loadable here ({type(exc).__name__}); not needed.")
    xg = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)
    xg.fit(X_flat[train], y_all[train])
    check("XGBoost (re-trained, fold 1)", to_real(xg.predict(X_test)), EXPECTED_R2["XGBoost"], 0.003)
    xg.save_model(str(MODELS / "xgboost_fold1.json"))

    if not ok:
        print("VERIFICATION FAILED - artifacts not written")
        return 1

    # Country list + latest-year reference data for the demo / charts
    artifacts = {
        "features": FEATURES,
        "window": WINDOW,
        "horizon": HORIZON,
        "x_center": sx.center_.tolist(),
        "x_scale": sx.scale_.tolist(),
        "y_mean": float(sy.mean_[0]),
        "y_scale": float(sy.scale_[0]),
        "shift": shift,
        "clip_lower": t_lower,
        "clip_upper": t_upper,
        "verified_fold1_r2": {k: EXPECTED_R2[k] for k in EXPECTED_R2},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")

    # Ready-to-upload examples built from real GMD data (actual outcome printed for reference)
    samples_dir = APP / "backend" / "samples"
    samples_dir.mkdir(exist_ok=True)
    for iso in ["USA", "GBR", "IND"]:
        g = raw[(raw["ISO3"] == iso)].sort_values("year")
        g = g[g["year"] <= 2020].dropna(subset=FEATURES).tail(10)
        if len(g) == 10 and g["year"].diff().dropna().eq(1).all():
            g[["year"] + FEATURES].to_csv(samples_dir / f"sample_{iso}.csv", index=False)
            actual = raw[(raw["ISO3"] == iso) & (raw["year"] > g["year"].max()) &
                         (raw["year"] <= g["year"].max() + 5)]["infl"].mean()
            print(f"sample_{iso}.csv  years {g['year'].min()}-{g['year'].max()}  actual next-5y avg infl = {actual:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
