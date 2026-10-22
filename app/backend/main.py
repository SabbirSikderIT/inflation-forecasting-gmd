"""Inflation Forecast API (FastAPI).

Run:  uvicorn main:app --reload      (http://localhost:8000, docs at /docs)
"""

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

import predictor
from database import PredictionRecord, get_db, init_db

BASE = Path(__file__).resolve().parent
SAMPLES_DIR = BASE / "samples"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB is plenty for a 10-year CSV


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    predictor.load_models()
    yield


app = FastAPI(title="Inflation Forecast API", version="1.0.0", lifespan=lifespan)

# Extra allowed browser origins for a hosted frontend, comma separated, no trailing slash:
#   CORS_ORIGINS=https://my-dashboard.vercel.app
EXTRA_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite dev server
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:4173", "http://127.0.0.1:4173",  # Vite preview
        *EXTRA_ORIGINS,
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _record_to_dict(r: PredictionRecord) -> dict:
    return {
        "id": r.id,
        "model": r.model_used,
        "prediction": round(r.predicted_inflation, 2),
        "all_predictions": {k: round(v, 2) for k, v in r.all_predictions.items()},
        "filename": r.filename,
        "last_year": r.last_year,
        "input_data": r.input_data,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# 🚨 UPDATED: GET এবং HEAD দুটো মেথডই সাপোর্ট করবে UptimeRobot-এর জন্য
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Inflation Forecast API is running"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_available": predictor.available_models(),
        "models_failed": predictor.load_errors,
    }


@app.get("/api/models")
def list_models():
    """Model metadata plus the cross-validated R2 scores from the study."""
    bench = json.loads((BASE / "artifacts" / "benchmark.json").read_text(encoding="utf-8"))
    r2 = {row["Model"]: row for row in bench["results"]}
    key_to_study = {"RandomForest": "Random Forest", "XGBoost": "XGBoost", "Persistence": "Persistence"}
    available = predictor.available_models()
    models = []
    for key, label in predictor.MODEL_LABELS.items():
        row = r2.get(key_to_study[key], {})
        models.append({
            "key": key,
            "label": label,
            "available": key in available,
            "fold1_r2": row.get("Fold 1 R²"),
            "fold2_r2": row.get("Fold 2 R²"),
            "mean_r2": row.get("Mean R²"),
        })
    return {
        "models": models,
        "benchmark": bench["results"],
        "features": predictor.FEATURES,
        "window": predictor.WINDOW,
        "horizon": predictor.HORIZON,
        "dataset": bench["dataset"],
        "n_countries": bench["n_countries"],
        "year_range": bench["year_range"],
        "n_samples": bench["n_samples"],
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("RandomForest"),
    db: Session = Depends(get_db),
):
    if model_name not in predictor.MODEL_LABELS:
        raise HTTPException(400, f"Invalid model '{model_name}'. Choose one of: {', '.join(predictor.MODEL_LABELS)}.")
    if model_name not in predictor.available_models():
        raise HTTPException(503, f"{predictor.MODEL_LABELS[model_name]} is not loaded on the server.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (limit 2 MB).")
    if not raw.strip():
        raise HTTPException(422, "The uploaded file is empty.")
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8-sig")))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(422, f"Could not read the file as a CSV: {exc}")

    try:
        window = predictor.prepare(df)
    except predictor.InputError as exc:
        raise HTTPException(422, str(exc))

    all_preds = predictor.predict_all(window)
    value = all_preds[model_name]
    last_year = int(window["year"].iloc[-1])
    history = [{"year": int(y), "infl": round(float(v), 2)} for y, v in zip(window["year"], window["infl"])]

    db.add(PredictionRecord(
        model_used=model_name,
        predicted_inflation=value,
        all_predictions=all_preds,
        filename=(file.filename or "")[:200],
        last_year=last_year,
        input_data=window.to_dict(orient="records"),
    ))
    db.commit()

    return {
        "prediction": round(value, 2),
        "model": model_name,
        "model_label": predictor.MODEL_LABELS[model_name],
        "all_predictions": {k: round(v, 2) for k, v in all_preds.items()},
        "history": history,
        "forecast_start": last_year + 1,
        "forecast_end": last_year + predictor.HORIZON,
    }


@app.get("/api/predictions")
def predictions(limit: int = 20, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    rows: List[PredictionRecord] = db.scalars(
        select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)
    ).all()
    return [_record_to_dict(r) for r in rows]


@app.delete("/api/predictions/{record_id}")
def delete_prediction(record_id: int, db: Session = Depends(get_db)):
    rec = db.get(PredictionRecord, record_id)
    if rec is None:
        raise HTTPException(404, "Prediction not found.")
    db.delete(rec)
    db.commit()
    return {"deleted": record_id}


@app.delete("/api/predictions")
def clear_predictions(db: Session = Depends(get_db)):
    n = db.query(PredictionRecord).delete()
    db.commit()
    return {"deleted": n}


@app.get("/api/samples")
def samples():
    return sorted(p.name for p in SAMPLES_DIR.glob("sample_*.csv"))


@app.get("/api/samples/{name}")
def sample(name: str):
    path = SAMPLES_DIR / name
    if path.parent != SAMPLES_DIR or not name.startswith("sample_") or path.suffix != ".csv" or not path.is_file():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(path, media_type="text/csv", filename=name)
