"""Inflation Forecast API (FastAPI)."""
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
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    predictor.load_models()
    yield

app = FastAPI(title="Inflation Forecast API", version="1.0.0", lifespan=lifespan)

EXTRA_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:4173", "http://127.0.0.1:4173",
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
    try:
        bench = json.loads((BASE / "artifacts" / "benchmark.json").read_text(encoding="utf-8"))
    except Exception:
        bench = {"results": [], "dataset": "GMD 2025", "n_countries": 52, "year_range": "1970-2025", "n_samples": 0}
        
    r2 = {row["Model"]: row for row in bench["results"]}
    
    key_to_study = {
        "Persistence": "Persistence",
        "AdaBoost": "AdaBoost",
        "Custom VAE-Boost": "Custom VAE-Boost",
        "Ensemble": "Proposed AOWE Ensemble",
        "Custom MLTE": "Custom MLTE",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost"
    }
    
    available = predictor.available_models()
    models = []
    for key, label in predictor.MODEL_LABELS.items():
        row = r2.get(key_to_study.get(key, ""), {})
        models.append({
            "key": key,
            "label": label,
            "available": key in available,
            "fold1_r2": row.get("Fold 1 R²", 0.0),
            "fold2_r2": row.get("Fold 2 R²", 0.0),
            "mean_r2": row.get("Mean R²", 0.0),
        })
    return {
        "models": models,
        "benchmark": bench["results"],
        "features": predictor.FEATURES,
        "window": predictor.WINDOW,
        "horizon": predictor.HORIZON,
        "dataset": bench.get("dataset", "GMD 2025"),
        "n_countries": bench.get("n_countries", 52),
        "year_range": bench.get("year_range", "1970-2025"),
        "n_samples": bench.get("n_samples", 0),
    }

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("Ensemble"),
    db: Session = Depends(get_db),
):
    if model_name not in predictor.MODEL_LABELS:
        raise HTTPException(400, f"Invalid model '{model_name}'.")
    if model_name not in predictor.available_models():
        raise HTTPException(503, f"{predictor.MODEL_LABELS[model_name]} is not loaded.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (limit 2 MB).")
    if not raw.strip():
        raise HTTPException(422, "The uploaded file is empty.")
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8-sig")))
    except Exception as exc:
        raise HTTPException(422, f"Could not read CSV: {exc}")

    history = []
    last_year = 2020
    if 'year' in df.columns and 'infl' in df.columns:
        df_tail = df.tail(predictor.WINDOW)
        last_year = int(df_tail["year"].iloc[-1])
        history = [{"year": int(y), "infl": round(float(v), 2)} for y, v in zip(df_tail["year"], df_tail["infl"])]

    try:
        window_dict = predictor.prepare(df)
        df_processed = window_dict['df']
    except predictor.InputError as exc:
        raise HTTPException(422, str(exc))

    all_preds = predictor.predict_all(window_dict)
    value = all_preds[model_name]

    db.add(PredictionRecord(
        model_used=model_name,
        predicted_inflation=value,
        all_predictions=all_preds,
        filename=(file.filename or "")[:200],
        last_year=last_year,
        input_data=df_processed.to_dict(orient="records"),
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
