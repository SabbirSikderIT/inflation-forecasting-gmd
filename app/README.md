# Inflation Forecaster – dashboard

A web dashboard for the GMD 5-year-ahead inflation forecasting benchmark.
Upload 10 years of macro data (CSV) and get the forecast of the **average inflation of the next 5 years**
from Random Forest, XGBoost and the Persistence baseline, next to the study's cross-validated results.

```
app/
├── backend/    FastAPI + SQLite (predictions are stored and listed in the UI)
│   ├── models/        random_forest_fold1.pkl, xgboost_fold1.json
│   ├── artifacts/     preprocessing.json (scalers), benchmark.json (study R²)
│   ├── samples/       real example CSVs (USA, GBR, IND; 2011-2020)
│   └── tests/test_api.py
├── frontend/   React (Vite) + Recharts
└── scripts/export_artifacts.py   rebuilds + verifies artifacts from data/gmd.csv
```

## Run it

Needs **Python 3.10–3.13** (not 3.14: scikit-learn 1.6.1, which the saved Random Forest requires, has no 3.14 wheels) and Node 18+.

**Terminal 1 – backend**
```bash
cd app/backend
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload        # http://localhost:8000  (API docs: /docs)
```

**Terminal 2 – frontend**
```bash
cd app/frontend
npm install
npm run dev                      # http://localhost:5173
```

Open http://localhost:5173 and click one of the example countries, or upload your own CSV
(`year,infl,rGDP_pc,unemp,govdebt_GDP,CA_GDP,USDfx,REER,pop`, ≥ 10 consecutive years, gaps of up to 2 years are filled
like in training). `http://localhost:5173/#demo=USA` opens a ready-made forecast.

Tests: `cd backend && venv\Scripts\python -m tests.test_api`

## How predictions are made

Identical to the notebook: fill gaps ≤ 2 years → RobustScaler on the 8 features → flatten the 10×8 window → model →
`exp(y·σ + μ) − shift`. Persistence = mean of the last 5 years of `infl`, clipped to the training range.
`scripts/export_artifacts.py` re-creates the scaler parameters from `data/gmd.csv` and proves them correct by
reproducing the published Fold-1 R² (Persistence 0.547, Random Forest 0.599, XGBoost 0.559 vs 0.560).

## Important notes

* The instruction PDF returned a hard-coded `4.75` for the ML models; this app runs the real models.
* **`Models/xgboost_fold1.pkl` is not the Fold-1 model.** On the Fold-1 test set it scores R² = 0.651, not the
  published 0.560, so it was trained on more data than the Fold-1 training set. The dashboard therefore uses
  an XGBoost re-trained with the notebook's exact settings (train ≤ 2000, 200 trees, depth 3, lr 0.1, seed 42;
  R² = 0.559) saved as `backend/models/xgboost_fold1.json`. The original pickle is left untouched.
* The study's main finding applies: Persistence beats the ML models, and none of them anticipates shocks
  (e.g. the 2021-22 inflation surge). Treat the numbers as research output.
