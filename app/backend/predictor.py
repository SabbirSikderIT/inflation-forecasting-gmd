import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# --- Constants ---
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

FEATURES = ['infl', 'rGDP_pc', 'unemp', 'govdebt_GDP', 'CA_GDP', 'USDfx', 'REER', 'pop']
WINDOW = 10
HORIZON = 5

# --- Global Variables for Models ---
models = {}
preprocessors = {}
available_models_list = []
load_errors = {}

class InputError(Exception):
    pass

# --- PyTorch VAE Architecture (Must match training) ---
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc_mu = nn.Linear(32, 16)
        self.fc_logvar = nn.Linear(32, 16)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

# --- Load Models ---
def load_models():
    global available_models_list, load_errors
    
    try:
        # 1. Load Preprocessors
        preprocessors['imputer'] = joblib.load(MODEL_DIR / "imputer.pkl")
        preprocessors['scaler_X'] = joblib.load(MODEL_DIR / "scaler_X.pkl")
        preprocessors['scaler_y'] = joblib.load(MODEL_DIR / "scaler_y.pkl")
        
        with open(MODEL_DIR / "target_params.json", "r") as f:
            preprocessors['target_params'] = json.load(f)
            
        with open(MODEL_DIR / "ensemble_weights.json", "r") as f:
            preprocessors['ensemble_weights'] = json.load(f)

        # 2. Load ML Models
        models['AdaBoost'] = joblib.load(MODEL_DIR / "adaboost.pkl")
        models['Custom MLTE'] = joblib.load(MODEL_DIR / "custom_mlte.pkl")
        models['Custom VAE-Boost'] = joblib.load(MODEL_DIR / "custom_vae_boost.pkl")
        
        # 3. Load VAE Encoder (PyTorch)
        vae_enc = VAE_Encoder(WINDOW * len(FEATURES))
        vae_enc.load_state_dict(torch.load(MODEL_DIR / "vae_encoder.pth", map_location=torch.device('cpu')))
        vae_enc.eval()
        models['VAE_Encoder'] = vae_enc
        
        available_models_list = ['Ensemble'] + list(models.keys())[:-1] # Exclude VAE_Encoder from direct selection
        
    except Exception as e:
        load_errors['general'] = str(e)
        print(f"❌ Error loading models: {e}")

def available_models():
    return available_models_list

MODEL_LABELS = {
    "Ensemble": "Proposed AOWE Ensemble",
    "AdaBoost": "AdaBoost",
    "Custom MLTE": "Custom MLTE",
    "Custom VAE-Boost": "Custom VAE-Boost"
}

# --- Preprocess Data ---
def prepare(df):
    # Check required columns
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise InputError(f"Missing required columns: {missing_cols}")
        
    df_window = df[FEATURES].tail(WINDOW)
    if len(df_window) < WINDOW:
        raise InputError(f"At least {WINDOW} years of data required.")
        
    X_raw = preprocessors['imputer'].transform(df_window)
    X_scaled = preprocessors['scaler_X'].transform(X_raw)
    
    return {'df': df_window, 'X_scaled': X_scaled}

# --- Predict ---
def predict_all(window_data):
    X_scaled = window_data['X_scaled']
    X_flat = X_scaled.flatten().reshape(1, -1) # Shape: (1, 80)
    
    preds_scaled = {}
    
    # 1. AdaBoost
    preds_scaled['AdaBoost'] = models['AdaBoost'].predict(X_flat)[0]
    
    # 2. Custom MLTE
    preds_scaled['Custom MLTE'] = models['Custom MLTE'].predict(X_flat)[0]
    
    # 3. Custom VAE-Boost
    with torch.no_grad():
        mu, _ = models['VAE_Encoder'](torch.tensor(X_flat, dtype=torch.float32))
        X_hybrid = np.hstack((X_flat, mu.numpy()))
    preds_scaled['Custom VAE-Boost'] = models['Custom VAE-Boost'].predict(X_hybrid)[0]
    
    # 4. Ensemble (Weighted Average)
    ensemble_pred = 0
    for name, weight in preprocessors['ensemble_weights'].items():
        if name in preds_scaled:
            ensemble_pred += weight * preds_scaled[name]
    
    # 5. Inverse Transform all to Real Space (%)
    def inverse(pred):
        log_p = preprocessors['scaler_y'].inverse_transform([[pred]])[0][0]
        real_p = np.exp(log_p) - preprocessors['target_params']['shift']
        return np.clip(real_p, preprocessors['target_params']['t_lower'], preprocessors['target_params']['t_upper'])
    
    final_preds = {k: round(float(inverse(v)), 2) for k, v in preds_scaled.items()}
    final_preds['Ensemble'] = round(float(inverse(ensemble_pred)), 2)
    
    return final_preds
