"""Training loop and evaluation utilities."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from src.models import CNN_LSTM_Attention


def train_epoch(model, loader, optimizer, criterion, device):
    """One epoch of training."""
    model.train()
    total, n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred, _ = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * X_b.size(0)
        n += X_b.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation: returns (loss, predictions, targets)."""
    model.eval()
    total, n = 0.0, 0
    preds, ys = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred, _ = model(X_b)
        loss = criterion(pred, y_b)
        total += loss.item() * X_b.size(0)
        n += X_b.size(0)
        preds.append(pred.cpu().numpy())
        ys.append(y_b.cpu().numpy())
    return total / n, np.concatenate(preds), np.concatenate(ys)


def train_cnn_lstm(
    X_train, y_train, X_test, y_test,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42
) -> Tuple[CNN_LSTM_Attention, dict]:
    """
    Full training pipeline with early stopping.
    
    Returns:
        (trained_model, history_dict)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    
    model = CNN_LSTM_Attention(
        n_features=X_train.shape[2],
        seq_len=X_train.shape[1]
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, _, _ = evaluate(model, test_loader, criterion, device)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        scheduler.step(vl)
        
        if vl < best_loss:
            best_loss = vl
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    return model, history