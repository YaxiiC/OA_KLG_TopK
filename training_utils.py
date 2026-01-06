"""
Training Utilities for KLGrade Prediction

This module contains:
- compute_metrics: Calculate accuracy, macro-F1, and QWK
- train_epoch: Train for one epoch
- validate: Validate model
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from models import JointModel

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and QWK."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # QWK (Quadratic Weighted Kappa)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "qwk": qwk
    }
    
    return metrics


def train_epoch(
    model: JointModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    warmup_epochs: int,
    lambda_k: float,
    lambda_k_start: float,
    warmup_thr_start: float,
    warmup_thr_end: float
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Update warmup parameters
    if epoch < warmup_epochs:
        # Linear interpolation
        alpha = epoch / warmup_epochs
        current_lambda_k = lambda_k_start + alpha * (lambda_k - lambda_k_start)
        current_threshold = warmup_thr_start + alpha * (warmup_thr_end - warmup_thr_start)
        model.set_warmup_threshold(current_threshold)
        model.set_use_hard_topk(False)
        is_warmup = True
    else:
        current_lambda_k = lambda_k
        model.set_use_hard_topk(True)
        is_warmup = False
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        radiomics = batch["radiomics"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits, gates = model(images, radiomics, return_gates=True)
        
        # Classification loss
        loss_cls = criterion(logits, labels)
        
        # Sparsity loss (target-k regularizer)
        p = torch.sigmoid(model.selector.gate_head(model.selector.backbone(images)))
        loss_k = ((p.sum(dim=1) - model.k) ** 2).mean()
        
        # Total loss
        loss = loss_cls + current_lambda_k * loss_k
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


def validate(
    model: JointModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            radiomics = batch["radiomics"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images, radiomics)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probas = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probas) if all_probas else None
    )
    
    results = {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probas": np.vstack(all_probas) if all_probas else np.array([])
    }
    
    return avg_loss, metrics, results

