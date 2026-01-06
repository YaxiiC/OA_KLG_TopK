"""
Joint Training: Image-Conditioned Feature Selector + KLGrade Classifier

Main training script that orchestrates:
1. Data loading and preprocessing
2. Model initialization
3. Training loop with two-stage gating
4. Inference and saving results

Example usage:
    python train_selector_klgrade.py `
        --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
        --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
        --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_train\radiomics_results.csv" `
        --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_test\radiomics_results.csv" `
        --klgrade-train-csv "C:\Users\chris\MICCAI2026\labels\klgrade_train.csv" `
        --outdir "output_training" `
        --k 15 `
        --warmup-epochs 20 `
        --epochs 100 `
        --batch-size 8 `
        --lr 1e-4
"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import from separate modules
from models import JointModel
from data_loader import (
    load_radiomics_long_format,
    load_klgrade_labels,
    KLGradeDataset
)
from training_utils import train_epoch, validate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train image-conditioned feature selector + KLGrade classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument("--images-tr", type=str, required=True,
                        help="Training images directory")
    parser.add_argument("--images-ts", type=str, required=True,
                        help="Test images directory")
    parser.add_argument("--radiomics-train-csv", type=str, required=True,
                        help="Training radiomics CSV (long format)")
    parser.add_argument("--radiomics-test-csv", type=str, required=True,
                        help="Test radiomics CSV (long format)")
    parser.add_argument("--klgrade-train-csv", type=str, required=True,
                        help="Training KLGrade labels CSV")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    
    # Model hyperparameters
    parser.add_argument("--k", type=int, default=15,
                        help="Top-k features to select")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained backbone")
    
    # Training hyperparameters
    parser.add_argument("--warmup-epochs", type=int, default=20,
                        help="Number of warmup epochs")
    parser.add_argument("--warmup-thr-start", type=float, default=0.0,
                        help="Warmup threshold start")
    parser.add_argument("--warmup-thr-end", type=float, default=0.5,
                        help="Warmup threshold end")
    parser.add_argument("--lambda-k-start", type=float, default=0.005,
                        help="Lambda_k start value")
    parser.add_argument("--lambda-k", type=float, default=0.05,
                        help="Lambda_k end value")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio")
    
    # Other
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)
    
    # Load radiomics
    radiomics_train, roi_names, feature_names, missing_stats_train = load_radiomics_long_format(
        Path(args.radiomics_train_csv)
    )
    radiomics_test, _, _, missing_stats_test = load_radiomics_long_format(
        Path(args.radiomics_test_csv),
        expected_rois=roi_names,
        expected_features=feature_names
    )
    
    # Save feature mapping
    feature_mapping = {}
    for idx, (roi, feat) in enumerate([(r, f) for r in roi_names for f in feature_names]):
        feature_mapping[idx] = f"{roi}:{feat}"
    
    with open(outdir / "feature_names.json", "w") as f:
        json.dump(feature_mapping, f, indent=2)
    logger.info(f"Saved feature mapping to {outdir / 'feature_names.json'}")
    
    # Load labels
    labels_train = load_klgrade_labels(Path(args.klgrade_train_csv))
    
    # Get common case IDs
    train_case_ids = list(set(radiomics_train.keys()) & set(labels_train.keys()))
    logger.info(f"Training cases with both radiomics and labels: {len(train_case_ids)}")
    
    # Stratified train/val split
    train_ids, val_ids = train_test_split(
        train_case_ids,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[labels_train[cid] for cid in train_case_ids]
    )
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # Normalize radiomics (fit on train only)
    logger.info("Fitting radiomics scaler on training set...")
    train_radiomics_array = np.array([radiomics_train[cid] for cid in train_ids])
    scaler = StandardScaler()
    scaler.fit(train_radiomics_array)
    
    # Apply scaler
    for cid in train_ids:
        radiomics_train[cid] = scaler.transform(radiomics_train[cid].reshape(1, -1))[0]
    for cid in val_ids:
        if cid in radiomics_train:
            radiomics_train[cid] = scaler.transform(radiomics_train[cid].reshape(1, -1))[0]
    for cid in radiomics_test.keys():
        radiomics_test[cid] = scaler.transform(radiomics_test[cid].reshape(1, -1))[0]
    
    # Save scaler
    joblib.dump(scaler, outdir / "radiomics_scaler.joblib")
    logger.info(f"Saved scaler to {outdir / 'radiomics_scaler.joblib'}")
    
    # Create datasets
    train_dataset = KLGradeDataset(
        train_ids,
        Path(args.images_tr),
        radiomics_train,
        labels_train,
        target_shape=(32, 128, 128)
    )
    val_dataset = KLGradeDataset(
        val_ids,
        Path(args.images_tr),
        radiomics_train,
        labels_train,
        target_shape=(32, 128, 128)
    )
    test_dataset = KLGradeDataset(
        list(radiomics_test.keys()),
        Path(args.images_ts),
        radiomics_test,
        labels_dict=None,
        target_shape=(32, 128, 128)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Create model
    n_features = len(feature_mapping)
    logger.info(f"Model will gate {n_features} features, selecting top-{args.k}")
    
    model = JointModel(
        n_features=n_features,
        n_classes=5,
        pretrained=args.pretrained,
        k=args.k,
        warmup_threshold=args.warmup_thr_start,
        use_hard_topk=False
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    best_val_f1 = -1.0
    patience_counter = 0
    train_history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.warmup_epochs,
            args.lambda_k,
            args.lambda_k_start,
            args.warmup_thr_start,
            args.warmup_thr_end
        )
        
        # Validate
        val_loss, val_metrics, _ = validate(model, val_loader, criterion, device)
        
        # Log
        is_warmup = epoch <= args.warmup_epochs
        stage = "WARMUP" if is_warmup else "HARD-TOPK"
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} [{stage}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val QWK: {val_metrics['qwk']:.4f}"
        )
        
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["qwk"]
        })
        
        # Early stopping
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "args": vars(args)
            }, outdir / "best_model.pt")
            logger.info(f"  → New best val F1: {best_val_f1:.4f}, saved model")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Log stage switch
        if epoch == args.warmup_epochs:
            logger.info("=" * 80)
            logger.info("SWITCHING FROM WARMUP TO HARD TOP-K")
            logger.info("=" * 80)
    
    # Save training history
    pd.DataFrame(train_history).to_csv(outdir / "training_history.csv", index=False)
    
    # Load best model for inference
    logger.info("Loading best model for inference...")
    checkpoint = torch.load(outdir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Inference on train and test
    logger.info("=" * 80)
    logger.info("Running inference...")
    logger.info("=" * 80)
    
    model.set_use_hard_topk(True)  # Use hard top-k for inference
    
    for split_name, dataset, loader in [("train", train_dataset, train_loader),
                                         ("test", test_dataset, test_loader)]:
        logger.info(f"Processing {split_name} set...")
        
        model.eval()
        all_case_ids = []
        all_preds = []
        all_probas = []
        all_selected_features = []
        
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                radiomics = batch["radiomics"].to(device)
                case_ids = batch["case_id"]
                
                logits, p = model(images, radiomics, return_gates=True)
                probas = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                # Get top-k features for each sample
                _, topk_indices = torch.topk(p, args.k, dim=1)
                
                for i in range(len(case_ids)):
                    case_id = case_ids[i]
                    all_case_ids.append(case_id)
                    all_preds.append(preds[i].item())
                    all_probas.append(probas[i].cpu().numpy())
                    
                    # Get selected features
                    topk_idx = topk_indices[i].cpu().numpy()
                    topk_names = [feature_mapping[int(idx)] for idx in topk_idx]
                    topk_gates = p[i][topk_indices[i]].cpu().numpy().tolist()
                    
                    all_selected_features.append({
                        "case_id": case_id,
                        "topk_indices": topk_idx.tolist(),
                        "topk_names": topk_names,
                        "topk_gates": topk_gates
                    })
        
        # Save predictions
        pred_df = pd.DataFrame({
            "case_id": all_case_ids,
            "pred_class": all_preds,
            "prob_0": [p[0] for p in all_probas],
            "prob_1": [p[1] for p in all_probas],
            "prob_2": [p[2] for p in all_probas],
            "prob_3": [p[3] for p in all_probas],
            "prob_4": [p[4] for p in all_probas]
        })
        pred_df.to_csv(outdir / f"predictions_{split_name}.csv", index=False)
        logger.info(f"Saved predictions to {outdir / f'predictions_{split_name}.csv'}")
        
        # Save selected features
        with open(outdir / f"selected_features_{split_name}.json", "w") as f:
            json.dump(all_selected_features, f, indent=2)
        logger.info(f"Saved selected features to {outdir / f'selected_features_{split_name}.json'}")
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
