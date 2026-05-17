"""
Training Script
===============
Step 11 — Train the CrowdInstabilityDetector.

Loss      : BCEWithLogitsLoss
Optimizer : Adam
Epochs    : 10–20 (configurable)

Usage:
    python train.py --seq_dir sequences --epochs 15 --batch_size 16

The best model (lowest val loss) is saved to:
    model/best_model.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset  import get_dataloaders, CrowdDataset
from cnn_lstm import (
    CrowdInstabilityDetector,
    compute_motion_entropy,
    compute_directional_variance,
)


# ── run_epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for seq_batch, label_batch in loader:
            seq_batch   = seq_batch.to(device)    # (B, T, 1, 64, 64)
            label_batch = label_batch.to(device)  # (B,)

            # Compute motion metrics on CPU (no gradient needed)
            with torch.no_grad():
                mag_np  = seq_batch[:, :, 0, :, :].cpu().numpy()   # (B, T, 64, 64)
                ang_np  = np.zeros_like(mag_np, dtype=np.float32)  # angle fallback
                ent     = compute_motion_entropy(mag_np)            # (B, 1)
                var     = compute_directional_variance(ang_np)      # (B, 1)
                metrics = torch.from_numpy(
                    np.concatenate([ent, var], axis=1)
                ).to(device)                                        # (B, 2)

            logits = model(seq_batch, metrics)          # (B, 1)
            loss = criterion(logits.squeeze(-1), label_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * seq_batch.size(0)
            preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(label_batch.cpu().long().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    return avg_loss, accuracy


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    os.makedirs("model", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device : {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        seq_dir    = args.seq_dir,
        batch_size = args.batch_size,
        val_split  = args.val_split,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CrowdInstabilityDetector(
        seq_len    = 16,
        feature_dim= 128,
        hidden_dim = 256,
        num_layers = 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters : {total_params:,}")

    # ── Pos weight — read labels directly, DO NOT iterate __getitem__ ─────────
    # Get the underlying dataset (unwrap Subset if needed)
    base_dataset = (
        train_loader.dataset.dataset
        if hasattr(train_loader.dataset, "dataset")
        else train_loader.dataset
    )

    # Labels are stored as a plain list in CrowdDataset — read directly
    if hasattr(base_dataset, "labels"):
        label_array = np.array(base_dataset.labels)
    else:
        # Fallback: read from the Subset indices only (still fast, no file I/O)
        indices = train_loader.dataset.indices
        label_array = np.array([base_dataset.labels[i] for i in indices])

    n_pos      = max(int(label_array.sum()), 1)
    n_neg      = max(len(label_array) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"[Train] Class counts — normal={n_neg}  abnormal={n_pos}")
    print(f"[Train] Pos weight   : {pos_weight.item():.3f}")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs  |  batch={args.batch_size}  lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)

        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:>3}/{args.epochs}  |  "
            f"Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  |  "
            f"Val  loss={va_loss:.4f}  acc={va_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "optimizer"  : optimizer.state_dict(),
                    "val_loss"   : va_loss,
                    "val_acc"    : va_acc,
                    "args"       : vars(args),
                },
                "model/best_model.pt",
            )
            print(f"  ✓ Best model saved  (val_loss={va_loss:.4f}  val_acc={va_acc:.4f})")

    print(f"\n[Train] Done. Best val loss: {best_val_loss:.4f}")
    np.save("model/history.npy", history)
    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Crowd Instability Detector")
    p.add_argument("--seq_dir",    type=str,   default="sequences", help="Path to sequences dir")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val_split",  type=float, default=0.2)
    args = p.parse_args()
    train(args)