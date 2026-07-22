"""
Train the dual-branch fusion model (ECG Transformer + demographics).

Usage:
    # full fused model (proposal's main model)
    python -m src.train_fusion --data-dir data/processed --epochs 30

    # ablation: same architecture, demographic branch disabled
    python -m src.train_fusion --data-dir data/processed --epochs 30 --no-demo --run-name ecg_only_transformer

Saves the best checkpoint (by val macro F1) to checkpoints/<run_name>.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PTBXLDataset
from src.models.fusion_model import FusionModel

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_macro_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_demo: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_y, all_prob = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for ecg, demo, y, _ecg_id in tqdm(loader, leave=False):
            ecg, y = ecg.to(device), y.to(device)
            demo = demo.to(device) if use_demo else None

            logits = model(ecg, demo)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * ecg.size(0)
            all_y.append(y.cpu().numpy())
            all_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_prob)
    macro_f1 = compute_macro_f1(y_true, y_prob)
    return avg_loss, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train dual-branch fusion model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--no-demo", action="store_true", help="Disable demographic branch (ablation)")
    parser.add_argument("--run-name", type=str, default=None,
                         help="Checkpoint filename (default: fusion_model.pt or ecg_only_transformer.pt)")
    args = parser.parse_args()

    use_demo = not args.no_demo
    run_name = args.run_name or ("fusion_model" if use_demo else "ecg_only_transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} | use_demo={use_demo} | run_name={run_name}")

    train_ds = PTBXLDataset(args.data_dir / "train.npz")
    val_ds = PTBXLDataset(args.data_dir / "val.npz")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = FusionModel(n_classes=len(SUPERCLASSES), use_demo=use_demo).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True, use_demo=use_demo)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False, use_demo=use_demo)
        scheduler.step(val_f1)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = args.checkpoint_dir / f"{run_name}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "superclasses": SUPERCLASSES,
                "use_demo": use_demo,
            }, ckpt_path)
            print(f"  -> new best val macro F1 ({val_f1:.4f}), saved to {ckpt_path}")

    print(f"Training complete. Best val macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()