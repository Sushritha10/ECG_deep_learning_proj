"""
PyTorch Dataset wrapping the .npz files produced by src/data/etl.py.

Usage:
    from src.data.dataset import PTBXLDataset
    from torch.utils.data import DataLoader

    train_ds = PTBXLDataset("data/processed/train.npz")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for ecg, demo, y, ecg_id in train_loader:
        ...
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.X = data["X"]          # (N, 12, 1000)
        self.demo = data["demo"]    # (N, 5) -> age, sex, height, weight, bmi (standardized)
        self.y = data["y"]          # (N, 5) multi-hot superclass labels
        self.ecg_id = data["ecg_id"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ecg = torch.from_numpy(self.X[idx]).float()
        demo = torch.from_numpy(self.demo[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        ecg_id = int(self.ecg_id[idx])
        return ecg, demo, y, ecg_id
