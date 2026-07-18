"""
ETL pipeline for PTB-XL (Physikalisch-Technische Bundesanstalt XL ECG Dataset).

Extract  : load raw waveforms (via wfdb) + metadata CSV
Transform: bandpass + notch filter, per-lead z-score normalization,
           SCP-code -> 5-superclass multi-label mapping, demographic
           cleaning (BMI, imputation, encoding)
Load     : write one .npz per split (train/val/test) with:
           X        (N, 12, 1000)  float32   filtered/normalized ECG
           demo     (N, 5)         float32   [age, sex, height, weight, bmi] (standardized)
           y        (N, 5)         float32   multi-hot superclass labels
           ecg_id   (N,)           int64     PTB-XL record id, for traceability

Usage:
    python -m src.data.etl --raw-dir data/raw --out-dir data/processed \
        --config configs/etl_config.yaml
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #

def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Load ptbxl_database.csv and parse scp_codes into python dicts."""
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_statements(raw_dir: Path) -> pd.DataFrame:
    """Load scp_statements.csv, keep only rows that map to a diagnostic superclass."""
    scp_df = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    return scp_df[scp_df.diagnostic == 1]


def load_raw_signal(raw_dir: Path, row: pd.Series, sampling_rate: int) -> np.ndarray:
    """Load one record's 12-lead waveform via wfdb. Returns (12, signal_length)."""
    path_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    record_path = raw_dir / row[path_col]
    signal, _ = wfdb.rdsamp(str(record_path))
    return signal.T.astype(np.float32)  # (12, T)


# --------------------------------------------------------------------------- #
# Transform: signal processing
# --------------------------------------------------------------------------- #

def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass, applied per lead. signal: (12, T)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return np.stack([filtfilt(b, a, lead) for lead in signal])


def notch_filter(signal: np.ndarray, fs: float, notch_hz: float, q: float = 30.0) -> np.ndarray:
    """Remove powerline interference (50/60 Hz) per lead."""
    b, a = iirnotch(notch_hz, q, fs)
    return np.stack([filtfilt(b, a, lead) for lead in signal])


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalization. signal: (12, T)."""
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-8
    return (signal - mean) / std


def preprocess_signal(signal: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    signal = bandpass_filter(
        signal, fs,
        cfg["filtering"]["bandpass_low_hz"],
        cfg["filtering"]["bandpass_high_hz"],
        cfg["filtering"]["filter_order"],
    )
    signal = notch_filter(signal, fs, cfg["filtering"]["notch_hz"])
    signal = zscore_normalize(signal)
    return signal.astype(np.float32)


# --------------------------------------------------------------------------- #
# Transform: labels
# --------------------------------------------------------------------------- #

def aggregate_superclass(scp_codes: dict, scp_df: pd.DataFrame, superclasses: list) -> list:
    """Map a record's scp_codes dict to a set of the 5 diagnostic superclasses."""
    classes = set()
    for code in scp_codes.keys():
        if code in scp_df.index:
            sc = scp_df.loc[code, "diagnostic_class"]
            if sc in superclasses:
                classes.add(sc)
    return sorted(classes)


def build_multihot(df: pd.DataFrame, superclasses: list) -> np.ndarray:
    """(N, 5) multi-hot label matrix from df['diagnostic_superclass'] (list per row)."""
    y = np.zeros((len(df), len(superclasses)), dtype=np.float32)
    for i, classes in enumerate(df["diagnostic_superclass"]):
        for c in classes:
            y[i, superclasses.index(c)] = 1.0
    return y


# --------------------------------------------------------------------------- #
# Transform: demographics
# --------------------------------------------------------------------------- #

def clean_demographics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    demo = df[["age", "sex", "height", "weight"]].copy()

    strategy = cfg["demographics"]["impute_strategy"]
    for col in ["age", "height", "weight"]:
        if strategy == "median":
            demo[col] = demo[col].fillna(demo[col].median())
        else:
            demo[col] = demo[col].fillna(demo[col].mean())
    demo["sex"] = demo["sex"].fillna(demo["sex"].mode()[0])

    if cfg["demographics"]["compute_bmi"]:
        demo["bmi"] = demo["weight"] / ((demo["height"] / 100.0) ** 2)
        demo["bmi"] = demo["bmi"].fillna(demo["bmi"].median())

    return demo


# --------------------------------------------------------------------------- #
# Load: assemble splits and write to disk
# --------------------------------------------------------------------------- #

def run_etl(raw_dir: Path, out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = cfg["data"]["sampling_rate"]
    superclasses = cfg["labels"]["superclasses"]

    print("[1/5] Loading metadata ...")
    df = load_metadata(raw_dir)
    scp_df = load_scp_statements(raw_dir)

    print("[2/5] Aggregating diagnostic superclasses ...")
    df["diagnostic_superclass"] = df["scp_codes"].apply(
        lambda codes: aggregate_superclass(codes, scp_df, superclasses)
    )
    # drop records with no mapped superclass at all
    df = df[df["diagnostic_superclass"].map(len) > 0]

    print("[3/5] Cleaning demographics (age, sex, height, weight, BMI) ...")
    demo_df = clean_demographics(df, cfg)
    demo_scaler = StandardScaler()
    demo_values = demo_df[["age", "sex", "height", "weight", "bmi"]].values.astype(np.float32)

    print("[4/5] Building splits by strat_fold ...")
    fold_map = {
        "train": cfg["splits"]["train_folds"],
        "val": [cfg["splits"]["val_fold"]],
        "test": [cfg["splits"]["test_fold"]],
    }

    # fit demographic scaler on TRAIN ONLY to avoid leakage
    train_mask = df["strat_fold"].isin(fold_map["train"])
    demo_scaler.fit(demo_values[train_mask.values])

    for split, folds in fold_map.items():
        mask = df["strat_fold"].isin(folds)
        split_df = df[mask]
        split_demo = demo_scaler.transform(demo_values[mask.values])
        split_y = build_multihot(split_df, superclasses)

        print(f"[5/5] Processing {split} split: {len(split_df)} records ...")
        X = np.zeros((len(split_df), cfg["data"]["n_leads"], cfg["data"]["signal_length"]), dtype=np.float32)
        for i, (ecg_id, row) in enumerate(tqdm(split_df.iterrows(), total=len(split_df))):
            raw_signal = load_raw_signal(raw_dir, row, fs)
            X[i] = preprocess_signal(raw_signal, fs, cfg)

        np.savez_compressed(
            out_dir / f"{split}.npz",
            X=X,
            demo=split_demo.astype(np.float32),
            y=split_y,
            ecg_id=split_df.index.values,
        )
        print(f"  -> saved {out_dir / f'{split}.npz'}  X{X.shape} demo{split_demo.shape} y{split_y.shape}")

    print("ETL complete.")


def main():
    parser = argparse.ArgumentParser(description="PTB-XL ETL pipeline")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/etl_config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_etl(args.raw_dir, args.out_dir, cfg)


if __name__ == "__main__":
    main()
