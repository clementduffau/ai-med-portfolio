from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# List of all NIH label names (will become binary columns 0/1)
DEFAULT_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def parse_labels(df, label_col, labels):
    """
    Convert the raw label string (pipe-separated) into binary multi-hot columns.
    Also creates 'has_finding' = 1 if any pathology (excluding 'No Finding') is present.
    """
    s = df[label_col].fillna("")
    # Normalize one label name to match column naming convention
    s = s.str.replace("Pleural Thickening", "Pleural_Thickening", regex=False)

    # Create one binary column per label
    for lab in labels:
        df[lab] = s.str.contains(rf"(^|\|){lab}(\||$)", regex=True).astype(np.uint8)

    # Any pathology (excluding "No Finding")?
    patho_labels = [l for l in labels if l != "No Finding"]
    df["has_finding"] = (df[patho_labels].sum(axis=1) > 0).astype(np.uint8)
    return df


def build_patient_split(df: pd.DataFrame, patient_col: str, seed: int,
                        train_ratio: float, val_ratio: float, test_ratio: float) -> pd.DataFrame:
    """
    Split the dataset at the PATIENT level to avoid leakage (same patient in multiple splits).
    Adds a 'split' column with values: train / val / test.
    """
    patients = df[patient_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    def assign(pid):
        if pid in train_p:
            return "train"
        elif pid in val_p:
            return "val"
        return "test"

    df["split"] = df[patient_col].map(assign)
    return df


def resolve_image_paths(df: pd.DataFrame, images_dir: Path, image_col: str) -> pd.DataFrame:
    """
    Scan the images folder and map each filename (Image Index) to its full path on disk.
    Adds 'image_path'. Rows with missing images are dropped.
    """
    all_imgs = list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.jpeg"))
    idx = {p.name: p for p in all_imgs}

    paths = []
    missing = []
    for name in df[image_col].astype(str).tolist():
        p = idx.get(name)
        if p is None:
            paths.append(None)
            missing.append(name)
        else:
            paths.append(str(p))

    df["image_path"] = paths
    if missing:
        df = df[df["image_path"].notna()].copy()
    return df


def add_class_id_and_filter(df: pd.DataFrame, selected_labels: list[str]) -> pd.DataFrame:
    """
    Build a single class label 'class_id' for CLASS-CONDITIONAL diffusion.
    Keeps only:
      - 'No Finding' alone
      - OR exactly one pathology among selected_labels (single-label samples)

    class_id mapping follows selected_labels order:
      0 = No Finding, 1 = Atelectasis, ...
    """
    if "No Finding" not in selected_labels:
        raise ValueError("selected_labels must include 'No Finding'.")

    patho = [l for l in selected_labels if l != "No Finding"]
    df["n_patho_selected"] = df[patho].sum(axis=1).astype(int)

    # Keep only clean single-label cases
    keep = ((df["No Finding"] == 1) & (df["n_patho_selected"] == 0)) | \
           ((df["No Finding"] == 0) & (df["n_patho_selected"] == 1))
    df = df[keep].copy()

    # Build class_id (0 for No Finding, otherwise the active pathology index)
    class_id = np.zeros(len(df), dtype=np.int64)
    for i, lab in enumerate(selected_labels):
        if lab == "No Finding":
            continue
        class_id[df[lab].values.astype(bool)] = i

    df["class_id"] = class_id
    return df


def main():
    """
    Main preprocessing pipeline:
      1) Read NIH CSV
      2) Parse labels -> multi-hot columns
      3) Filter to a small set of labels and create class_id (single-label only)
      4) Split by patient (train/val/test)
      5) Resolve image paths
      6) Save metadata.parquet and split CSVs
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--image-col", type=str, default="Image Index")
    ap.add_argument("--patient-col", type=str, default="Patient ID")
    ap.add_argument("--label-col", type=str, default="Finding Labels")
    ap.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    # Check required columns exist
    need = {args.image_col, args.patient_col, args.label_col}
    missing_cols = need - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}. Available: {list(df.columns)}")

    # Parse labels + create class_id for conditional generation
    df = parse_labels(df, label_col=args.label_col, labels=args.labels)
    selected = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax"]
    df = add_class_id_and_filter(df, selected_labels=selected)

    # Patient-level split + map filenames to actual file paths
    df = build_patient_split(df, patient_col=args.patient_col, seed=args.seed,
                             train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    df = resolve_image_paths(df, images_dir=args.images_dir, image_col=args.image_col)

    # Anti-leak sanity check: each patient must belong to a single split
    grp = df.groupby(args.patient_col)["split"].nunique()
    leaked = (grp > 1).sum()
    if leaked != 0:
        bad = grp[grp > 1].index[:10].tolist()
        raise RuntimeError(f"Patient leakage detected: {leaked}. Examples: {bad}")

    # Save full metadata + per-split CSVs
    meta_path = args.out_dir / "metadata.parquet"
    df.to_parquet(meta_path, index=False)

    for sp in ["train", "val", "test"]:
        sdf = df[df["split"] == sp].copy()
        out_csv = splits_dir / f"{sp}.csv"
        keep_cols = [args.image_col, args.patient_col, args.label_col,
                     "image_path", "split", "has_finding", "class_id"] + args.labels
        keep_cols = [c for c in keep_cols if c in sdf.columns]
        sdf[keep_cols].to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
