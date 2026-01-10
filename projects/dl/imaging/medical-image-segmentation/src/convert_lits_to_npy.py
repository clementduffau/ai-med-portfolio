from __future__ import annotations

from pathlib import Path
import re
import random
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm


def extract_id(p: Path, prefix: str) -> int | None:
    """Extract numeric id from filename like 'volume-12.nii' or 'segmentation-12.nii'."""
    m = re.search(rf"{prefix}-(\d+)\.nii(\.gz)?$", p.name)
    return int(m.group(1)) if m else None


def build_pairs(root: Path):
    """Find all volumes and masks, then match them by id."""
    vol_files = sorted(list(root.rglob("volume-*.nii")) + list(root.rglob("volume-*.nii.gz")))
    seg_files = sorted(
        list((root / "segmentations").rglob("segmentation-*.nii")) +
        list((root / "segmentations").rglob("segmentation-*.nii.gz"))
    )

    vol_map = {extract_id(p, "volume"): p for p in vol_files}
    seg_map = {extract_id(p, "segmentation"): p for p in seg_files}

    # Keep only ids that exist in both maps
    common = sorted(set(vol_map.keys()) & set(seg_map.keys()))
    pairs = [(i, vol_map[i], seg_map[i]) for i in common if i is not None]
    return pairs


def load_nii_fast(path: Path) -> np.ndarray:
    """Load NIfTI file as numpy array (faster than get_fdata())."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    return arr


def choose_slices(mask3d: np.ndarray, only_with_mask: bool, empty_ratio: float, rng: np.random.Generator):
    """
    Select slice indices (z) to export.
    - If only_with_mask: keep slices where mask is non-empty
    - Optionally add a fraction of empty slices (empty_ratio)
    """
    z_all = np.arange(mask3d.shape[2])
    z_with = np.where(mask3d.sum(axis=(0, 1)) > 0)[0]

    if not only_with_mask:
        return z_all.astype(int).tolist()

    z_keep = list(z_with.astype(int))
    z_empty = np.array([z for z in z_all if z not in set(z_keep)], dtype=int)

    if len(z_keep) > 0 and len(z_empty) > 0 and empty_ratio > 0:
        n_add = int(len(z_keep) * empty_ratio)
        n_add = min(n_add, len(z_empty))
        if n_add > 0:
            add = rng.choice(z_empty, size=n_add, replace=False)
            z_keep.extend(add.tolist())

    return z_keep


def normalize_ct_slice(vol2d: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Clip HU values then normalize to [0, 1]."""
    vol2d = np.clip(vol2d, clip_min, clip_max)
    vol2d = (vol2d - vol2d.min()) / (vol2d.max() - vol2d.min() + 1e-8)
    return vol2d.astype(np.float32)


def main(
    root: str,
    out_dir: str,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    only_with_mask: bool = True,
    empty_ratio: float = 0.1,
    clip_min: float = -200.0,
    clip_max: float = 250.0,
):
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build (patient_id, volume_path, mask_path) pairs
    pairs = build_pairs(root)
    assert len(pairs) > 0, f"No pairs found under {root}"

    # Shuffle and split by patient/volume (to avoid leakage across splits)
    rng_py = random.Random(seed)
    pairs_shuf = pairs.copy()
    rng_py.shuffle(pairs_shuf)

    n = len(pairs_shuf)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    assert n_train + n_val < n

    splits = {
        "train": pairs_shuf[:n_train],
        "val": pairs_shuf[n_train:n_train + n_val],
        "test": pairs_shuf[n_train + n_val:],
    }

    rng = np.random.default_rng(seed)
    rows = []

    # Export slices to npy per split
    for split_name, split_pairs in splits.items():
        img_dir = out / split_name / "images"
        msk_dir = out / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n== {split_name}: {len(split_pairs)} volumes ==")

        for pid, vol_path, seg_path in tqdm(split_pairs, desc=f"Export {split_name}"):
            vol3d = load_nii_fast(vol_path)  # [H, W, D]
            msk3d = load_nii_fast(seg_path)  # [H, W, D]

            # Select which slices to export
            z_list = choose_slices(msk3d, only_with_mask, empty_ratio, rng)

            for z in z_list:
                vol2d = vol3d[:, :, z]
                msk2d = msk3d[:, :, z]

                # Normalize CT slice
                vol2d = normalize_ct_slice(vol2d, clip_min, clip_max)

                # Keep mask values in {0, 1, 2} as uint8
                msk2d = np.clip(msk2d, 0, 2).astype(np.uint8)

                # Save files
                stem = f"p{pid:03d}_z{z:03d}"
                img_path = img_dir / f"{stem}.npy"
                msk_path = msk_dir / f"{stem}.npy"

                np.save(img_path, vol2d)  # [H,W] float32
                np.save(msk_path, msk2d)  # [H,W] uint8

                # Store metadata row
                rows.append({
                    "split": split_name,
                    "patient_id": pid,
                    "z": z,
                    "image_path": str(img_path),
                    "mask_path": str(msk_path),
                    "has_mask": int(msk2d.sum() > 0),
                })

    # Save metadata for later loading and analysis
    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)

    print("\nSaved:", out)
    print("metadata:", out / "metadata.csv")
    print(df.groupby("split")["has_mask"].agg(["count", "sum"]).rename(columns={"sum": "num_with_mask"}))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="LiTS root containing volume_pt*/ and segmentations/")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for npy dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--only_with_mask", action="store_true")
    ap.add_argument("--empty_ratio", type=float, default=0.1)
    ap.add_argument("--clip_min", type=float, default=-200.0)
    ap.add_argument("--clip_max", type=float, default=250.0)
    args = ap.parse_args()

    main(
        root=args.root,
        out_dir=args.out_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        only_with_mask=args.only_with_mask,
        empty_ratio=args.empty_ratio,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
