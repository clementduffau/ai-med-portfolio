from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class Dataset_diff(Dataset):
    """
    PyTorch Dataset for DDPM training on NIH Chest X-ray splits.

    Expected split CSV columns:
      - image_path: full path to the image file on disk
      - Patient ID: patient identifier (optional but recommended)
      - class_id: integer class label for conditional diffusion
      - One column per label in `labels` (0/1) if you want multi-label conditioning (optional)

    Output (per sample):
      - img: torch.Tensor of shape (C, H, W) normalized to [-1, 1]
      - class_id: int (or torch.long if you prefer)
      - image_path: str
      - pid: int (patient id or -1 if not available)
    """

    def __init__(
        self,
        split_csv: str,
        labels: Sequence[str],
        image_size: int,
        image_mode: str,
    ):
        """
        Args:
            split_csv: path to the train/val/test CSV (created by your preprocessing script)
            labels: list of label column names present in the CSV (e.g., "No Finding", "Effusion", ...)
            image_size: target square size (e.g., 256)
            image_mode: PIL mode, usually "L" (grayscale) for CXR or "RGB"
        """
        self.split_csv = Path(split_csv)
        self.df = pd.read_csv(split_csv)

        self.labels = list(labels)
        self.image_size = int(image_size)
        self.image_mode = str(image_mode)

        # Sanity check: make sure all label columns exist in the CSV
        for label in self.labels:
            if label not in self.df.columns:
                raise ValueError(f"label {label} not in {self.split_csv}")

        # Patient ID column name (NIH uses "Patient ID")
        self.patient_col = "Patient ID" if "Patient ID" in self.df.columns else "None"

    def __len__(self):
        """Return number of samples in the split."""
        return len(self.df)

    def _load_image(self, path) -> torch.Tensor:
        """
        Load an image from disk and preprocess it for diffusion.

        Steps:
          1) Load with PIL and convert to the desired mode (e.g., "L")
          2) Resize to (image_size, image_size)
          3) Convert to float32 in [0,1]
          4) Reorder to (C,H,W)
          5) Normalize to [-1, 1] (common for diffusion models)
        """
        img = Image.open(path).convert(self.image_mode)
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        img_arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

        # If grayscale: (H, W) -> (1, H, W)
        # If RGB: (H, W, 3) -> (3, H, W)
        if img_arr.ndim == 2:
            img_arr = img_arr[None, ...]
        else:
            img_arr = np.transpose(img_arr, (2, 0, 1))

        # Normalize to [-1, 1]
        img_arr = img_arr * 2.0 - 1.0
        return torch.from_numpy(img_arr)

    def _build_cond(self, row: pd.Series) -> torch.Tensor:
        """
        Build a conditioning vector from label columns.

        Note:
          - This is useful if you want multi-label conditioning (vector of size L).
          - For pure class-conditional diffusion, you might not need this, since you already have class_id.
        """
        # Should read values from the row for each label column
        return torch.tensor([float(row[lab]) for lab in self.labels], dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one training example as a dictionary.

        Keys:
          - img: preprocessed image tensor (C, H, W) in [-1, 1]
          - class_id: integer class label (used for conditional diffusion)
          - image_path: original path to the image
          - pid: patient id (or -1 if not available)
        """
        row = self.df.iloc[idx]

        image_path = str(row["image_path"])
        img = self._load_image(image_path)

        # Optional multi-label vector (not returned here, but available if needed)
        # cond = self._build_cond(row)

        # Patient id
        pid = int(row[self.patient_col]) if self.patient_col != "None" else -1

        # Class id for conditional diffusion
        class_id = int(row["class_id"])
        class_id = torch.tensor(class_id, dtype = torch.long)

        return {
            "image": img,
            "class_id": class_id,  
            "image_path": image_path,
            "pid": pid,
        }
