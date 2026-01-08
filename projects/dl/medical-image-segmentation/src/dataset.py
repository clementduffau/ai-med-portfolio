from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import random
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl



class Dataset_seg(Dataset):
    def __init__(self, split_dir: str):
        split_dir = Path(split_dir)
        self.imgs = sorted((split_dir / "images").glob("*.npy"))
        self.msks = sorted((split_dir / "masks").glob("*.npy"))
        assert len(self.imgs) == len(self.msks)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.load(self.imgs[idx]).astype(np.float32)  # [H,W]
        msk = np.load(self.msks[idx]).astype(np.int64)    # [H,W] 0/1/2

        x = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1)  # [3,H,W] : rgb image for cnn 
        y = torch.from_numpy(msk).long()                      # [H,W] : int64 for crossentropyloss
        return x, y



