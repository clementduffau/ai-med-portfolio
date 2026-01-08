
from pathlib import Path
import re
import torch
from functools import lru_cache
import nibabel as nib
import numpy as np




@staticmethod
def dice_macro(preds, targets, num_classes=3, eps=1e-6):
    # preds, targets: [B,H,W]
    dices = []
    for c in range(num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice)
    return torch.stack(dices).mean(), torch.stack(dices)


