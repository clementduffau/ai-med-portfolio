from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import BinCLMDataset

def collate_clm(batch):
    xs, ys = zip(*batch)  # each is [T]
    x = torch.stack(xs, dim=0)  # [B, T]
    y = torch.stack(ys, dim=0)  # [B, T]
    return {"input_ids": x, "labels": y}



class DataModuleLLMFromBins(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        tokens_dir = Path(self.args.tokens_dir)
        assert tokens_dir.exists(), f"Missing tokens_dir: {tokens_dir}"

        train_bins = sorted(tokens_dir.glob("train_*.bin"))
        val_bin = tokens_dir / "val.bin"
        test_bin = tokens_dir / "test.bin"

        assert train_bins, f"No train_*.bin found in {tokens_dir}"
        assert val_bin.exists(), f"Missing {val_bin}"
        assert test_bin.exists(), f"Missing {test_bin}"

        dtype = np.uint16 if self.args.dtype == "uint16" else np.uint32

        if stage in (None, "fit"):
            self.train_ds = BinCLMDataset(
                bin_paths=train_bins,
                block_size=self.args.block_size,
                dtype=dtype,
                seed=self.args.seed,
                samples_per_epoch=self.args.train_samples_per_epoch,
            )
            self.val_ds = BinCLMDataset(
                bin_paths=[val_bin],
                block_size=self.args.block_size,
                dtype=dtype,
                seed=self.args.seed + 1,
                samples_per_epoch=self.args.val_samples_per_epoch,
            )

        if stage in (None, "test"):
            self.test_ds = BinCLMDataset(
                bin_paths=[test_bin],
                block_size=self.args.block_size,
                dtype=dtype,
                seed=self.args.seed + 2,
                samples_per_epoch=self.args.test_samples_per_epoch,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size_train,
            shuffle=False,  # IMPORTANT: dataset itself is random
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=True,
            collate_fn=collate_clm,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=True,
            collate_fn=collate_clm,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=True,
            collate_fn=collate_clm,
        )