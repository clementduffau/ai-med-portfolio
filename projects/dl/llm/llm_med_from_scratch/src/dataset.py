
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class BinCLMDataset(Dataset):
    """
    Random-window dataset for causal LM.

    Each item returns:
      x: [T] token ids
      y: [T] next-token ids (shifted by 1)

    We sample a random contiguous chunk from a memmapped array of tokens.
    """

    def __init__(
        self,
        bin_paths: List[Path],
        block_size: int,
        dtype: np.dtype = np.uint16,
        seed: int = 42,
        samples_per_epoch: int = 200_000,  # controls epoch length
    ):
        super().__init__()
        assert len(bin_paths) > 0, "bin_paths must be non-empty"
        self.bin_paths = bin_paths
        self.block_size = int(block_size)
        self.dtype = dtype
        self.samples_per_epoch = int(samples_per_epoch)

        self.rng = random.Random(seed)

        # memmap each shard (fast, low RAM)
        self.arrs = []
        self.sizes = []
        for p in self.bin_paths:
            p = Path(p)
            assert p.exists(), f"Missing bin file: {p}"
            arr = np.memmap(p, dtype=self.dtype, mode="r")
            assert arr.size > (self.block_size + 1), f"Bin too small for block_size: {p}"
            self.arrs.append(arr)
            self.sizes.append(arr.size)

        # weights to sample shards proportionally to size
        total = float(sum(self.sizes))
        self.weights = [s / total for s in self.sizes]

    def __len__(self) -> int:
        # We define an "epoch" as fixed number of random samples
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        # choose a shard by size
        shard_i = self._weighted_choice(self.weights, self.rng.random())
        arr = self.arrs[shard_i]
        n = arr.size

        # pick random start so that we can take block_size+1 tokens
        start = self.rng.randint(0, n - (self.block_size + 2))
        chunk = np.array(arr[start : start + self.block_size + 1], dtype=np.int64)

        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y

    @staticmethod
    def _weighted_choice(weights, r):
        # weights sum to 1
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1