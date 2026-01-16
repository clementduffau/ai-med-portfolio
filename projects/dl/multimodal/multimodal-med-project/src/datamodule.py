from datasets import load_from_disk
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from transformers import AutoProcessor
import torch


class DataModule_multi(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for multimodal image–text training (BLIP-style).

    Responsibilities:
      - Load a Hugging Face dataset saved on disk (train / validation / test splits)
      - Apply on-the-fly image preprocessing and text tokenization
      - Create PyTorch DataLoaders with proper batching and padding
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()  # Save args inside Lightning checkpoints
        self.args = args

        # Processor handles both image preprocessing and text tokenization
        self.processor = AutoProcessor.from_pretrained(args.model_name)

        # Datasets will be initialized in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        """
        Load dataset splits and attach transforms.
        Called automatically by PyTorch Lightning.
        """
        data_dir = Path(self.args.data_dir)
        ds = load_from_disk(data_dir)

        if stage in (None, "fit"):
            # Training split
            self.train_ds = ds["train"]
            self.train_ds.set_transform(self._transform)

            # Validation split
            self.val_ds = ds["validation"]
            self.val_ds.set_transform(self._transform)

        if stage == "test":
            # Test split
            self.test_ds = ds["test"]
            self.test_ds.set_transform(self._transform)

    def _transform(self, batch):
        """
        Transform a single HF dataset example:
          - Convert PIL image to normalized tensor
          - Tokenize caption text
          - Create labels for language modeling
        """
        image = batch["image"]
        text = batch.get("caption", batch.get("text", ""))

        # Apply image preprocessing + text tokenization
        enc = self.processor(
            image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
        )

        # Remove fake batch dimension (1, ...)
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        # Create labels for cross-entropy loss
        # Padding tokens are ignored by setting them to -100
        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        enc["labels"] = labels

        return enc

    def _collate_fn(self, batch):
        """
        Stack individual examples into a batch.
        All tensors have the same shape due to padding.
        """
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    def train_dataloader(self):
        """
        DataLoader for training.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size_train,
            shuffle=True,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """
        DataLoader for validation.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        """
        DataLoader for testing.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self._collate_fn,
        )
