
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import torch

class DataModule_nlp(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.ds = None
        
    def setup(self, stage=None):
        self.ds = load_from_disk(self.args.data_dir)
                
        def tokenize_and_expand(batch):
            enc = self.tokenizer(
                batch[self.args.text_col],
                truncation=True,
                max_length=self.args.max_length,
                stride=self.args.stride,
                return_overflowing_tokens=True,
            )

            # For each produced chunk, tells which original example in this batch it comes from
            mapping = enc.pop("overflow_to_sample_mapping")  # list[int], length = num_chunks

            # Duplicate labels + note_id for each chunk
            enc[self.args.label_col] = [batch[self.args.label_col][i] for i in mapping]
            enc["note_id"] = [batch["note_id"][i] for i in mapping]

            return enc

        remove_cols = self.ds["train"].column_names
        self.ds = self.ds.map(
            tokenize_and_expand,
            batched=True,                      
            remove_columns=remove_cols # remove raw text after tokenization
        )
        
        self.ds.set_format(type="torch", columns=["input_ids", "attention_mask", self.args.label_col, "note_id"])

                
    def train_dataloader(self):
        return DataLoader(
            self.ds["train"],
            batch_size=self.args.batch_size_train,
            num_workers=self.args.num_workers,
            shuffle=True,
            persistent_workers=self.args.persistent_workers,     
            pin_memory=True,
            collate_fn = self._collate 
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds["validation"],
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            shuffle=False,
            persistent_workers=self.args.persistent_workers,     
            pin_memory=True,
            collate_fn = self._collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds["test"],
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn = self._collate
        )
        
    def _collate(self, features):
        labels = torch.stack([f[self.args.label_col] for f in features]).float()
        note_map = torch.tensor([int(f["note_id"]) for f in features], dtype=torch.long)
        for f in features:
            del f[self.args.label_col]
            del f["note_id"]
        batch = self.collator(features)
        batch["labels"] = labels
        batch["note_map"] = note_map
        return batch
        
        
