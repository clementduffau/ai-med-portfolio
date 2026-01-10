
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import torch
from utils import build_text



class DataModule_llm(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for instruction-tuning a CausalLM.

    This module:
      1) Loads a preprocessed Hugging Face dataset from disk (DatasetDict).
      2) Tokenizes each example into:
         - input_ids
         - attention_mask
         - labels (same as input_ids but with -100 on prompt tokens)
      3) Provides train/val/test dataloaders using a custom collator that pads
         input_ids/attention_mask and pads labels with -100.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        self.collator = collate_custom(self.tokenizer)
        
        self.ds = None
        
    def setup(self, stage=None):
        self.ds = load_dataset("json", data_files={"train": str(self.args.train_data_dir), 
                                                   "validation": str(self.args.val_data_dir), 
                                                   "test": str(self.args.test_data_dir)})
        self.ds = self.ds.map(build_text)
        # Remove all original columns after tokenization (keep only input_ids/attention_mask/labels)
        remove_cols = self.ds["train"].column_names
        
        def tokenize_and_expand(batch):
            """
            Tokenize a batch of examples and create labels for causal LM training.

            We build:
              - full = tokenizer(text = prompt + answer)
              - prompt = tokenizer(prompt only)
            Then:
              labels = input_ids
              labels[:prompt_len] = -100  (ignore prompt tokens in the loss)

            NOTE:
              - We do NOT pad here (padding is done in the collator at batch time).
              - If max_length is too small, the answer part may get truncated.
            """
            # Tokenize the concatenation: (prompt + answer)
            full = self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.args.max_length,
                # stride=args.stride,
                # return_overflowing_tokens=True,
            )
            # Tokenize prompt only to get prompt token lengths
            prompt = self.tokenizer(
                batch["input"],
                truncation = True,
                max_length = self.args.max_length,
            )
            input_ids = full["input_ids"]
            attention_mask = full["attention_mask"]
            prompt_len = [len(x) for x in prompt["input_ids"]]
            # Build labels with -100 on prompt tokens
            labels = []
            for input_ids_i, p_len in zip(input_ids, prompt_len):
                labels_i = input_ids_i.copy()
                labels_i[:p_len] = [-100]*p_len
                labels.append(labels_i)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        # Apply tokenization to all splits.
        # batched=True means "batch" is a dict of lists.
        self.ds = self.ds.map(
            tokenize_and_expand,
            batched=True,                      
            remove_columns=remove_cols # remove raw text after tokenization
        )
        
        # self.ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

                
    def train_dataloader(self):
        return DataLoader(
            self.ds["train"],
            batch_size=self.args.batch_size_train,
            num_workers=self.args.num_workers,
            shuffle=True,
            persistent_workers=self.args.persistent_workers,     
            pin_memory=True,
            collate_fn = self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds["validation"],
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            shuffle=False,
            persistent_workers=self.args.persistent_workers,     
            pin_memory=True,
            collate_fn = self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds["test"],
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn = self.collator,
        )
        
        
    

class collate_custom():
    def __init__(self, tokenizer):
        """
    Custom collator for causal language modeling with instruction-tuning masking.

    - Pads input_ids / attention_mask using tokenizer.pad
    - Pads labels to the same sequence length using -100 (ignored by the loss)
    """
        self.tokenizer = tokenizer
    
        
    def __call__(self,features):
        """
        features: list of dicts, each dict contains:
          - input_ids: List[int]
          - attention_mask: List[int]
          - labels: List[int] (same length as input_ids, with -100 on the prompt tokens)
        returns: dict of torch tensors:
          - input_ids: LongTensor [B, T]
          - attention_mask: LongTensor [B, T]
          - labels: LongTensor [B, T]
        """
        labels = [f["labels"] for f in features]
        # Remove labels before padding input_ids/attention_mask
        #    (tokenizer.pad doesn't know how to pad our labels with -100)
        features_wo_labels = []
        for f in features:
            f = dict(f)          # avoid mutating the original dict
            f.pop("labels", None)
            features_wo_labels.append(f)
            
        batch = self.tokenizer.pad(features_wo_labels, padding= True,  return_tensors = "pt")
        T = batch["input_ids"].shape[1]
        # Pad labels to the same length T using -100
        labels_padded_list = []
        for labels_i in labels:
            pad_len = T - len(labels_i)
            labels_i_padded = labels_i + [-100]*pad_len
            labels_padded_list.append(labels_i_padded)
        labels_padded = torch.tensor(labels_padded_list, dtype = torch.long)
        batch["labels"] = labels_padded
        return batch
        