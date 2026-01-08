
from dataset import Dataset_seg
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path


class DataModule_seg(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):

        if stage in (None,"fit"):
           
            self.train_ds = Dataset_seg(
                self.args.train_path, 
            )
            self.val_ds = Dataset_seg(
                self.args.val_path, 
            )
                   
        if stage == "test":
            self.test_ds = Dataset_seg(
                self.args.test_path,
                
            )
                
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size_train,
            num_workers=self.args.num_workers,
            shuffle=True,
            persistent_workers=self.args.persistent_workers,     
            pin_memory=True    
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            shuffle=False,
            persistent_workers=self.args.persistent_workers,     
            pin_memory=True  
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
