from dataset import Dataset_diff
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
    

class DataModule_diff(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for diffusion training.

    Responsibilities:
      - Create Dataset objects for train/val/test from precomputed split CSV files
      - Create DataLoaders with the correct batch size, workers, shuffling, etc.
    """

    def __init__(self, args):
        
        super().__init__()
        self.save_hyperparameters()  # saves args into the Lightning checkpoint
        self.args = args

        # Will be initialized in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        processed_dir = Path(self.args.processed_dir)
        split_dir = processed_dir / "splits"

        # Paths to split CSV files
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        test_csv = split_dir / "test.csv"

        # Create train/val datasets when fitting
        if stage in (None, "fit"):
            self.train_ds = Dataset_diff(
                train_csv,
                self.args.labels,
                self.args.img_size,
                self.args.img_mode
            )
            self.val_ds = Dataset_diff(
                val_csv,
                self.args.labels,
                self.args.img_size,
                self.args.img_mode
            )

        # Create test dataset when testing
        if stage == "test":
            self.test_ds = Dataset_diff(
                test_csv,
                self.args.labels,
                self.args.img_size,
                self.args.img_mode
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
