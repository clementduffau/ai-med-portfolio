import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from torchmetrics.classification import MulticlassJaccardIndex
from utils import dice_macro
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau


class Modelelightning_seg(pl.LightningModule):
    def __init__(
        self,
        cfg_model
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = float(cfg_model.lr)
        self.adam_beta1 = float(cfg_model.adam_beta1)
        self.adam_beta2 = float(cfg_model.adam_beta2)
        self.scheduler_type = cfg_model.scheduler_type

        # Model
        self.num_classes = 3
        
        self.model =smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,     
            classes=self.num_classes,         # 0 back, 1 liver, 2 tumor
            activation=None
        )
        
        self.dice_loss = smp.losses.DiceLoss(
            mode = "multiclass",
            from_logits = True
        )

        self.entropy_loss = nn.CrossEntropyLoss()

        self.iou = MulticlassJaccardIndex(num_classes=self.num_classes, average="macro") # macro : classes not balanced
        np.random.seed(42)
        


    def forward(self, x):
            return self.model(x)


    def _loss_fn(self,logits, targets):
        return 0.5*self.entropy_loss(logits, targets) + 0.5 * self.dice_loss(logits, targets)
    

    def training_step(self, batch):
        
        x, y = batch
        logits = self.forward(x)

        loss = self._loss_fn(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch):
        with torch.no_grad():
            x, y = batch

            logits = self.forward(x)

            preds = torch.argmax(logits, dim = 1) # [B,H,W]
            loss = self._loss_fn(logits, y)

            iou = self.iou(preds, y)

            self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/iou", iou, on_epoch=True, prog_bar=True, sync_dist=True)

            dice_mean, dice_per_class = dice_macro(preds, y, num_classes=self.num_classes)

            self.log("val/dice", dice_mean, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/dice_liver", dice_per_class[1], on_epoch=True, sync_dist=True)
            self.log("val/dice_tumor", dice_per_class[2], on_epoch=True, sync_dist=True)

        return {"val_loss": loss, "val_iou": iou}


    def configure_optimizers(self):


        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
        )


        if self.scheduler_type == "plateau":
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=2,
                ),
                "monitor": "val/loss", 
                "interval": "epoch", 
                "frequency": 1,
            }
        
        elif self.scheduler_type == "None":
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
