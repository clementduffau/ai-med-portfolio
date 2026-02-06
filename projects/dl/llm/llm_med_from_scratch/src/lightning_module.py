import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from model import GPT

class llm_pl(pl.LightningModule):
    def __init__(
        self,
        cfg_model,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = float(cfg_model.lr)
        self.weight_decay = float(cfg_model.weight_decay)
        self.adam_beta1 = float(cfg_model.adam_beta1)
        self.adam_beta2 = float(cfg_model.adam_beta2)
        self.warmup_ratio = float(cfg_model.warmup_ratio)

        # Model
        self.model = GPT(cfg_model)
        
        np.random.seed(42)
        

    def forward(self, input_ids, labels= None):
            return self.model(input_ids, targets=labels)


    def training_step(self, batch, batch_idx):
        """
        - input_ids: [B, L]
        - labels: [B, L]   
        """
        logits, loss = self.model(batch["input_ids"], targets = batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        logits, loss = self.forward(batch["input_ids"], batch["labels"])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        ppl = torch.exp(loss.detach())
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
        

    def configure_optimizers(self):
    
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay = self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps = total_steps
            ),
            "interval": "step", 
        }
        
    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
