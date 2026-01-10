import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

class Clinicalbert_pl(pl.LightningModule):
    def __init__(
        self,
        cfg_model
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg_model.model_name
        self.lr = float(cfg_model.lr)
        self.num_labels = int(cfg_model.num_labels)
        self.adam_beta1 = float(cfg_model.adam_beta1)
        self.adam_beta2 = float(cfg_model.adam_beta2)
        self.warmup_ratio = float(cfg_model.warmup_ratio)

        # Model
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels = self.num_labels,
            problem_type = "multi_label_classification"
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_logits = []
        self.val_labels = []
        self.val_note_map = []
        
        np.random.seed(42)
        

    def forward(self, **batch):
            return self.model(**batch)


    def training_step(self, batch):
        """
        - input_ids: [B, L]
        - attention_mask: [B, L]
        - labels: [B, C]   (multi-label)
        """
        labels = batch.pop("labels") # remove labels from batch
        note_map = batch.pop("note_map")
        outputs = self.model(**batch)
        logits = outputs.logits # [B, C]
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch):
        with torch.no_grad():
            labels = batch.pop("labels")
            note_map = batch.pop("note_map") # [B] mapping each chunk -> original note id (or index)
            outputs = self.model(**batch)
            logits = outputs.logits
            loss = self.loss_fn(logits, labels)
    
            self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            
            self.val_logits.append(logits.detach().cpu())
            self.val_labels.append(labels.detach().cpu())
            self.val_note_map.append(note_map.detach().cpu())
            
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """
        If one clinical note is split into multiple chunks, the model produces multiple
        predictions (one per chunk). To evaluate correctly "per note", we must aggregate
        chunk predictions into a single prediction for the original note.

        We do max pooling on probabilities:
          - probs_note[note] = max over chunks(prob_chunk)
        This matches the intuition: if any chunk contains evidence for a condition,
        the whole note should be positive for that label.
        """
        logits = torch.cat(self.val_logits, dim=0)
        y_true = torch.cat(self.val_labels, dim=0).numpy()
        
        probs = torch.sigmoid(logits.float()).cpu()
        note_map = torch.cat(self.val_note_map, dim=0).numpy()
        unique_note = np.unique(note_map)
        C = probs.shape[1]
        
        probs_note = np.zeros((len(unique_note), C), dtype = np.float32)
        y_true_note = np.zeros((len(unique_note), C), dtype = np.int32)
        
        note_to_row = {nid : i for i, nid in enumerate(unique_note)}
        
        for i in range(probs.shape[0]):
            r = note_to_row[note_map[i]]
            probs_note[r] = np.maximum(probs_note[r], probs[i])
            y_true_note[r] = np.maximum(y_true_note[r], y_true[i])
            
        y_pred_note = (probs_note >= 0.5).astype(np.int32)
        f1_micro = f1_score(y_true_note, y_pred_note, average="micro", zero_division=0)
        f1_macro = f1_score(y_true_note, y_pred_note, average="macro", zero_division=0)
        subset_acc = accuracy_score(y_true_note, y_pred_note) 
        
        self.log("val/f1_micro", f1_micro, prog_bar=True)
        self.log("val/f1_macro", f1_macro, prog_bar=False)
        self.log("val/subset_acc", subset_acc, prog_bar=False)
        
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_note_map.clear()
        

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
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
    
