import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class llm_pl(pl.LightningModule):
    def __init__(
        self,
        cfg_model,
        cfg_lora,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg_model.model_name
        self.lr = float(cfg_model.lr)
        self.weight_decay = float(cfg_model.weight_decay)
        self.adam_beta1 = float(cfg_model.adam_beta1)
        self.adam_beta2 = float(cfg_model.adam_beta2)
        self.warmup_ratio = float(cfg_model.warmup_ratio)
        self.r = int(cfg_lora.r)
        self.lora_alpha = float(cfg_lora.lora_alpha)
        self.target_modules = list(cfg_lora.target_modules)
        self.lora_dropout = float(cfg_lora.lora_dropout)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token
        
        print("model token id",self.model.config.pad_token_id)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        parameters = []
        parameters_trainable = []
        for _, p in self.model.named_parameters():
            parameters.append(p.numel())
            if p.requires_grad:
                parameters_trainable.append(p.numel())
                
        print("ratio trainable parameters", (sum(parameters_trainable)/sum(parameters))*100)
        
        np.random.seed(42)
        

    def forward(self, **batch):
            return self.model(**batch)


    def training_step(self, batch, batch_idx):
        """
        - input_ids: [B, L]
        - attention_mask: [B, L]
        - labels: [B, L]   
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
    
            self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return {"val_loss": loss}
        

    def configure_optimizers(self):
        llora_params = [p for _, p in self.model.named_parameters()
                        if p.requires_grad]
        
        optimizer = optim.AdamW(
            llora_params,
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
    
