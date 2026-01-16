import pytorch_lightning as pl
import torch
import torch.optim as optim
import numpy as np
from transformers import (
    BlipForConditionalGeneration,
    get_linear_schedule_with_warmup,
    AutoProcessor,
)
import evaluate


class multimodal_pl(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal image-to-text captioning
    using BLIP (image → caption).

    Responsibilities:
      - Load BLIP model and processor
      - Handle training and validation steps
      - Generate captions during validation
      - Compute ROUGE-L score at the end of each validation epoch
    """

    def __init__(self, cfg_model):
        super().__init__()
        self.save_hyperparameters()

        # Optimization hyperparameters
        self.model_name = cfg_model.model_name
        self.lr = float(cfg_model.lr)
        self.weight_decay = float(cfg_model.weight_decay)
        self.adam_beta1 = float(cfg_model.adam_beta1)
        self.adam_beta2 = float(cfg_model.adam_beta2)
        self.warmup_ratio = float(cfg_model.warmup_ratio)

        # Processor handles image preprocessing + text tokenization
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # BLIP model for conditional generation (image → text)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)

        # Ensure a valid pad token is defined
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Set padding token ID in model config (must be an integer)
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        print("Model pad_token_id:", self.model.config.pad_token_id)

        # ROUGE metric for caption evaluation
        self.rouge = evaluate.load("rouge")

        # Buffers to accumulate predictions and references during validation
        self.val_preds = []
        self.val_refs = []

        np.random.seed(42)

    def forward(self, **batch):
        """
        Forward pass wrapper.
        Expects:
          - pixel_values
          - input_ids
          - attention_mask
          - labels
        """
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        """
        Training step:
          - Compute cross-entropy loss for caption generation
          - Log training loss
        """
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation step:
          - Compute validation loss
          - Periodically generate captions for ROUGE evaluation
        """
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Generate captions on a subset of batches to reduce overhead
        if batch_idx % 8 == 0:
            gen_ids = self.model.generate(
                pixel_values=batch["pixel_values"],
                max_new_tokens=64,
                num_beams=3,
            )

            # Decode generated captions
            preds = self.processor.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True
            )

            # Decode ground-truth captions
            refs = self.processor.tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )

            self.val_preds.extend([p.strip() for p in preds])
            self.val_refs.extend([r.strip() for r in refs])

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """
        Compute ROUGE-L score at the end of the validation epoch.
        """
        if len(self.val_preds) == 0:
            return

        scores = self.rouge.compute(
            predictions=self.val_preds,
            references=self.val_refs,
            rouge_types=["rougeL"],
        )

        rouge_l = scores["rougeL"]
        self.log(
            "val/rougeL",
            rouge_l,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Clear buffers for next epoch
        self.val_preds.clear()
        self.val_refs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        Uses AdamW with linear warmup.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
