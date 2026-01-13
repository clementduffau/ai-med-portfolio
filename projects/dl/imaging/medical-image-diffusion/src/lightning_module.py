import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from diffusers import DDPMScheduler, UNet2DModel


class Modelelightning_diff(pl.LightningModule):
    """
    Conditional DDPM LightningModule.

    This module trains a diffusion model to predict the added Gaussian noise (epsilon)
    given:
      - a noisy image x_t
      - a diffusion timestep t
      - a class label (class_id) for conditional generation

    Training objective (standard DDPM epsilon-prediction):
      loss = MSE(eps_pred, eps_true)

    Notes:
      - Input images must be normalized to [-1, 1]
      - class_id must be a torch.long tensor of shape (B,)
    """

    def __init__(self, cfg_model):
        
        super().__init__()
        self.save_hyperparameters()

        # Optimizer hyperparams
        self.lr = float(cfg_model.lr)
        self.adam_beta1 = float(cfg_model.adam_beta1)
        self.adam_beta2 = float(cfg_model.adam_beta2)
        self.weight_decay = float(cfg_model.weight_decay)
        self.scheduler_type = getattr(cfg_model, "scheduler_type", "cosine")

        # Image/model shape
        self.image_size = int(cfg_model.img_size)
        self.in_channels = int(cfg_model.in_channels)
        self.num_classes = int(cfg_model.num_classes)

        # Diffusion noise scheduler (forward diffusion process)
        # It defines how much noise is added at each timestep.
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(cfg_model.num_train_timesteps),
            beta_schedule=cfg_model.beta_schedule,  
        )

        # Diffusion U-Net with class conditioning:
        # num_class_embeds enables an internal class embedding table of size num_classes.
        self.model = UNet2DModel(
            sample_size=self.image_size,
            in_channels=self.in_channels,
            out_channels=self.in_channels,  # predict epsilon (same channels as input)
            layers_per_block=int(cfg_model.layers_per_block),
            block_out_channels=tuple(cfg_model.block_out_channels),
            down_block_types=tuple(cfg_model.down_block_types),  
            up_block_types=tuple(cfg_model.up_block_types),
            num_class_embeds=self.num_classes,
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the conditional diffusion model.

        Args:
            noisy_images: x_t, shape (B, C, H, W)
            timesteps: t, shape (B,) long
            class_labels: class_id, shape (B,) long

        Returns:
            eps_pred: predicted noise, shape (B, C, H, W)
        """
        return self.model(noisy_images, timesteps, class_labels=class_labels).sample

    def training_step(self, batch, batch_idx):
        """
        One training step (DDPM epsilon-prediction):

        1) Take a clean image x0
        2) Sample a random timestep t
        3) Sample Gaussian noise eps
        4) Create a noisy image x_t = add_noise(x0, eps, t)
        5) Predict eps_pred = UNet(x_t, t, class_id)
        6) Compute loss = MSE(eps_pred, eps)
        """
        x0 = batch["image"]              # clean image in [-1, 1], shape (B,C,H,W)
        class_id = batch["class_id"]     # class labels, shape (B,) ideally torch.long
        bsz = x0.shape[0]

        # Sample ground-truth noise epsilon
        noise = torch.randn_like(x0)

        # Sample a diffusion timestep for each element in the batch
        t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=x0.device,
            dtype=torch.long,
        )

        # Forward diffusion: create noisy image x_t
        xt = self.noise_scheduler.add_noise(x0, noise, t)

        # Predict epsilon given (x_t, t, class_id)
        eps_pred = self.forward(xt, t, class_id)

        # MSE between predicted noise and true noise
        loss = F.mse_loss(eps_pred, noise)  

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        """
        Validation step: same computation as training (but no gradients).
        """
        x0 = batch["image"]
        class_id = batch["class_id"]
        bsz = x0.shape[0]

        noise = torch.randn_like(x0)
        t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=x0.device,
            dtype=torch.long,
        )

        xt = self.noise_scheduler.add_noise(x0, noise, t)
        eps_pred = self.forward(xt, t, class_id)

        loss = F.mse_loss(eps_pred, noise)  

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure optimizer + LR scheduler.

        Here we use AdamW + CosineAnnealingLR:
          - T_max is set to max_epochs so LR decays smoothly across all epochs
          - eta_min is the minimum LR at the end of the cosine schedule
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.weight_decay, 
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
