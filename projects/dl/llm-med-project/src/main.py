from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import wandb

from datamodule import DataModule_llm
from lightning_module import llm_pl


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    seed_everything(cfg.get("seed", 42), workers=True)

    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_trainer = cfg.trainer
    cfg_lora = cfg_model.lora

    out_dir = Path.cwd()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",              
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,             
    )
    

    if not cfg.get("config", None):
        run_name = f"run-{datetime.now().strftime('%d%m')}"
        wandb_config = cfg.get("wandb", {})
        wandb_logger = None
        if wandb_config.get("use", False):
            wandb_logger = WandbLogger(
                project=wandb_config.get("project", "default"),
                # id="lyj8w9vz",
                # resume="auto",
                name=run_name,
                log_model=wandb_config.get("log_model", False),
                mode=wandb_config.get("mode", "online"),
            )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=cfg_trainer.patience,
        mode= "min",
        strict=True,
    )

    datamodule = DataModule_llm(cfg_data)
    model = llm_pl(cfg_model, cfg_lora)
    trainer = pl.Trainer(
            max_epochs=cfg_trainer.max_epochs,
            accelerator=cfg_trainer.accelerator,
            devices=cfg_trainer.devices,
            precision=cfg_trainer.precision,
            gradient_clip_val=cfg_trainer.gradient_clip_val,
            accumulate_grad_batches=cfg_trainer.accumulate_grad_batches,
            logger=wandb_logger,
            log_every_n_steps=cfg_trainer.log_every_n_steps,
            val_check_interval=cfg_trainer.val_check_interval,
            callbacks=[lr_monitor,ckpt_best, early_stopping],
        )
    
    trainer.fit(model, datamodule = datamodule)

if __name__ == "__main__":
    main()
    wandb.finish()