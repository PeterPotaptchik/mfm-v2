import hydra
import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from distcfm.losses import get_denoiser_loss_fn
from distcfm.data import get_data_module
from distcfm.models.model_wrapper import DenoiserWrapper
from distcfm.utils import TrainingModuleDenoiser, EMAWeightAveraging, DenoiserSamplingCallback

@hydra.main(config_path="../conf/train/", config_name="config_denoiser.yaml", version_base="1.3")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        config=dict(cfg),
    )

    # get the log directory
    OmegaConf.set_struct(cfg, False)
    log_dir = os.path.join(hydra.utils.get_original_cwd(), 
                           hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    cfg.work_dir = log_dir

    unwrapped_model = instantiate(cfg.model)
    sde = instantiate(cfg.sde)
    model = DenoiserWrapper(unwrapped_model, sde, cfg.use_snr_parametrisation)

    # data
    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit") 
    datamodule.setup(stage="test")

    # scalers
    scaler, inverse_scaler = datamodule.scaler, datamodule.inverse_scaler
    loss_fn = get_denoiser_loss_fn(cfg, sde)
    train_module = TrainingModuleDenoiser(cfg, model, loss_fn, sde, inverse_scaler,)

    ema_callback = EMAWeightAveraging(cfg.trainer.ema.decay)
    sampling_callback = DenoiserSamplingCallback(cfg, sde, inverse_scaler)

    # checkpointing
    periodic_checkpoint = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoints",
        filename="periodic-{epoch:02d}-{step}",
        every_n_train_steps=cfg.trainer.get("checkpoint_every_n_steps", 10000),
        save_top_k=-1,  
    )

    best_checkpoint = ModelCheckpoint(
        monitor="val_ema/total_loss",
        dirpath=f"{log_dir}/checkpoints",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_steps=cfg.trainer.num_train_steps,
        accelerator="gpu",
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[periodic_checkpoint, best_checkpoint, ema_callback, sampling_callback],
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision
    )

    trainer.fit(train_module, datamodule=datamodule, 
                ckpt_path=cfg.resume_from_checkpoint)

if __name__ == "__main__":
    main()
