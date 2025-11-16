import hydra
import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
torch.set_float32_matmul_precision("high") 
torch.backends.cudnn.allow_tf32 = True   

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from distcfm.losses import get_consistency_loss_fn
from distcfm.data import get_data_module
from distcfm.utils.evaluation import get_conditioning_data
from distcfm.models.model_wrapper import SIModelWrapper
from distcfm.utils import TrainingModule, EMAWeightAveraging, SamplingCallback

@hydra.main(config_path="../conf/train/", config_name="config.yaml", version_base="1.3")
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
    unwrapped_model = torch.compile(unwrapped_model)
    SI = instantiate(cfg.SI)
    model = SIModelWrapper(unwrapped_model, SI, cfg.use_parametrization)
    
    # data
    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit") 
    datamodule.setup(stage="test")

    # get a sample of conditioning data
    test_dataloader = datamodule.test_dataloader()
    test_data = get_conditioning_data(test_dataloader, num_samples=cfg.sampling.n_conditioning_samples)

    scaler, inverse_scaler = datamodule.scaler, datamodule.inverse_scaler
    loss_fn = get_consistency_loss_fn(cfg, SI)
    train_module = TrainingModule(cfg, model, loss_fn, SI)

    if cfg.compile:
        train_module = torch.compile(train_module)
    
    # callbacks
    ema_callback = EMAWeightAveraging(cfg.trainer.ema.decay)
    sampling_callback = SamplingCallback(cfg, test_data, inverse_scaler, SI)
    
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
        callbacks=[ema_callback, sampling_callback, periodic_checkpoint, best_checkpoint],
        strategy=DDPStrategy(find_unused_parameters=False),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val
    )

    trainer.fit(train_module, datamodule=datamodule, 
                ckpt_path=cfg.resume_from_checkpoint)

if __name__ == "__main__":
    main()
