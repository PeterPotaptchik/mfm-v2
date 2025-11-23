import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from distcfm.data import get_data_module
from distcfm.losses import get_consistency_loss_fn
from distcfm.models.model_wrapper import SIModelWrapper
from distcfm.utils import EMAWeightAveraging, SamplingCallback, TrainingModule
from distcfm.utils.evaluation import get_conditioning_data
from distcfm.utils.repa_utils import get_repa_z_dims


@hydra.main(config_path="../conf/train/", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        config=dict(cfg),
    )

    OmegaConf.set_struct(cfg, False)
    log_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    cfg.work_dir = log_dir

    z_dim = get_repa_z_dims()
    cfg.model.z_dim = z_dim
    model = instantiate(cfg.model)
    model = torch.compile(model)
    SI = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, cfg.use_parametrization)

    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    test_data = get_conditioning_data(
        datamodule.test_dataloader(),
        num_samples=cfg.sampling.n_conditioning_samples,
    )
    inverse_scaler = datamodule.inverse_scaler

    loss_fn = get_consistency_loss_fn(cfg, SI)
    train_module = TrainingModule(cfg, model, loss_fn, SI)

    if cfg.compile:
        train_module = torch.compile(train_module)

    ema_callback = EMAWeightAveraging(cfg.trainer.ema.decay)
    sampling_callback = SamplingCallback(cfg, test_data, inverse_scaler, SI)
    periodic_checkpoint = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoints",
        filename="periodic-{epoch:02d}-{step}",
        every_n_train_steps=cfg.trainer.get("checkpoint_every_n_steps", 10000),
        save_top_k=-1,
    )
    print(f"Checkpoint every {cfg.trainer.get("checkpoint_every_n_steps", 10000)} steps to {log_dir}")
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
        num_nodes=cfg.trainer.get("num_nodes", 1),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[ema_callback, sampling_callback, periodic_checkpoint, best_checkpoint],
        strategy=DDPStrategy(find_unused_parameters=False),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    resume_path = cfg.get("resume_from_checkpoint")
    ckpt_path = resume_path
    if resume_path:
        print(f"Attempting to load checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")

        print("Checking for missing/unexpected keys...")
        missing_keys, unexpected_keys = train_module.load_state_dict(
            ckpt["state_dict"], strict=False
        )

        if missing_keys:
            print("\nMissing keys (reinitialized):")
            for key in missing_keys[:10]:
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
        else:
            print("No missing keys detected")

        if unexpected_keys:
            print("\nUnexpected keys (ignored):")
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")

        removed_state = False
        if ckpt.pop("optimizer_states", None) is not None:
            removed_state = True
            print("Removed optimizer_states to start with a fresh optimizer")
        if ckpt.pop("lr_schedulers", None) is not None:
            removed_state = True
            print("Removed lr_schedulers to reset scheduler state")

        if removed_state:
            ckpt_path = None
            print("Checkpoint missing optimizer state; trainer will start with fresh optimizer state\n")
        else:
            print("Checkpoint already weights-only; will rely on Lightning to restore remaining state\n")

    trainer.fit(train_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
