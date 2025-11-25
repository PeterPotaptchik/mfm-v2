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
from torchvision.datasets.utils import download_url
import torch.nn as nn


def download_sit_checkpoint(model_name='last.pt'):
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        print(f"Downloading SiT checkpoint {model_name}...")
        download_url(web_path, 'pretrained_models', filename=model_name)
    return local_path


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
    if cfg.compile:
        model = torch.compile(model)
    SI = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, cfg.use_parametrization)

    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    
    # For standard datasets
    test_dataloader = datamodule.test_dataloader()
    # If we can't shuffle the existing loader easily, we can just iterate and pick random
    # But standard test loaders are usually not shuffled.
    # Let's create a temporary shuffled loader
    if hasattr(datamodule, 'imagenet_val'):
            val_dataset = datamodule.imagenet_val
            target_classes = {978, 979, 980, 292}
            print(f"Filtering ImageNet validation set for classes {min(target_classes)}-{max(target_classes)}...")
            
            # Efficiently find indices using list comprehension on targets
            indices = [i for i, t in enumerate(val_dataset.targets) if t in target_classes]
            
            # Handle insufficient samples
            if len(indices) < cfg.sampling.n_conditioning_samples:
                print(f"Warning: Only found {len(indices)} samples. Repeating to fill batch.")
                indices = indices * (cfg.sampling.n_conditioning_samples // len(indices) + 1)
            
            # Select required samples
            perm = torch.randperm(len(indices))
            indices = [indices[i] for i in perm[:cfg.sampling.n_conditioning_samples].tolist()]
            test_data = torch.stack([val_dataset[i][0] for i in indices])
    else:
            # Fallback
            test_data = get_conditioning_data(
            test_dataloader,
            num_samples=cfg.sampling.n_conditioning_samples,
        )

    inverse_scaler = datamodule.inverse_scaler

    loss_fn = get_consistency_loss_fn(cfg, SI)
    train_module = TrainingModule(cfg, model, loss_fn, SI)

    if cfg.get("init_from_sit"):
        # sit_ckpt_path = download_sit_checkpoint()
        # print(f"Loading SiT checkpoint from {sit_ckpt_path}")
        # sit_ckpt = torch.load(sit_ckpt_path, map_location="cpu")
        print("HERE")
        sit_ckpt = torch.load("/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/ckpt/dmf_xl_2_256.pt", map_location="cpu", weights_only=False)
        
        if "ema" in sit_ckpt:
            print("Loading from EMA weights in SiT checkpoint")
            sit_state_dict = sit_ckpt["ema"]
        elif "model" in sit_ckpt:
            sit_state_dict = sit_ckpt["model"]
        else:
            sit_state_dict = sit_ckpt

        # Remove 'module.' prefix if present
        sit_state_dict = {k.replace("module.", ""): v for k, v in sit_state_dict.items()}

        # Target model (DiT)
        target_model = train_module.model.model
        if hasattr(target_model, "dit"):
            target_model = target_model.dit
        
        # # Handle proj_head renaming
        # keys_to_rename = []
        # for k in sit_state_dict.keys():
        #     if k.startswith("p`rojectors.0."):
        #         keys_to_rename.append(k)
        
        # for k in keys_to_rename:
        #     new_k = k.replace("projectors.0.", "proj_head.")
        #     sit_state_dict[new_k] = sit_state_dict.pop(k)

        # Load state dict
        print("Loading state dict into DiT...")
        missing, unexpected = target_model.load_state_dict(sit_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        
        # Initialize missing keys to zero
        if missing:
             print(f"Initializing {len(missing)} missing keys to zero...")
             for k in missing:
                 if k in target_model.state_dict():
                     target_model.state_dict()[k].data.zero_()

        # Initialize x_cond_embedder from x_embedder (checkpoint)
        print("Initializing x_cond_embedder from x_embedder...")
        target_model.x_cond_embedder.load_state_dict(target_model.x_embedder.state_dict())

        # Initialize s_embedder from t_embedder (checkpoint)
        print("Initializing s_embedder from t_embedder...")
        target_model.s_embedder.load_state_dict(target_model.t_embedder.state_dict())
        target_model.t_cond_embedder.load_state_dict(target_model.t_embedder.state_dict())
        target_model.s_embedder_second.load_state_dict(target_model.t_embedder.state_dict())
        target_model.t_embedder_second.load_state_dict(target_model.t_embedder.state_dict())

        # Zero out t_cond_embedder
        print("Zeroing out t_cond_embedder...")
        nn.init.constant_(target_model.t_cond_embedder.mlp[2].weight, 0)
        nn.init.constant_(target_model.t_cond_embedder.mlp[2].bias, 0)
        nn.init.constant_(target_model.t_embedder.mlp[2].weight, 0)
        nn.init.constant_(target_model.t_embedder.mlp[2].bias, 0)
        nn.init.constant_(target_model.s_embedder_second.mlp[2].weight, 0)
        nn.init.constant_(target_model.s_embedder_second.mlp[2].bias, 0)

        # Initialize x_cond_adaLN to gate off x_cond
        # We set weights to 0 and bias to [0, -1].
        # This gives shift=0, scale=-1.
        # modulate(x, shift, scale) = x * (1 + scale) + shift = x * 0 + 0 = 0.
        print("Initializing x_cond_adaLN to gate off x_cond (scale=-1)...")
        nn.init.constant_(target_model.x_cond_adaLN[-1].weight, 0)
        nn.init.constant_(target_model.x_cond_adaLN[-1].bias, 0)
        half_dim = target_model.x_cond_adaLN[-1].bias.shape[0] // 2
        nn.init.constant_(target_model.x_cond_adaLN[-1].bias[half_dim:], -1)
        
        print("SiT checkpoint loaded successfully.")

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
