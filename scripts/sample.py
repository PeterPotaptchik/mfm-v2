import hydra
import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from cleanfid import fid
from distcfm.data import get_data_module
from distcfm.utils.evaluation import load_model, set_seed
from distcfm.utils.evaluation import posterior_sampling_fn, get_conditioning_data, plot_posterior_samples, load_model, save_for_fid
from distcfm.models.model_wrapper import SDEModelWrapper
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

@hydra.main(config_path="../conf/sample/", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # load training config
    print("Loading training config from:", cfg.train_config)
    train_cfg = OmegaConf.load(cfg.train_config)
    OmegaConf.set_struct(train_cfg, False)
    log_dir = str(Path(cfg.checkpoint_dir).parent.parent)
    set_seed(cfg.seed)
    
    # load parameters of the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = instantiate(train_cfg.model)
    model.to(device)
    checkpoint = torch.load(cfg.checkpoint_dir, 
                            map_location=device)
    
    unwrapped_model = load_model(model, checkpoint)
    sde = instantiate(train_cfg.sde)

    try:
        use_noise_parametrization = train_cfg.model.use_noise_parametrization
    except AttributeError:
        use_noise_parametrization = False

    # consistency model is wrapped!
    if cfg.posterior_sampler != "distributional_diffusion":
        model = SDEModelWrapper(unwrapped_model, sde, use_noise_parametrization)

    if cfg.use_ema == True:
        print("Loading EMA parameters...")
        ema = ExponentialMovingAverage(model.parameters(), 
                                       decay=train_cfg.trainer.ema.decay)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.to(device)
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    else: # how 
        print("Not using EMA parameters...")
    
    # data module for conditioning on noisy samples
    datamodule = get_data_module(train_cfg)
    datamodule.setup(stage="test")

    test_dataloader = datamodule.test_dataloader()

    x = get_conditioning_data(test_dataloader, 
                              num_samples=cfg.n_conditioning_samples)
    x = x.to(device)

    posterior_dir = os.path.join(log_dir, "posterior_samples")
    os.makedirs(posterior_dir, exist_ok=True)

    for t_cond in cfg.conditioning_times:
        x_t_res, x_0_res = [], []

        if cfg.posterior_sampler == "consistency":
            sample_label = f"t_{t_cond}_const_{cfg.consistency.steps}_{cfg.use_ema}"
        elif cfg.posterior_sampler == "ode":
            sample_label = f"t_{t_cond}_ode_{cfg.ode.steps}_{cfg.use_ema}"
        elif cfg.posterior_sampler == "distributional_diffusion":
            sample_label = f"t_{t_cond}_dist_{cfg.use_ema}"

        for i in range(cfg.n_conditioning_samples//cfg.batch_size):
            x_batch = x[i*cfg.batch_size:(i+1)*cfg.batch_size]
            t_cond_tensor = torch.full((cfg.batch_size,), 
                                       t_cond, device=device)
            with torch.no_grad():
                x_t, x_0_samples = posterior_sampling_fn(cfg, 
                                                        model, 
                                                        sde, 
                                                        x_batch, 
                                                        t_cond_tensor,
                                                        inverse_scaler=datamodule.inverse_scaler)
            x_t_res.append(x_t)
            x_0_res.append(x_0_samples)

        x_t_res = torch.cat(x_t_res, dim=0)
        x_0_res = torch.cat(x_0_res, dim=0)

        # plot posterior samples
        title = f"posterior samples at t = {t_cond}"
        save_path = os.path.join(posterior_dir, f"{sample_label}.png")
        plot_posterior_samples(datamodule.inverse_scaler(x[0:10].cpu().numpy()),
                               x_t_res[0:10].clamp(0,1).cpu().numpy(),
                               x_0_res[0:10, 0:10].clamp(0,1).cpu().numpy(),
                               save_path, 
                               title) 
        if cfg.compute_fid:
            # (N, M, C, H, W) -> (N*M, C, H*W)
            x_0_res = x_0_res.view(-1, *x_0_res.shape[2:])
            assert train_cfg.dataset.name == "cifar10", \
                "FID computation is only implemented for CIFAR-10 dataset."
            assert x_0_res.shape[0] > 1000, \
                "Not enough samples for FID computation."
            print("shape", x_0_res.shape, "min", x_0_res.min(), "max", x_0_res.max())
            fid_save_dir = os.path.join(log_dir, "fid_samples", sample_label) 
            save_for_fid(x_0_res.cpu(), fid_save_dir)
            fid_score = fid.compute_fid(fid_save_dir, 
                                        dataset_name="cifar10",
                                        mode="clean",
                                        dataset_split="train",
                                        dataset_res=32)
            with open(os.path.join(log_dir, "fid_scores.txt"), "a") as f:
                f.write(f"{sample_label}: {fid_score}\n")

        
if __name__ == "__main__":
    import sys
    print("sys.argv:", sys.argv)
    main()
