import torch
import os
from torchvision.io import write_png
import numpy as np
import random
import matplotlib.pyplot as plt

from distcfm.SI.samplers import consistency_sampler_fn, ode_sampler_fn

def get_conditioning_data(test_dataloader, num_samples=256,):
    """Get conditioning data from the test dataloader."""
    data = None
    for _, batch in enumerate(test_dataloader):
        try:
            x, _ = batch
        except:
            x = batch
        if data is None:
            data = x
        else:
            data = torch.cat((data, x), dim=0)
        if data.shape[0] >= num_samples:
            break
    
    return data[:num_samples]

def plot_posterior_samples(x_0, x_t, x_0_samples,
                           save_path, title,):
    def _to_numpy_channels_last(array):
        """Convert tensor/array in CHW to numpy HWC for plotting."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        if array.ndim == 3 and (array.shape[0] <= 4 and array.shape[0] != array.shape[-1]):
            array = np.transpose(array, (1, 2, 0))
        return array

    x_0 = x_0.detach().cpu() if isinstance(x_0, torch.Tensor) else x_0
    x_t = x_t.detach().cpu() if isinstance(x_t, torch.Tensor) else x_t
    x_0_samples = x_0_samples.detach().cpu() if isinstance(x_0_samples, torch.Tensor) else x_0_samples

    x_0 = np.asarray(x_0)
    x_t = np.asarray(x_t)
    x_0_samples = np.asarray(x_0_samples)

    N, M, C = x_0_samples.shape[:3]
    cmap = 'gray' if C == 1 else None

    f, axs = plt.subplots(N, M + 2, figsize=((M + 2) * 3, N * 3))
    if N == 1:
        axs = axs.reshape(1, -1)

    for i in range(N):
        axs[i, 0].imshow(_to_numpy_channels_last(x_0[i]), cmap=cmap)
        axs[i, 0].set_title("x_0")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(_to_numpy_channels_last(x_t[i]), cmap=cmap)
        axs[i, 1].set_title("x_t")
        axs[i, 1].axis('off')

        for j in range(M):
            axs[i, j + 2].imshow(_to_numpy_channels_last(x_0_samples[i, j]), cmap=cmap)
            axs[i, j + 2].set_title(f"x_0 sample {j + 1}")
            axs[i, j + 2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    return f


def plot_inverse_samples(x, x_recon, y, save_path, title,):
    columns = 3  # Always 3 columns: one for x, one for y, one for x_recon
    rows = x.shape[0]
    f, axs = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))

    if rows == 1:
        axs = axs.reshape(1, -1)
    
    if x.shape[1] == 1:
        cmap = 'gray'
    else:
        cmap = None

    for i in range(rows):
        x_i = x[i].transpose(1, 2, 0)
        y_i = y[i].transpose(1, 2, 0)
        x_recon_i = x_recon[i].transpose(1, 2, 0)

        axs[i, 0].imshow(x_i, cmap=cmap)
        axs[i, 0].set_title(f"X {i + 1}")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(y_i, cmap=cmap)
        axs[i, 1].set_title(f"Y {i + 1}")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(x_recon_i, cmap=cmap)
        axs[i, 2].set_title(f"X Reconstructed {i + 1}")
        axs[i, 2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(f)

def load_model(model, checkpoint):
    """Load the model checkpoint."""
    state_dict = checkpoint['state_dict']
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.model._orig_mod."):
            new_key = key[22:]
            model_state_dict[new_key] = value
        elif key.startswith("model.model."):
            new_key = key[12:]
            model_state_dict[new_key] = value 
        elif key.startswith("model."):
            new_key = key[6:] 
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    res = model.load_state_dict(model_state_dict, 
                                strict=False)
    if len(res.missing_keys) > 0:
        print("Missing keys when loading model:", res.missing_keys)
        if len(res.missing_keys) > 3:
            raise ValueError("Too many missing keys!")
    if len(res.unexpected_keys) > 0:
        print("Unexpected keys when loading model:", res.unexpected_keys)

    return model

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

def posterior_sampling_fn(cfg, 
                          model, 
                          xt_cond, # [B, C, H, W],
                          t_cond, # [B, 1]
                          n_samples_per_image=4,
                          inverse_scaler =lambda x: (x+1)/2,
                          eps_start=None):
    """Sample from the posterior distribution using the model."""
    # Repeat to get multiple posterior samples per noisy (xt_cond, t_cond)
    xt_cond_batched = xt_cond.repeat_interleave(n_samples_per_image, dim=0)  # [N*B, C, H, W]
    t_cond_batched = t_cond.repeat_interleave(n_samples_per_image, dim=0)  # [N*B,]

    if cfg.posterior_sampler == "consistency":
        x_sample = consistency_sampler_fn(model,
                                          xt_cond_batched, 
                                          t_cond=t_cond_batched, 
                                          n_steps=cfg.consistency.steps, 
                                          eps_start=eps_start)
    elif cfg.posterior_sampler == "ode":
        x_sample = ode_sampler_fn(model,
                                  xt_cond_batched,
                                  t_cond=t_cond_batched,
                                  n_steps=cfg.ode.steps,
                                  eps_start=eps_start)
        
    elif cfg.posterior_sampler == "distributional_diffusion":
        noise_population = torch.randn_like(xt_cond_batched, device=xt_cond_batched.device)
        x_sample = model(xt_cond_batched, t_cond_batched, noise_population)
    else:
        raise ValueError(f"Unknown posterior sampler: {cfg.posterior_sampler}")

    x_sample = x_sample.view(-1, 
                             n_samples_per_image, 
                             *x_sample.shape[1:])  # [B, N, C, H, W]
    
    # [B, C, H, W], # [B, N, C, H, W]
    return inverse_scaler(xt_cond), inverse_scaler(x_sample)


def save_for_fid(tensor_bchw: torch.Tensor, out_dir: str, prefix="sample"):
    """
    tensor_bchw: (B, C, H, W), values in [0,1] or [-1,1]
    Saves PNGs as prefix_000000.png, ...
    """
    os.makedirs(out_dir, exist_ok=True)

    # Clamp and convert to uint8 CHW
    imgs = (tensor_bchw.clamp(0, 1) * 255).to(torch.uint8).cpu()

    # Zero-pad for stable sorting
    pad = len(str(len(imgs)-1))
    for i, img in enumerate(imgs):
        # img is (C,H,W); write_png supports 1 or 3 channels
        write_png(img, os.path.join(out_dir, f"{prefix}_{i:0{pad}d}.png"))

def get_l2_distance(x_recon, x):
    x_recon_flat = x_recon.view(x_recon.shape[0], -1)
    x_flat = x.view(x.shape[0], -1)
    return torch.norm(x_recon_flat - x_flat, p=2, dim=1)


def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
