import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torchvision
import os
from collections import OrderedDict
from distcfm.data import get_data_module
from distcfm.models.model_wrapper import DenoiserWrapper
from distcfm.utils.evaluation import load_model
from torch_ema import ExponentialMovingAverage
from distcfm.utils.evaluation import save_for_fid
from cleanfid import fid
import tqdm


@hydra.main(config_path="../conf/sample/", config_name="config_denoiser.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """
    Generates images from a trained denoiser model using a specified checkpoint,
    loading Exponential Moving Average (EMA) weights directly for inference.
    """
    # --- 1. Setup and Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the training configuration
    print(f"Loading training config from: {cfg.train_config_path}")
    train_cfg = OmegaConf.load(cfg.train_config_path)

    # --- 2. Instantiate Model, SDE, and Data ---
    sde = instantiate(train_cfg.sde)
    # unwrapped_model = instantiate(train_cfg.model)
    # # The DenoiserWrapper is still needed as it was part of the trained model's architecture
    # model = DenoiserWrapper(unwrapped_model, sde, train_cfg.use_snr_parametrisation)
    # model.to(device)

    datamodule = get_data_module(train_cfg)
    inverse_scaler = datamodule.inverse_scaler

    # --- 3. Load Checkpoint and EMA Weights Directly ---
    print(f"Loading checkpoint from: {cfg.checkpoint_path}")
    

    # if False: #cfg.use_ema and 'ema_state_dict' in checkpoint:
    #     print("Extracting and loading EMA weights directly...")
    #     shadow_params = checkpoint['ema_state_dict']['shadow_params']
        
    #     # Get the correctly ordered keys from the non-EMA state_dict
    #     original_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith("model.model.")]
        
    #     # Clean the keys by removing the extra 'model.' prefix
    #     cleaned_keys = [key.replace("model.model.", "model.", 1) for key in original_keys]

    #     # Create the new state dict
    #     new_state_dict = OrderedDict(zip(cleaned_keys, shadow_params))
        
    # else:
    #     print("Using standard model weights.")
    #     original_state_dict = checkpoint['state_dict']
    #     new_state_dict = OrderedDict()
    #     for key, value in original_state_dict.items():
    #         # Clean the keys by removing the extra 'model.' prefix
    #         if key.startswith("model.model."):
    #             new_key = key.replace("model.model.", "model.", 1)
    #             new_state_dict[new_key] = value

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    uncond_model = instantiate(train_cfg.model)
    uncond_model.to(device)
    # uncond_checkpoint = torch.load(cfg.uncond_checkpoint_dir, 
                            # map_location=device)
    uncond_model = load_model(uncond_model, checkpoint)
    uncond_model = DenoiserWrapper(uncond_model, sde, 
                                   train_cfg.use_snr_parametrisation)
    
    if cfg.use_ema == True:
        # load uncond
        ema = ExponentialMovingAverage(uncond_model.parameters(), 
                                       decay=train_cfg.trainer.ema.decay)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.to(device)
        ema.store(uncond_model.parameters())
        ema.copy_to(uncond_model.parameters())
    
    model = uncond_model
    

    model.eval()

    # --- 4. DDIM Sampling ---
    print("Starting DDIM sampling...")

    samples_lst = []

    for i in tqdm.tqdm(range(0, cfg.n_samples, cfg.batch_size)):
        num_samples = min(cfg.batch_size, cfg.n_samples - i)
        with torch.no_grad():
            x = torch.randn(
                num_samples, 
                train_cfg.dataset.img_channels, 
                train_cfg.dataset.img_resolution, 
                train_cfg.dataset.img_resolution, 
                device=device
            )
            time_steps = torch.linspace(sde.t_max - 1e-2, 1e-4, cfg.n_steps + 1, device=device) # Start slightly from t_max

            for i in range(cfg.n_steps):
                t_now, t_next = time_steps[i], time_steps[i+1]
                alpha_t, var_t = sde.get_coefficients(t_now)
                sigma_t = torch.sqrt(var_t)
                
                predicted_noise = model(x, t_now.expand(cfg.batch_size))
                x_0_pred = (x - sigma_t * predicted_noise) / alpha_t

                alpha_t_next, var_t_next = sde.get_coefficients(t_next)
                sigma_t_next = torch.sqrt(var_t_next)

                x = alpha_t_next * x_0_pred + sigma_t_next * predicted_noise

            generated_images = inverse_scaler(x.cpu())
            samples_lst.append(generated_images)
    
    x_0_res = torch.cat(samples_lst, dim=0)
    x_0_res = torch.clamp(x_0_res, min=0.0, max=1.0)

    # --- 5. Save the Output ---
    output_dir = os.getcwd() 
    output_path = os.path.join(output_dir, cfg.output_filename)
    
    num_plot = 100
    torchvision.utils.save_image(x_0_res[:num_plot], output_path, nrow=int(num_plot**0.5), normalize=True)
    print(f"Generated images saved to: {output_path}")

    if cfg.compute_fid:
        # (N, M, C, H, W) -> (N*M, C, H*W)
        print("shape", x_0_res.shape, "min", x_0_res.min(), "max", x_0_res.max())
        fid_save_dir = os.path.join(output_dir, "fid_samples") 
        save_for_fid(x_0_res.cpu(), fid_save_dir)
        fid_score = fid.compute_fid(fid_save_dir, 
                                    dataset_name="cifar10",
                                    mode="clean",
                                    dataset_split="train",
                                    dataset_res=32)
        print(fid_score)
        with open(os.path.join(output_dir, "fid_scores.txt"), "a") as f:
            f.write(f"DDIM n_steps={cfg.n_steps}: {fid_score}\n")


if __name__ == "__main__":
    main()
