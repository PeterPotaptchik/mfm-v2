import tqdm
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F

from distcfm.data import get_data_module
from distcfm.utils.evaluation import get_conditioning_data, load_model, plot_inverse_samples, get_l2_distance, set_seed
from distcfm.sde.score_fn import get_unconditional_score_fn, get_dps_score_fn, get_iwae_score_fn, get_mc_score_fn
from distcfm.sde.samplers import EulerMaruyamaSampler
from distcfm.models.model_wrapper import SDEModelWrapper, DenoiserWrapper

@hydra.main(config_path="../conf/sample/", config_name="config_inverse.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # set seed
    set_seed(cfg.seed)
    
    uncond_cfg = OmegaConf.load(cfg.uncond_config)
    dist_cfg = OmegaConf.load(cfg.dist_config)  

    # load parameters of the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sde = instantiate(uncond_cfg.sde)

    # load unconditional model
    uncond_model = instantiate(uncond_cfg.model)
    uncond_model.to(device)
    uncond_checkpoint = torch.load(cfg.uncond_checkpoint_dir, 
                            map_location=device)
    uncond_model = load_model(uncond_model, uncond_checkpoint)
    uncond_model = DenoiserWrapper(uncond_model, sde, 
                                   uncond_cfg.use_snr_parametrisation)
    
    # load consistency model
    dist_model = instantiate(dist_cfg.model)
    dist_model.to(device)
    dist_checkpoint = torch.load(cfg.dist_checkpoint_dir, 
                            map_location=device)
    unwrapped_dist_model = load_model(dist_model, dist_checkpoint)
    dist_model = SDEModelWrapper(unwrapped_dist_model, sde, False)

    if cfg.use_ema == True:
        # load uncond
        ema = ExponentialMovingAverage(uncond_model.parameters(), 
                                       decay=uncond_cfg.trainer.ema.decay)
        ema.load_state_dict(uncond_checkpoint['ema_state_dict'])
        ema.to(device)
        ema.store(uncond_model.parameters())
        ema.copy_to(uncond_model.parameters())
        # load dist
        dist_ema = ExponentialMovingAverage(dist_model.parameters(), 
                                       decay=dist_cfg.trainer.ema.decay)
        dist_ema.load_state_dict(dist_checkpoint['ema_state_dict'])
        dist_ema.to(device)
        dist_ema.store(dist_model.parameters())
        dist_ema.copy_to(dist_model.parameters())

    # data module for conditioning on noisy samples
    datamodule = get_data_module(uncond_cfg)
    datamodule.setup(stage="test")

    test_dataloader = datamodule.test_dataloader()

    x = get_conditioning_data(test_dataloader, 
                              num_samples=cfg.sampling.n_samples,)
    x = datamodule.inverse_scaler(x)
    x = x.to(device)

    inverse_problem = instantiate(cfg.inverse_problem)
    y = inverse_problem.forward(x)
    
    # get sampler
    sampler = EulerMaruyamaSampler(sde, num_steps=cfg.sampling.euler_steps,)
    # get the unconditional score_fn
    unconditional_score_fn = get_unconditional_score_fn(uncond_model, sde)
    
    # big if 
    
    # get the conditional score fn
    if cfg.conditional_score == "iwae":
        conditional_score_fn = get_iwae_score_fn(dist_model,
                                                 inverse_problem.log_likelihood,
                                                 datamodule.inverse_scaler,
                                                 mc_samples=cfg.iwae.n_samples,
                                                 type="cfm")
        save_dir = cfg.dist_dir
    elif cfg.conditional_score == "mc":
        conditional_score_fn = get_mc_score_fn(dist_model,
                                                 inverse_problem.log_likelihood,
                                                 datamodule.inverse_scaler,
                                                 sde,
                                                 mc_samples=cfg.mc.n_samples,
                                                 type="cfm")
        save_dir = cfg.dist_dir
    elif cfg.conditional_score == "dps":
        conditional_score_fn = get_dps_score_fn(uncond_model,
                                                sde,
                                                inverse_problem.log_likelihood,
                                                datamodule.inverse_scaler,)
        save_dir = cfg.uncond_dir
    else:
        raise ValueError(f"Unknown conditional score function: {cfg.conditional_score}")
    
    samples = []

    for i in tqdm.tqdm(range(0, y.shape[0], cfg.sampling.batch_size), desc="Sampling"):
        y_batch = y[i:i + cfg.sampling.batch_size]
        joint_score_fn = lambda x, t: unconditional_score_fn(x, t) + conditional_score_fn(x, t, y_batch)
        samples_batch = sampler.sample((y_batch.shape[0], *datamodule.dims),
                                        joint_score_fn,
                                        inverse_scalar_fn=datamodule.inverse_scaler,
                                        grads=True,
                                        clip=cfg.sampling.clip_each_step,)

        samples.append(samples_batch)
    
    samples = torch.cat(samples, dim=0)

    if cfg.inverse_problem.name == "sr":
        inverse_problem_label = f"sr_{cfg.inverse_problem.noise_sigma}_{cfg.inverse_problem.factor}"
    elif cfg.inverse_problem.name == "ip":
        inverse_problem_label = f"ip_{cfg.inverse_problem.noise_sigma}_{cfg.inverse_problem.type}"
    elif cfg.inverse_problem.name == "deblur":
        inverse_problem_label = f"deblur_{cfg.inverse_problem.noise_sigma}_{cfg.inverse_problem.kernel_sigma}_{cfg.inverse_problem.kernel_size}"
    elif cfg.inverse_problem.name == "colourization":
        inverse_problem_label = f"colourization_{cfg.inverse_problem.noise_sigma}_{cfg.inverse_problem.weights}"

    # make dir for inverse problem 
    inverse_problem_dir = os.path.join(save_dir, inverse_problem_label)
    os.makedirs(inverse_problem_dir, exist_ok=True)

    if cfg.conditional_score == "dps":
        save_path = f"dps_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}"
    elif cfg.conditional_score == "iwae":
        save_path = f"iwae_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}_{cfg.iwae.n_samples}"
    elif cfg.conditional_score == "mc":
        save_path = f"mc_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}_{cfg.mc.n_samples}"

    # make dir for samples 
    samples_dir = os.path.join(inverse_problem_dir, save_path)
    print("saving to:", samples_dir)

    os.makedirs(samples_dir, exist_ok=True)
    
    # plot
    plot_inverse_samples(x[:16].cpu().detach().numpy(),
                         samples[:16].cpu().detach().numpy(),
                         y[:16].cpu().detach().numpy(),
                         save_path=os.path.join(samples_dir, "example_samples.png"),
                         title="Inverse Samples")
    
    print("samples", samples.min(), samples.max(), "x", x.min(), x.max())
    
    # save
    torch.save(samples, os.path.join(samples_dir, "x_recon.pt"))
    torch.save(x, os.path.join(samples_dir, "x.pt"))
    torch.save(y, os.path.join(samples_dir, "y.pt"))

    # evaluate the samples
    l2_distance = get_l2_distance(samples, x)
    mean_l2_distance = l2_distance.mean().item()
    print("Mean L2 distance", mean_l2_distance)

    # ssim, lpips, psnr
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    
    ssim_score = ssim(samples, x).item()
    psnr_score = psnr(samples, x).item()
    lpips_score = lpips(samples, x).item()

    print(f"SSIM: {ssim_score:.4f}")
    print(f"PSNR: {psnr_score:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")
    
    with torch.no_grad():
        x_up   = F.interpolate(x, size=224, mode='bilinear', align_corners=False, antialias=True)
        y_up   = F.interpolate(samples, size=224, mode='bilinear', align_corners=False, antialias=True)
        lpips_upscaled = lpips(y_up, x_up)  
        lpips_score_upscaled  = lpips_upscaled.mean().item()
        print(f"LPIPS Upscaled: {lpips_score_upscaled:.4f}")

    # add to a text file
    with open(os.path.join(samples_dir, "metrics.txt"), "w") as f:
        f.write(f"SSIM: {ssim_score:.4f}\n")
        f.write(f"PSNR: {psnr_score:.4f}\n")
        f.write(f"LPIPS: {lpips_score:.4f}\n")
        f.write(f"LPIPS Upscaled: {lpips_score_upscaled:.4f}\n")

if __name__ == "__main__":
    main()