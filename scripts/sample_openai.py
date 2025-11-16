from argparse import ArgumentParser
import os
import time
import math

import torch
import torchvision

from distcfm.utils import get_diffusion
from distcfm.sde import VPSDE
from distcfm.sde.samplers import EulerMaruyamaSampler

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

def get_unconditional_score_fn_ddpm(model, sde, rescale_time_fn):
    def score_fn(x, t):
        _, var_t = sde.get_coefficients(t)
        var_t = broadcast_to_shape(var_t, x.shape)
        rescaled_time = rescale_time_fn(t)
        eps_pred = model(x, rescaled_time)
        score = - eps_pred / torch.sqrt(var_t)
        score = score.detach()
        return score
    return score_fn

def main(args):
    # assert args.noise_schedule == "cosine", "Only cosine noise schedule is supported"
    
    # load model
    unet, _, _, _ = get_diffusion(args)
    # load sde
    sde = VPSDE(t_max=1.0, noise_schedule=args.noise_schedule)
    
    # function to map our time in range [0, 1] (where 1 is noise), to openai model input [0, max_train_input]
    # see _scale_timesteps in https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L764
    rescale_time_fn = lambda t: (t * 1000).round().long()
    
    # wrap up in a score function 
    score_fn = get_unconditional_score_fn_ddpm(unet, sde, rescale_time_fn)

    euler_maruyama_sampler = EulerMaruyamaSampler(sde, num_steps=args.inference_steps, t_eps=0.001)
    
    total_samples = []
    for i in range(0, args.num_samples, args.batch_size):
        n_samples = min(args.batch_size, args.num_samples - i)
        samples = euler_maruyama_sampler.sample((n_samples, 3, args.image_size, args.image_size),
                                                score_fn=score_fn, inverse_scalar_fn=lambda x: (x + 1)/2,
                                                clip=False)
        total_samples.append(samples)
        print("sampled", samples.shape)

    total_samples = torch.cat(total_samples, dim=0)
    print("total_samples", total_samples.shape)

    nrow = int(math.ceil(math.sqrt(args.num_samples)))
    grid = torchvision.utils.make_grid(total_samples, nrow=nrow, normalize=True)                    
    
    # save
    save_dir = f"{args.logging_dir}/samples_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    torchvision.utils.save_image(grid, f"{save_dir}/samples.png")
    torch.save(total_samples, f"{save_dir}/samples.pt")

if __name__ == "__main__":
    parser = ArgumentParser()
    # model loading args
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--image_size", type=int, default=32)
    # grab a checkpoint: https://github.com/openai/improved-diffusion/tree/main and drop below
    parser.add_argument("--model_name_or_path", type=str, default="/vols/bitbucket/saravanan/distributional-mf/ckpts/openai_cifar10_linear.pt")
    parser.add_argument("--train_steps", type=int, default=4000)
    parser.add_argument("--inference_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--device", type=str, default="cuda")
    # sampling args
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    # logging dir
    parser.add_argument("--logging_dir", type=str, default="/vols/bitbucket/saravanan/distributional-mf/cache")
    args = parser.parse_args()
    
    main(args)