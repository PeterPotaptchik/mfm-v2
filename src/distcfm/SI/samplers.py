import math
import torch
import tqdm
from torchdiffeq import odeint

from distcfm.losses.utils import broadcast_to_shape

@torch.no_grad()
def ode_sampler_fn(model, xt_cond, t_cond, n_steps=100, solver='euler', 
                   eps_start=None, v_type="standard", labels=None, 
                   cfg_scales=None, x_cond_scales=None):
    if v_type == "standard":
        ode_func = lambda t, x: model.v(t.expand(x.shape[0],), 
                                        t.expand(x.shape[0],), 
                                        x, t_cond, xt_cond)
    elif v_type == "cfg_mfm":
        ode_func = lambda t, x: model.v_cfg_mfm(t.expand(x.shape[0],), 
                                                t.expand(x.shape[0],), 
                                                x, t_cond, xt_cond,
                                                class_labels=labels, 
                                                cfg_scales=cfg_scales,
                                                x_cond_scales=x_cond_scales)
    elif v_type == "cfg":
        ode_func = lambda t, x: model.v_cfg(t.expand(x.shape[0],), 
                                            t.expand(x.shape[0],), 
                                            x, t_cond, xt_cond,
                                            class_labels=labels, 
                                            cfg_scales=cfg_scales)
    else:
        raise ValueError(f"Unknown velocity: {v_type}")

    if eps_start is None:
        eps_start = torch.randn_like(xt_cond)  # Initial condition
    else:
        eps_start = eps_start
    times = torch.linspace(0, 1, n_steps, device=eps_start.device)  # Time points
    sampling_hist = odeint(ode_func, eps_start, times, method=solver, atol=1e-5, rtol=1e-5)  # Last time point
    return sampling_hist[-1]

def consistency_sampler_fn(model, xt_cond, t_cond, n_steps=1, eps_start=None, labels=None):
    if eps_start is None:
        eps_start = torch.randn_like(xt_cond) # [B, C, H, W]
    else:
        eps_start = eps_start
    flow_timesteps = torch.linspace(0, 1,
                                    n_steps + 1, 
                                    device=eps_start.device)
    sampling_hist = torch.zeros((n_steps + 1, *eps_start.shape),
                                device=eps_start.device)
    sampling_hist[0] = eps_start

    for i in range(len(flow_timesteps) - 1):
        s, u = flow_timesteps[i], flow_timesteps[i + 1]
        xs = sampling_hist[i]
        s = s.repeat(xs.shape[0],)
        u = u.repeat(xs.shape[0],)
        vsu = model.v(s, u, xs, t_cond, xt_cond)
        xu = model.X(s, u, xs, vsu)
        sampling_hist[i + 1] = xu
    return sampling_hist[-1]

def kernel_sampler_fn(model, shape, shape_decoded, SI, n_samples,
                      n_batch_size, n_steps=1,
                      inverse_scaler_fn=lambda x: (x+1)/2,
                      x0=None):
    device = next(model.parameters()).device
    samples = torch.zeros((n_samples, *shape_decoded), device=device)
    timesteps = torch.linspace(0.0, SI.t_max, n_steps + 1, device=device) 
    num_batches = math.ceil(n_samples / n_batch_size)

    with torch.no_grad():
        for i in tqdm.tqdm(range(num_batches), desc="Unconditional Batches"):
            start = i * n_batch_size
            end = min(start + n_batch_size, n_samples)
            cur_bs = end - start  

            if x0 is not None:
                xs = x0[start:end] 
            else:
                xs = torch.randn((cur_bs, *shape), device=device)
            
            t_zeros = torch.zeros((cur_bs,), device=device)
            t_ones = torch.ones((cur_bs,), device=device)
            
            for j in range(len(timesteps) - 1):
                s = timesteps[j]
                u = timesteps[j + 1]
                
                s_batch = torch.full((cur_bs,), s, device=device)
                
                eps_start = torch.randn_like(xs)
                x1 = model(t_zeros, t_ones, eps_start, s_batch, xs)
                noise = torch.randn_like(xs)

                alpha_u, beta_u = SI.get_coefficients(u) 
                alpha_u, beta_u = broadcast_to_shape(alpha_u, x1.shape), broadcast_to_shape(beta_u, x1.shape)
                xu = alpha_u * noise + beta_u * x1
                xs = xu
            
            samples[start:end] = inverse_scaler_fn(xs)

    return samples