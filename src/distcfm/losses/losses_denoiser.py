import torch
import torch.nn as nn
from distcfm.losses.utils import l2_loss
from .losses import broadcast_to_shape

def get_denoiser_loss_fn(cfg, sde):
    def loss_fn(model, x, step):
        device = x.device
        N = x.shape[0]  # batch size
        t = torch.rand(N, device=device) * cfg.sde.t_max
        alpha_t, var_t = sde.get_coefficients(t) # Shape: [B,]
        alpha_t, var_t = broadcast_to_shape(alpha_t, x.shape), broadcast_to_shape(var_t, x.shape)
        noise = torch.randn_like(x, device=device)
        xt = alpha_t * x + torch.sqrt(var_t) * noise
        
        if cfg.model.learn_loss_weighting:
            pred, loss_weighting = model(xt, t, return_loss_weighting=True)   
        else:
            pred = model(xt, t, return_loss_weighting=False)
            loss_weighting = torch.zeros_like(pred)

        loss, loss_unweighted = l2_loss(pred, noise, loss_weighting)
        
        return {"l2_loss": loss}, {"l2_loss_unweighted": loss_unweighted}

    return loss_fn
