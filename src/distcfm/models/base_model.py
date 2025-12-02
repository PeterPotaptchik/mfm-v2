import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import math
import numpy as np
import itertools

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1,):
        w = self.weight.to(torch.float32)
        # if self.training:
            # with torch.no_grad():
            #     self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def v(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        """Should return the velocity. Must be implemented by subclass."""
        pass
    
    def v_cfg(self, s, t, x, t_cond, x_cond, class_labels, null_labels, cfg_scales):
        device = s.device
        s_2 = torch.cat([s, s], dim=0)
        t_2 = torch.cat([t, t], dim=0)
        x_2 = torch.cat([x, x], dim=0)
        t_cond_2 = torch.cat([t_cond, t_cond], dim=0)
        x_cond_2 = torch.cat([x_cond, x_cond], dim=0)
        labels = torch.cat([null_labels, class_labels], dim=0)

        v = model.v(s_2, t_2, x_2, t_cond_2, x_cond_2, class_labels=labels,
                    cfg_scale=torch.ones_like(s_2, device=device))
        v_uncond, v_cond = v.chunk(2, dim=0)
        return v_uncond + broadcast_to_shape(cfg_scales, v_uncond.shape) * (v_cond - v_uncond)

    def X(self, s, t, x, v):
        s = broadcast_to_shape(s, x.shape)
        t = broadcast_to_shape(t, x.shape)
        return x + (t - s) * v

    def X_and_v(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        v = self.forward(s, t, x, t_cond, x_cond, class_labels=class_labels, **kwargs)
        return self.X(s, t, x, v), v
    
    def forward(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        """Forward pass that computes the map."""
        v = self.v(s, t, x, t_cond, x_cond, class_labels=class_labels, **kwargs)
        return self.X(s, t, x, v)
    
class LossWeightingNetwork(nn.Module):
    """
    A network to compute loss weighting based on timesteps.
    """
    def __init__(self, channels=128, clamp_min=-10.0, clamp_max=10.0):
        super().__init__()
        self.linear = MPConv(channels, 1, kernel=[])
        self.emb_fourier = MPFourier(channels)
        self.emb_noise_t = MPConv(channels, channels, kernel=[])
        self.emb_noise_t_cond = MPConv(channels, channels, kernel=[])
        self._weighting_stats = []
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def forward(self, t, t_cond):
        t_emb = self.emb_noise_t(self.emb_fourier(t))
        t_cond_emb = self.emb_noise_t_cond(self.emb_fourier(t_cond))
        combined = (t_emb + t_cond_emb) / math.sqrt(2.0)
        weighting = self.linear(combined)
        self._record_weighting_stats(weighting)
        weighting = weighting.reshape(-1, 1, 1, 1)
        weighting = torch.clamp(weighting, self.clamp_min, self.clamp_max)
        return weighting

    def _record_weighting_stats(self, weighting):
        with torch.no_grad():
            stats = {
                "weighting_mean": weighting.mean().detach(),
                "weighting_std": weighting.std(unbiased=False).detach(),
                "weighting_var_mean": weighting.exp().mean().detach(),
                "weighting_var_min": weighting.exp().min().detach(),
                "weighting_var_max": weighting.exp().max().detach(),
            }
        self._weighting_stats.append(stats)

    def pop_weighting_stats(self):
        if not self._weighting_stats:
            return None
        aggregated = {}
        for key in self._weighting_stats[0]:
            aggregated[key] = torch.stack([stats[key] for stats in self._weighting_stats]).mean().detach()
        self._weighting_stats.clear()
        return aggregated