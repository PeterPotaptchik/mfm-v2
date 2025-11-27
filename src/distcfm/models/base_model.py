import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import math
from distcfm.models.edm2 import MPConv, MPFourier
import itertools

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def v(self, s, t, x, t_cond, x_cond, class_labels=None):
        """Should return the velocity. Must be implemented by subclass."""
        pass

    def X(self, s, t, x, v):
        s = broadcast_to_shape(s, x.shape)
        t = broadcast_to_shape(t, x.shape)
        return x + (t - s) * v

    def X_and_v(self, s, t, x, t_cond, x_cond, class_labels=None):
        v = self.forward(s, t, x, t_cond, x_cond, class_labels=class_labels)
        return self.X(s, t, x, v), v
    
    def forward(self, s, t, x, t_cond, x_cond, class_labels=None):
        """Forward pass that computes the map."""
        v = self.v(s, t, x, t_cond, x_cond, class_labels=class_labels)
        return self.X(s, t, x, v)

class LossWeightingNetwork(nn.Module):
    """
    A network to compute loss weighting based on timesteps.
    """
    def __init__(self, channels=128, clamp_min=-10.0, clamp_max=10.0):
        super().__init__()
        self.linear = MPConv(channels, 1, kernel=[])
        self.linear_off_diag = MPConv(channels, 1, kernel=[])
        self.emb_fourier = MPFourier(channels)
        self.emb_noise_t = MPConv(channels, channels, kernel=[])
        self.emb_noise_s = MPConv(channels, channels, kernel=[])
        self.emb_noise_t_cond = MPConv(channels, channels, kernel=[])
        self._weighting_stats = []
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _compute_both_weights(self, s, t, t_cond):
        s_emb = self.emb_noise_s(self.emb_fourier(s))
        t_emb = self.emb_noise_t(self.emb_fourier(t))
        t_cond_emb = self.emb_noise_t_cond(self.emb_fourier(t_cond))
        combined = (s_emb + t_emb + t_cond_emb) / math.sqrt(3.0)
        diagonal_weighting = self.linear(combined)
        off_diagonal_weighting = self.linear_off_diag(combined)
        return diagonal_weighting, off_diagonal_weighting

    def _compute_weighting(self, s, t, t_cond, ema_state=None):
        if not ema_state:
            diag, off_diag = self._compute_both_weights(s, t, t_cond)
        else: # straight-through estimator
            diag_curr, off_diag_curr = self._compute_both_weights(s, t, t_cond)  # current
            diag_ema, off_diag_ema = ema_state._average_model.module.weighting_model._compute_both_weights(s, t, t_cond) # ema
            diag = diag_ema.detach() + (diag_curr - diag_curr.detach())
            off_diag = off_diag_ema.detach() + (off_diag_curr - off_diag_curr.detach())
        return diag, off_diag
    
    def forward(self, s, t, t_cond, ema_state=None):
        diagonal_weighting, off_diagonal_weighting = self._compute_weighting(s, t, t_cond, ema_state)
        is_diag_bool = torch.eq(s, t)
        is_diag = is_diag_bool.float()[:, None]   
        weighting = is_diag * diagonal_weighting + (1.0 - is_diag) * off_diagonal_weighting
        diagonal_weighting, off_diagonal_weighting = diagonal_weighting[is_diag_bool], off_diagonal_weighting[~is_diag_bool]
        self._record_weighting_stats(diagonal_weighting, off_diagonal_weighting)
        weighting = weighting.reshape(-1, 1, 1, 1)
        weighting = torch.clamp(weighting, self.clamp_min, self.clamp_max)
        return weighting

    def _record_weighting_stats(self, diagonal_weighting, off_diagonal_weighting):
        with torch.no_grad():
            stats = {
                "diag_weighting_mean": diagonal_weighting.mean().detach(),
                "diag_weighting_std": diagonal_weighting.std(unbiased=False).detach(),
                "diag_var_mean": diagonal_weighting.exp().mean().detach(),
                "off_diag_weighting_mean": off_diagonal_weighting.mean().detach(),
                "off_diag_weighting_std": off_diagonal_weighting.std(unbiased=False).detach(),
                "off_diag_var_mean": off_diagonal_weighting.exp().mean().detach(),
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