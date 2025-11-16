from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn.functional as F

def _safe_l2_norm(x: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)

def _make_gaussian_1d_kernel(kernel_size: int, sigma: float, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel  # shape: (K,)

def _gaussian_blur_2d(x: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    """
    Depthwise separable Gaussian blur for NCHW input.
    """
    b, c, h, w = x.shape
    k = int(6 * sigma + 1) if kernel_size is None else int(kernel_size)
    k = max(3, k | 1)  # ensure odd and at least 3
    device, dtype = x.device, x.dtype

    k1d = _make_gaussian_1d_kernel(k, float(sigma), device, dtype)
    kernel_x = k1d.view(1, 1, 1, k).repeat(c, 1, 1, 1)     # (C,1,1,K)
    kernel_y = k1d.view(1, 1, k, 1).repeat(c, 1, 1, 1)     # (C,1,K,1)

    pad = k // 2
    y = F.conv2d(x, kernel_x, bias=None, stride=1, padding=(0, pad), groups=c)
    y = F.conv2d(y, kernel_y, bias=None, stride=1, padding=(pad, 0), groups=c)

    return y

def _downsample(x: torch.Tensor, factor: int, mode: Literal['nearest','bicubic']) -> torch.Tensor:
    b, c, h, w = x.shape
    out = F.interpolate(
        x, size=(h // factor, w // factor),
        mode=mode, align_corners=False if mode in ('bicubic', 'bilinear') else None
    )
    return out

class InverseProblem(ABC):
    """Abstract base for inverse problems with a forward operator and likelihood."""
    def __init__(self, noise_sigma: float):
        self.noise_sigma = float(noise_sigma)

    @abstractmethod
    def _Ax(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        """y = A(x) + noise"""
        Ax = self._Ax(x)
        noise = torch.randn_like(Ax) * self.noise_sigma
        return Ax + noise

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log p(y | x) -> (batch,)"""
        Ax = self._Ax(x)
        diff = y - Ax        
        # print("diff zeros", (diff == 0).sum()/diff.numel())
        sq = torch.sum(diff * diff, dim=tuple(range(1, diff.ndim)))
        return -0.5 * (sq / (self.noise_sigma ** 2))

    def log_likelihood_dps(self, x: torch.Tensor, y: torch.Tensor, step_size: float) -> torch.Tensor:
        """Default DPS scaling variant; subclass can override if needed."""
        diff = y - self._Ax(x)
        b = diff.shape[0]
        diff_flat = diff.view(b, -1)
        diff_l2 = _safe_l2_norm(diff_flat, dim=1).detach()  # stop-gradient
        sq = torch.sum(diff_flat * diff_flat, dim=1)
        return -(step_size / diff_l2) * sq

class Inpainting(InverseProblem):
    def __init__(self, noise_sigma: float, type: str, 
                 box_half_width=0.125, **kwargs):
        super().__init__(noise_sigma)
        self.type = type
        self.box_half_width = box_half_width
        
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        mask = torch.zeros_like(x)
        
        if self.type == 'box':
            hw = int(self.box_half_width * min(h, w))
            y0, y1 = h // 2 - hw, h // 2 + hw
            x0, x1 = w // 2 - hw, w // 2 + hw
            mask[:, :, y0:y1, x0:x1] = 1.0
            mask = 1.0 - mask
        elif self.type == 'half':
            mask[:, :, :h // 2, :] = 1.0
        
        return mask

    def _Ax(self, x: torch.Tensor) -> torch.Tensor:
        m = self._mask(x)
        return x * m
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_sigma
        return self._Ax(x + noise)

class GaussianDeblur(InverseProblem):
    def __init__(self, noise_sigma: float, kernel_sigma: float, 
                 kernel_size: Optional[int] = None, **kwargs):
        super().__init__(noise_sigma)
        self.noise_sigma = noise_sigma
        self.kernel_sigma = kernel_sigma
        self.kernel_size = kernel_size

    def _Ax(self, x: torch.Tensor) -> torch.Tensor:
        return _gaussian_blur_2d(x, sigma=self.kernel_sigma, 
                                 kernel_size=self.kernel_size)

class SuperResolution(InverseProblem):
    def __init__(self, noise_sigma: float, factor: int, type: str, **kwargs):
        super().__init__(noise_sigma)
        self.factor = factor
        self.type = type

    def _Ax(self, x: torch.Tensor) -> torch.Tensor:
        return _downsample(x, self.factor, self.type)
    

class Colourization(InverseProblem):
    def __init__(
        self,
        noise_sigma: float,
        weights: Literal['bt601', 'bt709'] | torch.Tensor | tuple[float, float, float] = 'bt709',
        **kwargs
    ):
        super().__init__(noise_sigma)
        if isinstance(weights, str):
            if weights.lower() == 'bt601':
                w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
            elif weights.lower() == 'bt709':
                w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
        else:
            w = torch.tensor(weights, dtype=torch.float32)
        self.weights = w / w.sum()

    def _Ax(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        y = (x * w).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        y = y.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Ax = self._Ax(x)
        noise = torch.randn_like(Ax) * self.noise_sigma
        return Ax + noise