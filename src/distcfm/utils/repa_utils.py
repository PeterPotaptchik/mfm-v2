import os
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from diffusers import AutoencoderKL

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_repa_z_dims():
    return 768

class RepaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._saved_first_decoded_batch = False
        
        # Hardcoded DINOv2 base
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        del self.encoder.head
        self.encoder.head = torch.nn.Identity()
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # x is image. Assumed [-1, 1] from data loader scaler.
        # distcfm image_scaler scales [0, 1] to [-1, 1].
        # So we need to inverse scale to [0, 1].
        x = (x + 1) / 2.
        x = x.clamp(0, 1)

        # Preprocess for DINOv2
        resolution = x.shape[-1]
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        target_res = 224 * (resolution // 256)
        if target_res > 0 and target_res != resolution:
            x = torch.nn.functional.interpolate(x, size=(target_res, target_res), mode='bicubic')

        z = self.encoder.forward_features(x)
        z = z['x_norm_patchtokens']
        return z
