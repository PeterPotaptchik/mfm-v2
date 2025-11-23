import os
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from diffusers import AutoencoderKL

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_repa_z_dims():
    return 384

class RepaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._saved_first_decoded_batch = False
        
        # Hardcoded DINOv2 small
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        del self.encoder.head
        self.encoder.head = torch.nn.Identity()
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Load VAE if needed
        self.vae = None
        if cfg.dataset.name == "imagenet_latent":
             # We use the same VAE as in data module
             self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
             self.vae.eval()
             for p in self.vae.parameters():
                 p.requires_grad = False
             self.register_buffer('latents_scale', torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1))
             self.register_buffer('latents_bias', torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # x is x1 (target)
        # If latent, decode
        if self.vae is not None:
            # x is latent
            x = (x - self.latents_bias) / self.latents_scale
            x = self.vae.decode(x).sample
            # x is roughly [-1, 1]
            x = (x + 1) / 2.
            x = x.clamp(0, 1)
            if not self._saved_first_decoded_batch:
                self._saved_first_decoded_batch = True
                save_dir = getattr(self.cfg, "work_dir", ".")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "repa_first_decoded_batch.png")
                try:
                    save_image(x.detach().cpu(), save_path, nrow=min(8, x.shape[0]))
                    print(f"Saved first decoded VAE batch to {save_path}")
                except Exception as exc:
                    print(f"Failed to save decoded VAE batch: {exc}")
        else:
            # x is image. Assumed [-1, 1] from data loader scaler?
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
