# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
# from timm.models.vision_transformer import Attention, PatchEmbed, Mlp
from timm.models.vision_transformer import PatchEmbed, Mlp
from distcfm.models.attention import Attention, JointAttention
from distcfm.models.edm2 import MPConv, MPFourier
from distcfm.models.base_model import BaseModel

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size, use_cfg_embedding: bool = True):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

    def forward(self, labels, train, force_drop_ids=None):
        return self.embedding_table(labels)

class GuidanceEmbedderJoint(nn.Module):
    def __init__(self, d, class_ws_set, x_cond_ws_set, round_decimals=4):
        super().__init__()
        self.class_ws_set = list(class_ws_set)
        self.x_cond_ws_set = list(x_cond_ws_set)
        n_class_ws = len(self.class_ws_set)
        n_x_cond_ws = len(self.x_cond_ws_set)
        self.embedding_table = nn.Embedding(n_class_ws * n_x_cond_ws, d)
        
        # for indexing 
        self.round_decimals = round_decimals
        self.class2idx = {
            round(float(v), self.round_decimals): i
            for i, v in enumerate(self.class_ws_set)
        }
        self.xcond2idx = {
            round(float(v), self.round_decimals): j
            for j, v in enumerate(self.x_cond_ws_set)
        }

    def _map_ws_to_idx(self, ws, mapping):
        flat = ws.view(-1).tolist()
        idx_list = []
        for v in flat:
            key = round(float(v), self.round_decimals)
            if key not in mapping:
                raise ValueError(
                    f"Got ws={v} (rounded={key}), which is not in allowed set {list(mapping.keys())}"
                )
            idx_list.append(mapping[key])

        return torch.tensor(idx_list, dtype=torch.long, device=ws.device)

    def forward(self, class_ws, x_cond_ws):
        class_ws = class_ws.view(-1).float()
        x_cond_ws = x_cond_ws.view(-1).float()

        class_idx = self._map_ws_to_idx(class_ws, self.class2idx)   # (B,)
        xcond_idx = self._map_ws_to_idx(x_cond_ws, self.xcond2idx)  # (B,)
        n_x = len(self.x_cond_ws_set)
        combined_indices = class_idx * n_x + xcond_idx              # (B,)
        return self.embedding_table(combined_indices)   

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_joint_attention=False, input_size=None, patch_size=None, in_channels=None, **block_kwargs):
        super().__init__()
        self.use_joint_attention = use_joint_attention
        if use_joint_attention:
            self.x_cond_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            num_patches = self.x_cond_embedder.num_patches
            # Will use fixed sin-cos embedding:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
            self.joint_attn = JointAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.modulation_size = 7
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.modulation_size = 6

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, self.modulation_size * hidden_size, bias=True)
        )

    def forward(self, x, x_cond, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, *gate_ja = self.adaLN_modulation(c).chunk(self.modulation_size, dim=1)
        if self.use_joint_attention:
            x_cond = self.x_cond_embedder(x_cond) + self.pos_embed
            x = x + gate_msa.unsqueeze(1) * self.joint_attn(modulate(self.norm1(x), shift_msa, scale_msa), x_cond, gate_ja[0])
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        label_dim=1000,
        learn_sigma=False,
        encoder_depth=8,
        qk_norm=False,
        use_joint_attention=False,
        model_guidance_class_ws=[1.0],
        model_guidance_x_cond_ws=[1.0],
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_cond_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_cond_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        self.s_embedder = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_cond_embedder = TimestepEmbedder(hidden_size)

        self.s_embedder_second = TimestepEmbedder(hidden_size)
        self.t_embedder_second = TimestepEmbedder(hidden_size)

        self.label_dim = label_dim
        self.y_embedder = LabelEmbedder(label_dim, hidden_size)

        # Initialise guidance scale embedding
        self.guidance_embedder = GuidanceEmbedderJoint(hidden_size, model_guidance_class_ws, model_guidance_x_cond_ws)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm, use_joint_attention=use_joint_attention, input_size=input_size, patch_size=patch_size, in_channels=in_channels) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize x_cond_embedder to zero
        nn.init.constant_(self.x_cond_embedder.proj.weight, 0)
        nn.init.constant_(self.x_cond_embedder.proj.bias, 0)

        # Initialize guidance_embedder to zero embeddings
        nn.init.zeros_(self.guidance_embedder.embedding_table.weight)

        # Zero out t_cond_embedder
        nn.init.constant_(self.t_cond_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.t_cond_embedder.mlp[2].bias, 0)
        nn.init.constant_(self.t_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.t_embedder.mlp[2].bias, 0)
        nn.init.constant_(self.s_embedder_second.mlp[2].weight, 0)
        nn.init.constant_(self.s_embedder_second.mlp[2].bias, 0)

        # Initialize x_cond_adaLN to gate off x_cond
        # We set weights to 0 and bias to [0, -1].
        # This gives shift=0, scale=-1.
        # modulate(x, shift, scale) = x * (1 + scale) + shift = x * 0 + 0 = 0.
        print("Initializing x_cond_adaLN to gate off x_cond (scale=-1)...")
        nn.init.constant_(target_model.x_cond_adaLN[-1].weight, 0)
        nn.init.constant_(target_model.x_cond_adaLN[-1].bias, 0)
        half_dim = target_model.x_cond_adaLN[-1].bias.shape[0] // 2
        nn.init.constant_(target_model.x_cond_adaLN[-1].bias[half_dim:], -1)

        # Initialize DiTBlock specific components if joint attention is used
        for block in self.blocks:
            if hasattr(block, 'x_cond_embedder'):
                nn.init.constant_(block.x_cond_embedder.proj.weight, 0)
                nn.init.constant_(block.x_cond_embedder.proj.bias, 0)
            if hasattr(block, 'pos_embed'):
                num_patches = block.x_cond_embedder.num_patches
                grid_size = int(num_patches ** 0.5)
                pos_embed = get_2d_sincos_pos_embed(block.pos_embed.shape[-1], grid_size)
                block.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize x_cond_adaLN to zero
        nn.init.constant_(self.x_cond_adaLN[-1].weight, 0)
        nn.init.constant_(self.x_cond_adaLN[-1].bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_second.mlp[2].weight, std=0.02)
        nn.init.normal_(self.s_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.s_embedder_second.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, s, t, x, t_cond, x_cond, y, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Manually take all times to be 1-time to align with the SiT architecture
        s = 1 - s
        t = 1 - t
        t_cond = 1 - t_cond
        
        # for amortized classifier-free guidance
        class_cfg_scale = kwargs.get('cfg_scale', None) # [N,] or None
        x_cfg_scale = kwargs.get('x_cond_scale', None) # [N,] or None

        x_emb = self.x_embedder(x)
        x_cond_emb = self.x_cond_embedder(x_cond)
        
        t_cond = self.t_cond_embedder(t_cond)
        shift_cond, scale_cond = self.x_cond_adaLN(t_cond).chunk(2, dim=1)
        x_cond_emb = modulate(x_cond_emb, shift_cond, scale_cond)
        
        x = x_emb + x_cond_emb + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2        
        s_first = self.s_embedder(s)                   # (N, D)
        t_first = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        
        c = s_first + t_first + t_cond + y               # (N, D)
        
        # model guidance for MFM
        if class_cfg_scale is not None and x_cfg_scale is not None:
            guidance_emb = self.guidance_embedder(class_cfg_scale, x_cfg_scale)  # (N, D)
            c = c + guidance_emb

        for i, block in enumerate(self.blocks[:20]):
            x = block(x, x_cond, c)                           # (N, T, D)

        s_second = self.s_embedder_second(s)                   # (N, D)
        t_second = self.t_embedder_second(t)                   # (N, D)
        c = s_second + t_second + t_cond + y  
        for i, block in enumerate(self.blocks[20:]):
            x = block(x, x_cond, c)                      # (N, T, D)
            
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return -x

class DiTMFM(BaseModel):
    def __init__(self, learn_loss_weighting, channels=128, encoder_depth=8, **dit_kwargs):
        super().__init__()
        self.dit = DiT(encoder_depth=encoder_depth, **dit_kwargs)
        self.label_dim = self.dit.label_dim
        self.frozen = False

    def freeze_dit(self):
        for param in self.dit.parameters():
            param.requires_grad = False
            self.frozen = True
    
    def unfreeze_dit(self):
        for param in self.dit.parameters():
            param.requires_grad = True
            self.frozen = False

    def v(self, s, t, x, t_cond, x_cond, class_labels = None, **kwargs):
        if class_labels is None:
            class_labels = torch.full((x.shape[0],), self.label_dim, dtype=torch.long, device=x.device)
      
        v = self.dit(s, t, x, t_cond, x_cond, class_labels, **kwargs)
        return v
    
    def v_cfg(self, s, t, x, t_cond, x_cond, class_labels, cfg_scales, return_seperate=False):
        assert torch.equal(s, t), "implemented for velocity only!"
        device = s.device
        s_2 = torch.cat([s, s], dim=0)
        t_2 = torch.cat([t, t], dim=0)
        x_2 = torch.cat([x, x], dim=0)
        t_cond_2 = torch.cat([t_cond, t_cond], dim=0)
        x_cond_2 = torch.cat([x_cond, x_cond], dim=0)
        null_labels = torch.full((x.shape[0],), self.label_dim, dtype=torch.long, device=device)
        
        labels = torch.cat([null_labels, class_labels], dim=0)
        v = model.v(s_2, t_2, x_2, t_cond_2, x_cond_2, class_labels=labels,
                    cfg_scale=torch.ones_like(s_2, device=device))
        v_uncond, v_cond = v.chunk(2, dim=0)
        if return_seperate:
            return v_uncond, v_cond
        return v_uncond + broadcast_to_shape(cfg_scales, v_uncond.shape) * (v_cond - v_uncond)
    
    def v_cfg_mfm(self, s, t, x, t_cond, x_cond, class_labels, cfg_scales, x_cond_scales):
        assert torch.equal(s, t), "implemented for velocity only!"
        v_uncond, v_cond = self.v_cfg(s, t, x, t_cond, x_cond, class_labels, cfg_scales, return_seperate=True)
        t_zeros = torch.zeros_like(t_cond) # is this what we want?
        v_uncond_t0, v_cond_t0 = self.v_cfg(s, t, x, t_zeros, x_cond, class_labels, cfg_scales, return_seperate=True)
        
        # x_cond_mixing
        v_uncond_mixed = v_uncond_t0 + broadcast_to_shape(x_cond_scales, v_uncond.shape) * (v_uncond - v_uncond_t0)
        v_cond_mixed = v_cond_t0 + broadcast_to_shape(x_cond_scales, v_cond.shape) * (v_cond - v_cond_t0)
        
        return v_uncond_mixed + broadcast_to_shape(cfg_scales, v_uncond.shape) * (v_cond_mixed - v_uncond_mixed)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
