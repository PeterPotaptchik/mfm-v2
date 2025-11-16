import torch
import torch.nn as nn
import math
from distcfm.models.base_model import BaseModel


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        freq = torch.exp(torch.linspace(0, math.log(10000), half_dim))
        self.register_buffer("freq", freq)  # static buffer

    def forward(self, t):
        """
        t: Tensor of shape [B] or [B, 1]
        returns: [B, embedding_dim]
        """
        t = t.unsqueeze(-1)  # [B, 1]
        args = t * self.freq  # [B, half_dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, D]


class MLP(BaseModel):
    def __init__(self, input_dim, output_dim, 
                 hidden_dims, time_embedding_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.time_embedding_dim = time_embedding_dim

        # Time embedders
        self.embed_s = TimeEmbedding(time_embedding_dim)
        self.embed_t = TimeEmbedding(time_embedding_dim)
        self.embed_t_cond = TimeEmbedding(time_embedding_dim)

        # Linear projections to match hidden_dim[0]
        h0 = hidden_dims[0]
        self.s_proj = nn.Linear(time_embedding_dim, h0)
        self.t_proj = nn.Linear(time_embedding_dim, h0)
        self.t_cond_proj = nn.Linear(time_embedding_dim, h0)
        self.x_proj = nn.Linear(input_dim, h0)
        self.x_cond_proj = nn.Linear(input_dim, h0)

        # MLP
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def v(self, s, t, x, t_cond, x_cond):
        """
        s, t, t_cond: [B]
        x, x_cond: [B, D]
        """
        B, D = x.shape
        device = x.device

        s_emb = self.s_proj(self.embed_s(s))
        t_emb = self.t_proj(self.embed_t(t))
        x_emb = self.x_proj(x)
        x_cond_emb = self.x_cond_proj(x_cond)
        t_cond_emb = self.t_cond_proj(self.embed_t_cond(t_cond))

        h = s_emb + t_emb + t_cond_emb + x_emb + x_cond_emb
        return self.mlp(h)