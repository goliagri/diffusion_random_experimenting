"""
Simple MLP denoiser for diffusion models.
"""

import torch as th
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = th.log(th.tensor(10000.0)) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = th.cat([th.sin(emb), th.cos(emb)], dim=-1)
        return emb


class MLPDenoiser(nn.Module):
    """
    Simple MLP denoiser for diffusion.
    
    Takes noisy input, timestep t, and optional conditioning, outputs denoised prediction.
    """
    
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=4, time_embed_dim=64, cond_dim=1):
        """
        Args:
            input_dim: Dimension of the data being denoised
            hidden_dim: Hidden layer width
            num_layers: Number of MLP layers
            time_embed_dim: Dimension of timestep embedding
            cond_dim: Dimension of conditioning input (0 for unconditional)
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        
        # Time embedding: sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # Conditioning embedding (if cond_dim > 0)
        if cond_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            total_input_dim = input_dim + time_embed_dim + hidden_dim
        else:
            self.cond_embed = None
            total_input_dim = input_dim + time_embed_dim
        
        # Main network
        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t, **kwargs):
        """
        Args:
            x: Noisy input [batch, input_dim]
            t: Timesteps [batch]
            **kwargs: Conditioning values. Looks for 'cond' key by default.
        
        Returns:
            Denoised prediction [batch, input_dim]
        """
        # Embed time
        t_emb = self.time_embed(t)
        
        # Handle conditioning
        if self.cond_embed is not None:
            cond = kwargs.get('cond', None)
            if cond is None:
                cond = th.zeros(x.shape[0], self.cond_dim, device=x.device)
            c_emb = self.cond_embed(cond)
            combined = th.cat([x, t_emb, c_emb], dim=-1)
        else:
            combined = th.cat([x, t_emb], dim=-1)
        
        return self.net(combined)

