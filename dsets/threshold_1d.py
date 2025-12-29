"""
Simple 1D threshold dataset for diffusion experiments.

Samples x from N(0, 1), adds noise, then applies stochastic +/-5 shift:
  - if noisy_x <= -1: 100% chance of -5
  - if noisy_x >= +1: 100% chance of +5
  - between -1 and +1: linear interpolation of probability
"""

import torch
from torch.utils.data import Dataset


class Threshold1DDataset(Dataset):
    """
    1D dataset where target is shifted based on stochastic threshold of noisy input.
    
    x ~ N(0, 1)
    noisy_x = x + noise, where noise ~ N(0, noise_std)
    
    Stochastic +/-5 shift:
      - noisy_x <= -1: 100% chance of -5
      - noisy_x >= +1: 100% chance of +5
      - between -1 and +1: probability of +5 increases linearly from 0% to 100%
    
    This creates a soft transition zone, interesting for testing how well 
    diffusion models can learn probabilistic mappings.
    """
    
    def __init__(self, size: int, seed: int = None, noise_std: float = 1.0):
        """
        Args:
            size: Number of samples in the dataset
            seed: Random seed for reproducibility (None = random each time)
            noise_std: Standard deviation of noise added to x before threshold (default 1.0)
        """
        super().__init__()
        self.size = size
        self.noise_std = noise_std
        
        # Generate data
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        # Sample from standard normal
        self.x = torch.randn(size, 1, generator=generator)
        
        # Add noise to x before applying threshold
        noise = torch.randn(size, 1, generator=generator) * noise_std
        noisy_x = self.x + noise
        
        # Compute probability of +5: linear from 0 at -1 to 1 at +1, clamped
        prob_plus = torch.clamp((noisy_x + 1) / 2, 0, 1)
        
        # Sample uniform random to decide +5 or -5
        rand_samples = torch.rand(size, 1, generator=generator)
        use_plus = rand_samples < prob_plus
        
        # Apply stochastic shift
        self.y = torch.where(
            use_plus,
            noisy_x + 5.0,
            noisy_x - 5.0
        )
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """
        Returns:
            (y, cond_dict) tuple where:
                - y is the target value [1] that diffusion learns to generate
                - cond_dict contains conditioning info {'cond': input_value}
        """
        return self.y[idx], {'cond': self.x[idx]}
    
    def get_x(self, idx):
        """Get just the input x value."""
        return self.x[idx]
    
    def get_y(self, idx):
        """Get just the target y value."""
        return self.y[idx]
    
    @property
    def input_dim(self):
        return 1
    
    @property
    def output_dim(self):
        return 1


class Threshold1DUncondDataset(Dataset):
    """
    Unconditional version - just learns the marginal distribution of y.
    
    Useful for testing if diffusion can learn a bimodal distribution.
    """
    
    def __init__(self, size: int, seed: int = None, noise_std: float = 1.0):
        super().__init__()
        self.size = size
        self.noise_std = noise_std
        
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        x = torch.randn(size, 1, generator=generator)
        noise = torch.randn(size, 1, generator=generator) * noise_std
        noisy_x = x + noise
        
        # Stochastic +/-5 shift
        prob_plus = torch.clamp((noisy_x + 1) / 2, 0, 1)
        rand_samples = torch.rand(size, 1, generator=generator)
        use_plus = rand_samples < prob_plus
        
        self.y = torch.where(use_plus, noisy_x + 5.0, noisy_x - 5.0)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return just y with empty condition dict
        return self.y[idx], {}


if __name__ == "__main__":
    # Quick test
    ds = Threshold1DDataset(1000, seed=42, noise_std=1.0)
    print(f"Dataset size: {len(ds)}")
    print(f"Noise std: {ds.noise_std}")
    
    y, cond = ds[0]
    print(f"Sample 0: y={y.item():.3f}, x={cond['cond'].item():.3f}")
    
    # Show some samples (note: y now includes noise, so won't match simple x+/-5)
    print("\nSample data points (y includes noise before threshold):")
    for i in range(5):
        y, cond = ds[i]
        x = cond['cond'].item()
        print(f"x={x:.3f}, y={y.item():.3f}")

