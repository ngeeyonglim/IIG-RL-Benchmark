import torch # to train the simple model
from torch import nn, optim
import numpy as np
from collections import defaultdict

'''
https://arxiv.org/abs/1611.04717?utm_source=chatgpt.com
'''

class CountHasher:
    def __init__(self, nbins=128):
        self.nbins = nbins
        self.counts = defaultdict(int)

    def _key(self, obs: torch.Tensor):
        # will squash the obs, each feature will now live in 0-(nbins)
        x = obs.detach().cpu().numpy()

        if x.size == 0:  # safety
            return [()]
        
        x = np.tanh(x)                      # squash
        q = np.floor((x + 1.0) * 0.5 * (self.nbins - 1)).astype(np.int16)
        return [tuple(row) for row in q]
    


    def update_and_get_counts(self, obs: torch.Tensor):
        keys = self._key(obs)
        out = []
        for k in keys: # k = tuple of quantized features
            self.counts[k] += 1
            out.append(self.counts[k])
        return torch.tensor(out, dtype=torch.float32, device=obs.device)

    def get_counts(self, obs:torch.Tensor):
        keys = self._key(obs)
        out = [self.counts.get(k, 0) for k in keys]
        return torch.tensor(out, dtype=torch.float32, device=obs.device)
    
class IEModule(nn.Module):
    """
    small model that outputs the B(N) value for exploration to add to PPO loss
    """
    def __init__(self, obs_dim, lr=1e-3, c1=0.05, normalize=True, device="cpu"):
        super().__init__()
        self.device = device
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.c1 = c1
        self.normalize = normalize
        self.r_mean = torch.zeros(1, device=device)
        self.r_var  = torch.ones(1, device=device)
        self.r_count = 1e-6
        self.counter = CountHasher(nbins=128)
      
    @torch.no_grad()
    def intrinsic_reward(self, obs: torch.Tensor):
        pred = self.predictor(obs).squeeze(-1)
        r = self.c1 * pred
        if self.normalize:
            # runnign mean
            self.r_count += r.numel()
            delta = r.mean() - self.r_mean
            self.r_mean += delta * (r.numel() / self.r_count)
            self.r_var  += ((r - self.r_mean)**2).mean()
            std = torch.sqrt(self.r_var / max(self.r_count, 1.0))
            r = (r - self.r_mean) / (std + 1e-8)
        return r
  
    def update(self, obs: torch.Tensor):
        """
        Supervised training step
        """
        self.train()
        with torch.no_grad():
            N = self.counter.update_and_get_counts(obs)   # N = B, obs = [B, obsdim]
            target = torch.rsqrt(torch.clamp(N, min=1.0)) # 1/sqrt(N)

        pred = self.predictor(obs).squeeze(-1)
        loss = ((pred - target) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()



