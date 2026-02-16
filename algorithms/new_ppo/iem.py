import torch # to train the simple model
from torch import nn, optim
import numpy as np
from collections import defaultdict

'''
https://arxiv.org/abs/1611.04717
'''

# class CountHasher:
#     def __init__(self, nbins=256):
#         self.nbins = nbins
#         self.counts = defaultdict(int)

#     def _key(self, obs: torch.Tensor):
#         # will squash the obs, each feature will now live in 0-(nbins)
#         x = obs.detach().cpu().numpy()

#         if x.size == 0:  # safety
#             return [()]
        
#         x = np.clip(x, -1.0, 1.0)
#         q = np.floor((x + 1.0) * 0.5 * (self.nbins - 1)).astype(np.int16)
#         q = np.clip(q, 0, self.nbins - 1).astype(np.int16)

#         return [tuple(row) for row in q]
    
#     def update_and_get_counts(self, obs: torch.Tensor):
#         keys = self._key(obs)
#         out = []
#         for k in keys: # k = tuple of quantized features
#             self.counts[k] += 1
#             out.append(self.counts[k])
#         return torch.tensor(out, dtype=torch.float32, device=obs.device)

#     def get_counts(self, obs:torch.Tensor):
#         keys = self._key(obs)
#         out = [self.counts.get(k, 0) for k in keys]
#         return torch.tensor(out, dtype=torch.float32, device=obs.device)
    
class CountHasher:
    def __init__(self, nbins=256, decay=0.995):
        """
        decay: per-step decay factor lambda in (0,1]. 
               1.0 means no decay (original behavior).
        """
        print("Using CountHasher with decay =", decay)
        self.nbins = nbins
        self.decay = float(decay)

        # decayed counts -> float
        self.counts = defaultdict(float)

        # last time we updated this key (for lazy decay)
        self.last_seen_step = defaultdict(int)

        # global update step (increments once per update call, not per key)
        self.global_step = 0

    def _key(self, obs: torch.Tensor):
        x = obs.detach().cpu().numpy()
        if x.size == 0:
            return [()]

        x = np.clip(x, -1.0, 1.0)
        q = np.floor((x + 1.0) * 0.5 * (self.nbins - 1)).astype(np.int16)
        q = np.clip(q, 0, self.nbins - 1).astype(np.int16)
        return [tuple(row) for row in q]

    def _apply_lazy_decay(self, k):
        if self.decay >= 1.0:
            return

        last = self.last_seen_step[k]
        dt = self.global_step - last
        if dt > 0:
            # counts[k] *= decay^dt
            self.counts[k] *= self.decay ** dt

    def update_and_get_counts(self, obs: torch.Tensor):
        """
        Updates decayed counts for each key in obs and returns the updated counts.
        """
        self.global_step += 1

        keys = self._key(obs)
        out = []
        for k in keys:
            self._apply_lazy_decay(k)
            self.counts[k] += 1.0
            self.last_seen_step[k] = self.global_step
            out.append(self.counts[k])

        return torch.tensor(out, dtype=torch.float32, device=obs.device)

    def get_counts(self, obs: torch.Tensor):
        """
        Returns counts. Optionally applies lazy decay first (read-only decay).
        Usually you don't need apply_decay for training targets since update() calls update_and_get_counts().
        """
        keys = self._key(obs)
        out = []
        out = [self.counts.get(k, 0.0) for k in keys]

        return torch.tensor(out, dtype=torch.float32, device=obs.device)


class IEModule(nn.Module):
    """
    small model that outputs the B(N) value for exploration to add to PPO loss
    """
    def __init__(self, obs_dim, lr=1e-4, c1=0.05, alpha = 0.5, normalize=False, device="cpu"): # tuning of the magnitude of the reward already done with beta in iem_ppo
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
        self.counter = CountHasher()
        self.alpha = alpha
        print(self.alpha, "iem alpha value")
      
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
            target = torch.clamp(N, min=1.0).pow(-self.alpha) # 1/(N^alpha)
            # target = torch.rsqrt(torch.clamp(N, min=1.0)) # 1/sqrt(N)

        pred = self.predictor(obs).squeeze(-1)
        loss = ((pred - target) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()



