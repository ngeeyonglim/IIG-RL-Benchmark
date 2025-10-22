# algorithms/ppo/iem_distance_paper.py
import torch
from torch import nn, optim
import torch.nn.functional as F

class IEMDistancePaper(nn.Module):
    """
    IEM-PPO as in Zhang et al. (2020):
      - Regress N_xi(s_t, a_t, s_{t+k}) to the step count k (MSE).
      - Use r_int(t) = c1 * N_xi(s_t, a_t, s_{t+1}) during rollouts/learning.
    """
    def __init__(self, obs_dim, num_actions, hidden=256, lr=1e-3,
                 c1=0.05, normalize=True, device="cpu"):
        super().__init__()
        self.device = device
        self.c1 = c1
        self.normalize = normalize

        emb_dim = 32  # action embedding for discrete actions
        self.a_emb = nn.Embedding(num_actions, emb_dim)

        in_dim = obs_dim + emb_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        ).to(device)

        self.opt = optim.Adam(self.parameters(), lr=lr)

        # running stats to stabilize reward scale
        self.r_mean = torch.zeros(1, device=device)
        self.r_var  = torch.ones(1, device=device)
        self.r_count = 1e-6

    def _concat(self, s, a, sp):
        # s, sp: [B, D] float; a: [B] long
        ae = self.a_emb(a)                       # [B, E]
        return torch.cat([s, ae, sp], dim=-1)    # [B, D+E+D]

    # ---------- inference-time intrinsic reward (n = 1) ----------
    @torch.no_grad()
    def intrinsic_reward(self, s, a, sp, nonterminal_mask=None):
        """
        r_int = c1 * N_xi(s_t, a_t, s_{t+1})
        """
        self.eval()
        x = self._concat(s, a, sp)
        n_hat = self.net(x).squeeze(-1)         # [B], unconstrained; step counts are positive
        n_hat = torch.relu(n_hat)               # ensure non-negative prediction

        r = n_hat.clone()
        if nonterminal_mask is not None:
            r = r * nonterminal_mask.float()

        if self.normalize:
            # simple running z-score (batch-wise Welford-lite)
            n = r.numel()
            self.r_count += n
            batch_mean = r.mean()
            delta = batch_mean - self.r_mean
            self.r_mean += delta * (n / self.r_count)
            self.r_var  += ((r - self.r_mean)**2).mean()
            std = torch.sqrt(self.r_var / max(self.r_count, 1.0))
            r = (r - self.r_mean) / (std + 1e-8)

        return self.c1 * r

    # ---------- training step for the regressor ----------
    def update(self, s_seq, a_seq, done_seq, max_k=3):
        """
        Fit N_xi on (s_t, a_t, s_{t+k}) -> target k, using transitions from a rollout batch.

        Inputs are tensors shaped:
          s_seq   : [T, N, D]   observations at each step
          a_seq   : [T, N]      actions at each step (long)
          done_seq: [T, N]      1.0 if episode ended at t (i.e., transition t is terminal), else 0.0

        We form a training set of pairs for k in [1..max_k] that *do not cross terminals*.
        """
        self.train()
        T, N, D = s_seq.shape
        device = s_seq.device

        losses = []

        # For each k, build aligned (s_t, a_t, s_{t+k}) triples that don't cross episode boundaries
        for k in range(1, max_k + 1):
            # valid indices where t+k < T
            if T <= k:
                break

            s_t  = s_seq[:-k].reshape(-1, D)               # [(T-k)*N, D]
            a_t  = a_seq[:-k].reshape(-1).long()           # [(T-k)*N]
            s_tk = s_seq[k:].reshape(-1, D)                # [(T-k)*N, D]

            # Build a mask that zeroes out pairs that cross terminals:
            # A transition (t..t+k) is valid only if all intermediate steps are nonterminal.
            nonterm_mask = torch.ones_like(a_t, dtype=torch.float32, device=device)
            # if any of done_seq[t + j] == 1 for j in [0..k-1], drop it
            for j in range(k):
                nonterm = 1.0 - done_seq[j: j + T - k].reshape(-1)
                nonterm_mask = nonterm_mask * nonterm

            valid = nonterm_mask > 0.5
            if valid.sum() == 0:
                continue

            x = self._concat(s_t[valid], a_t[valid], s_tk[valid])     # [Bv, D+E+D]
            pred = self.net(x).squeeze(-1)
            pred = torch.relu(pred)                                   # non-negative steps
            target = torch.full_like(pred, float(k))                  # label = k (step count)

            loss = F.mse_loss(pred, target)
            losses.append(loss)

        if not losses:
            return 0.0

        total = torch.stack(losses).mean()
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        return float(total.detach().cpu().item())
