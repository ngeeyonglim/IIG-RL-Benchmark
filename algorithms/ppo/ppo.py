# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of PPO with Tsallis Entropy.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py.

Currently only supports the single-agent case.
"""

import time
import os
import hashlib
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

from utils import log_to_csv

INVALID_ACTION_PENALTY = -1e9


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    """A masked categorical."""

    # pylint: disable=dangerous-default-value
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PPOAgent(nn.Module):
    """A PPO agent module."""

    def __init__(self, num_actions, observation_shape, device):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )
        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        logits = self.actor(x)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            self.critic(x),
            probs.probs,
        )


class PPOAtariAgent(nn.Module):
    """A PPO Atari agent module."""

    def __init__(self, num_actions, observation_shape, device):
        super(PPOAtariAgent, self).__init__()
        # Note: this network is intended for atari games, taken from
        # https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.num_actions = num_actions
        self.device = device
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )

        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            self.critic(hidden),
            probs.probs,
        )


def legal_actions_to_mask(legal_actions_list, num_actions):
    """Converts a list of legal actions to a mask.

    The mask has size num actions with a 1 in a legal positions.

    Args:
      legal_actions_list: the list of legal actions
      num_actions: number of actions (width of mask)

    Returns:
      legal actions mask.
    """
    legal_actions_mask = torch.zeros(
        (len(legal_actions_list), num_actions), dtype=torch.bool
    )
    for i, legal_actions in enumerate(legal_actions_list):
        legal_actions_mask[i, legal_actions] = 1
    return legal_actions_mask


class PPO(nn.Module):
    """A PPO class.

    This class implements a PPO agent with IEM.
    """

    def __init__(
        self,
        input_shape,
        num_actions,
        num_players,
        num_envs=1,
        steps_per_batch=128,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        clip_coef=0.2,
        clip_vloss=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        device="cpu",
        agent_fn=PPOAtariAgent,
        log_file=None,
        tsallis_q=2.0,
        **kwargs,
    ):
        super().__init__()

        print("Using Tsallis-PPO")

        self.input_shape = (np.array(input_shape).prod(),)
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.log_file = log_file

        # Training settings
        self.num_envs = num_envs
        self.steps_per_batch = steps_per_batch
        self.batch_size = self.num_envs * self.steps_per_batch
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.anneal_lr = kwargs.get("anneal_lr", False)

        # Loss function
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        # Tsallis entropy configuration (q != 1). We will use normalized Tsallis entropy in [0, 1].
        self.tsallis_q = float(tsallis_q)
        if abs(self.tsallis_q - 1.0) < 1e-8:
            raise ValueError(
                "tsallis_q must be != 1.0 (q=1 corresponds to Shannon entropy)."
            )
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Initialize networks
        self.network = agent_fn(self.num_actions, self.input_shape, device).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

        # Initialize training buffers
        self.legal_actions_mask = torch.zeros(
            (self.steps_per_batch, self.num_envs, self.num_actions), dtype=torch.bool
        ).to(device)
        self.obs = torch.zeros(
            (self.steps_per_batch, self.num_envs, *self.input_shape)
        ).to(device)
        self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.logprobs = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.current_players = torch.zeros((self.steps_per_batch, self.num_envs)).to(
            device
        )

        # Initialize counters
        self.cur_batch_idx = 0
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()


        # Init for entropy logging
        self.accumulated_entropy = 0.0
        self.entropy_count = 0
        self.last_log_step = 0
        self.log_interval = 100000

        # ----------------------------
        # Visitation tracking (ported)
        # ----------------------------
        self.track_visitation = True
        self.visitation_topk = 500

        # Only enable file logging if we have a log_file path
        if self.log_file is None:
            self.track_visitation = False
            self.visitation_csv_path = None
        else:
            self.visitation_csv_path = self.log_file.replace(
                "train_log.csv", "visitation_topk.csv"
            )

        # Running counters over the current log window
        self._visit_counter_p0 = Counter()
        self._visit_counter_p1 = Counter()
        self._visit_total_p0 = 0
        self._visit_total_p1 = 0

        # --- B2: SimHash bucket visitation ---
        self.visitation_use_simhash = True

        # Number of bits in full SimHash signature (more bits = finer similarity)
        self.simhash_bits = 64

        # Number of prefix bits used as bucket id (grid size = 2^prefix_bits)
        # 10 -> 1024 buckets, 12 -> 4096 buckets
        self.simhash_prefix_bits = 14

        # Make it deterministic across runs if you want reproducible bucket layouts
        self.simhash_seed = 12345

        # Pre-sample random hyperplanes for SimHash: [bits, D]
        rng = np.random.default_rng(self.simhash_seed)
        D = int(np.array(self.input_shape).prod())
        self._simhash_planes = rng.standard_normal(size=(self.simhash_bits, D)).astype(
            np.float32
        )

        # Write CSV header once
        if (
            self.track_visitation
            and (self.visitation_csv_path is not None)
            and (not os.path.exists(self.visitation_csv_path))
        ):
            with open(self.visitation_csv_path, "w") as f:
                f.write("steps,player,bucket_id,count\n")

        self.visitation_quantize = True

    def get_value(self, x):
        return self.network.get_value(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        return self.network.get_action_and_value(x, legal_actions_mask, action)

    # ----------------------------
    # Visitation helpers (ported)
    # ----------------------------
    def _simhash_bucket_id(self, x: np.ndarray) -> int:
        """
        SimHash bucket for an info_state vector x (shape [D]).
        Returns an int in [0, 2^simhash_prefix_bits).
        """
        if self.visitation_quantize:
            v = np.rint(x).astype(np.float32, copy=False)
        else:
            v = x.astype(np.float32, copy=False)

        proj = self._simhash_planes @ v
        bits = (proj >= 0)

        k = self.simhash_prefix_bits
        bucket = 0
        for i in range(k):
            bucket = (bucket << 1) | int(bits[i])
        return bucket

    def _infoset_id_from_tensor(self, x: np.ndarray) -> str:
        """
        Stable infoset identity from info_state tensor.
        x: shape [D] numpy array (float or int)
        Returns: short hex string id
        """
        if self.visitation_quantize:
            xb = np.rint(x).astype(np.int8, copy=False).tobytes()
        else:
            xb = x.astype(np.float32, copy=False).tobytes()

        return hashlib.blake2b(xb, digest_size=8).hexdigest()

    def _tsallis_entropy_norm(
        self, probs: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalized Tsallis entropy in [0, 1] for categorical policies with varying legal action counts.

        Args:
            probs:      [B, A] probabilities (policy probabilities).
            legal_mask: [B, A] bool mask of legal actions.

        Returns:
            ent_norm: [B] normalized Tsallis entropy.
        """
        eps = 1e-12
        q = self.tsallis_q

        p = probs * legal_mask.to(probs.dtype)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        p = torch.clamp(p, min=0.0, max=1.0)

        sum_pq = (p**q).sum(dim=-1)  # [B]
        Hq = (1.0 - sum_pq) / (q - 1.0)

        n = legal_mask.sum(dim=-1).to(probs.dtype)
        n = torch.clamp(n, min=1.0)
        Hq_max = (1.0 - n ** (1.0 - q)) / (q - 1.0)

        ent_norm = Hq / (Hq_max + eps)
        return torch.clamp(ent_norm, 0.0, 1.0)

    def step(self, time_step, is_evaluation=False):
        if is_evaluation:
            with torch.no_grad():
                legal_actions_mask = legal_actions_to_mask(
                    [
                        ts.observations["legal_actions"][ts.current_player()]
                        for ts in time_step
                    ],
                    self.num_actions,
                ).to(self.device)
                obs = torch.Tensor(
                    np.array(
                        [
                            np.reshape(
                                ts.observations["info_state"][ts.current_player()],
                                self.input_shape,
                            )
                            for ts in time_step
                        ]
                    )
                ).to(self.device)
                action, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )
                return [
                    StepOutput(action=a.item(), probs=p) for (a, p) in zip(action, probs)
                ]
        else:
            with torch.no_grad():
                obs = torch.Tensor(
                    np.array(
                        [
                            np.reshape(
                                ts.observations["info_state"][ts.current_player()],
                                self.input_shape,
                            )
                            for ts in time_step
                        ]
                    )
                ).to(self.device)
                legal_actions_mask = legal_actions_to_mask(
                    [
                        ts.observations["legal_actions"][ts.current_player()]
                        for ts in time_step
                    ],
                    self.num_actions,
                ).to(self.device)
                current_players = torch.Tensor(
                    [ts.current_player() for ts in time_step]
                ).to(self.device)

                action, logprob, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )

                self.legal_actions_mask[self.cur_batch_idx] = legal_actions_mask
                self.obs[self.cur_batch_idx] = obs
                self.actions[self.cur_batch_idx] = action
                self.logprobs[self.cur_batch_idx] = logprob
                self.values[self.cur_batch_idx] = value.flatten()
                self.current_players[self.cur_batch_idx] = current_players

                tsallis_ent = self._tsallis_entropy_norm(
                    probs, legal_actions_mask
                )  # [num_envs]
                self.accumulated_entropy += tsallis_ent.mean().item()
                self.entropy_count += 1

                agent_output = [
                    StepOutput(action=a.item(), probs=p) for (a, p) in zip(action, probs)
                ]
                return agent_output

    def post_step(self, reward, done):
        self.rewards[self.cur_batch_idx] = torch.tensor(reward).to(self.device).view(-1)
        self.dones[self.cur_batch_idx] = torch.tensor(done).to(self.device).view(-1)

        self.total_steps_done += self.num_envs
        self.cur_batch_idx += 1

    def learn(self, time_step):
        next_obs = torch.Tensor(
            np.array(
                [
                    np.reshape(
                        ts.observations["info_state"][ts.current_player()],
                        self.input_shape,
                    )
                    for ts in time_step
                ]
            )
        ).to(self.device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_batch)):
                    nextvalues = (
                        next_value
                        if t == self.steps_per_batch - 1
                        else self.values[t + 1]
                    )
                    nextnonterminal = 1.0 - self.dones[t]
                    delta = (
                        self.rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.steps_per_batch)):
                    next_return = (
                        next_value
                        if t == self.steps_per_batch - 1
                        else returns[t + 1]
                    )
                    nextnonterminal = 1.0 - self.dones[t]
                    returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + self.input_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_players = self.current_players.reshape(-1)
        b_legal_actions_mask = self.legal_actions_mask.reshape((-1, self.num_actions))
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        B = b_advantages.shape[0]

        # ----------------------------------------
        # Visitation logging (accumulate per window)
        # ----------------------------------------
        if self.track_visitation:
            b_obs_np = b_obs.detach().cpu().numpy()  # [B, D]
            b_players_np = b_players.detach().cpu().numpy().astype(np.int32)  # [B]
            dones_np = self.dones.reshape(-1).detach().cpu().numpy()
            alive_np = (dones_np == 0)

            for i in range(B):
                if not alive_np[i]:
                    continue

                if self.visitation_use_simhash:
                    key = self._simhash_bucket_id(b_obs_np[i])  # int bucket
                else:
                    key = self._infoset_id_from_tensor(b_obs_np[i])  # hex hash

                if b_players_np[i] == 0:
                    self._visit_counter_p0[key] += 1
                    self._visit_total_p0 += 1
                elif b_players_np[i] == 1:
                    self._visit_counter_p1[key] += 1
                    self._visit_total_p1 += 1

    

        b_playersigns = -2.0 * b_players + 1.0
        b_advantages = b_advantages * b_playersigns

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, newvalue, new_probs = self.get_action_and_value(
                    b_obs[mb_inds],
                    legal_actions_mask=b_legal_actions_mask[mb_inds],
                    action=b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                tsallis_ent = self._tsallis_entropy_norm(
                    new_probs, b_legal_actions_mask[mb_inds]
                )
                entropy_loss = tsallis_ent.mean()

                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log every log_interval
        if self.total_steps_done - self.last_log_step >= self.log_interval:
            avg_entropy = self.accumulated_entropy / max(1, self.entropy_count)

            log_data = {
                "steps": self.total_steps_done,
                "avg_entropy": avg_entropy,
            }

            # ----------------------------
            # Visitation interval summary + dump (ported)
            # ----------------------------
            if self.track_visitation:

                def _counter_stats(counter: Counter, total_visits: int):
                    uniq = len(counter)
                    if total_visits <= 0 or uniq == 0:
                        return dict(total=0, uniq=0, top10_share=0.0, entropy_norm=0.0)

                    # top10 share
                    top10 = counter.most_common(10)
                    top10_sum = sum(cnt for _, cnt in top10)
                    top10_share = top10_sum / float(total_visits)

                    # normalized entropy of visitation distribution over unique keys
                    # H(p)/log(|S|)
                    probs = np.array([c for _, c in counter.items()], dtype=np.float64)
                    probs = probs / probs.sum()
                    ent = -np.sum(probs * np.log(probs + 1e-12))
                    if uniq <= 1:
                        ent_norm = 0.0
                    else:
                        ent_norm = float(ent / np.log(uniq))

                    return dict(
                        total=int(total_visits),
                        uniq=int(uniq),
                        top10_share=float(top10_share),
                        entropy_norm=float(ent_norm),
                    )

                s0 = _counter_stats(self._visit_counter_p0, self._visit_total_p0)
                s1 = _counter_stats(self._visit_counter_p1, self._visit_total_p1)

                log_data.update(
                    {
                        "visit_p0_interval_total": s0["total"],
                        "visit_p0_interval_unique": s0["uniq"],
                        "visit_p0_interval_top10_share": s0["top10_share"],
                        "visit_p0_interval_entropy_norm": s0["entropy_norm"],
                        "visit_p1_interval_total": s1["total"],
                        "visit_p1_interval_unique": s1["uniq"],
                        "visit_p1_interval_top10_share": s1["top10_share"],
                        "visit_p1_interval_entropy_norm": s1["entropy_norm"],
                    }
                )

                # Dump interval top-K to CSV (one row per state/bucket)
                if self.visitation_csv_path is not None:
                    with open(self.visitation_csv_path, "a") as f:
                        step = int(self.total_steps_done)
                        for bucket_id, cnt in self._visit_counter_p0.most_common(
                            self.visitation_topk
                        ):
                            f.write(f"{step},0,{bucket_id},{cnt}\n")
                        for bucket_id, cnt in self._visit_counter_p1.most_common(
                            self.visitation_topk
                        ):
                            f.write(f"{step},1,{bucket_id},{cnt}\n")

                # Reset interval counters
                self._visit_counter_p0.clear()
                self._visit_counter_p1.clear()
                self._visit_total_p0 = 0
                self._visit_total_p1 = 0

            # write to train_log.csv
            log_to_csv(log_data, self.log_file)

            # Reset trackers for the next interval
            self.accumulated_entropy = 0.0
            self.entropy_count = 0
            self.last_log_step = self.total_steps_done

        self.updates_done += 1
        self.cur_batch_idx = 0

    def save(self, path):
        torch.save(self.network.actor.state_dict(), path)

    def load(self, path):
        self.network.actor.load_state_dict(torch.load(path))

    def anneal_learning_rate(self, update, num_total_updates):
        frac = max(0, 1.0 - (update / num_total_updates))
        if frac < 0:
            raise ValueError("Annealing learning rate to < 0")
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow
