# IEM-E-Tsallis: Analysis, Failure Modes, and Future Directions

*Framed as a record of what was tried, why it did not work, and where to go next.*

---

## 1. What Is IEM-E-Tsallis?

IEM-E-Tsallis (algorithm name: `iem_e_ppo`) is IEM-Tsallis extended with an **epsilon-greedy dead-action revival** mechanism. It combines three components:

1. **Sparsemax policy head** — outputs a *sparse* probability distribution where actions below a projection threshold receive exactly zero probability, unlike softmax which always assigns positive probability to every action.
2. **Tsallis entropy regularisation** (`q=2.0`) — uses the Tsallis (quadratic) entropy as the policy regulariser, which has a tighter connection to sparsemax than Shannon entropy.
3. **Epsilon-revival exploration** — with probability `ε`, replaces the sampling distribution with a uniform distribution over "dead" actions (actions where `base_prob ≤ epsilon_dead_tol`), attempting to force the agent to revisit actions it has abandoned.

### Motivation and Hypothesis

IEM-Tsallis (without epsilon) was observed to learn sparse mixed strategies quickly but sometimes lock into sub-optimal support sets — collapsing to zero the very actions that, in a Nash equilibrium, should be played with non-zero probability. The hypothesis for IEM-E-Tsallis was:

> *If we can force occasional exploration of dead actions and generate IEM intrinsic rewards along those branches, the IEM signal will be strong enough to "revive" those actions in the policy gradient, preventing premature collapse to a wrong support set.*

This is a reasonable hypothesis. It is also wrong. The following sections explain why.

---

## 2. Empirical Results

All 36 runs were evaluated on `classical_phantom_ttt` over 10M steps. Exploitability is reported as `(expl0 + expl1) / 2`, where lower is better (a value of 0 = Nash equilibrium).

### 2.1 Comparison to IEM-Tsallis

| Metric | IEM-Tsallis (199 runs) | IEM-E-Tsallis (36 runs) |
|---|---|---|
| Min exploitability | **0.1091** | 0.4197 |
| 25th percentile | **0.1447** | 0.5291 |
| Median | **0.1742** | 0.5750 |
| Mean | **0.2502** | 0.5929 |
| Stdev | 0.1997 | 0.1023 |

IEM-E-Tsallis is **~4× worse** on best-case exploitability and worse at every percentile. The epsilon-revival mechanism does not help — it actively degrades performance, and it does so consistently: the standard deviation across IEM-E-Tsallis runs is lower, meaning the algorithm is reliably bad rather than occasionally failing.

### 2.2 Dead action revival collapses to zero

The diagnostic metric `eps_dead_action_reentered_fraction` measures: of all epsilon-selected dead actions, what fraction result in that action rising above the dead threshold *after* the PPO update? This directly measures whether the exploration is effective.

| Training phase | `eps_dead_action_reentered_fraction` | `eps_states_with_dead_fraction` | Avg entropy |
|---|---|---|---|
| Early (~110k steps) | ~7–10% | ~0.85 | ~0.73 |
| Mid (~4–5M steps) | ~1–3% | ~0.97 | ~0.25 |
| Late (~9–10M steps) | **~0%** | **~0.97–0.98** | **~0.14–0.19** |

By end of training, revival succeeds essentially never. Dead actions accumulate: nearly all states (~98%) contain at least one dead action by late training. The mechanism is active — epsilon is firing, dead actions are being selected (~20% of steps late in training) — but exploration quantity is not the problem. **Exploration quality is the problem.**

### 2.3 Diagnosis: support collapse is expected, but the collapse trajectory is the problem

The 98% dead-states figure requires contextualisation. It is important to note that **support collapse is not inherently a failure** — it is what sparsemax is designed to do. Nash equilibria in phantom TTT are supported on a small subset of actions, and a converged sparsemax policy should have many dead actions. The question is not whether dead actions exist, but whether they are the *right* dead actions.

The critical diagnostic is the absolute count and trajectory of dead actions. Classical phantom TTT has 9 distinct actions; sampled uniformly across game states, the average number of legal actions per player step is approximately **6.56** (median 7). Using this, the dead action counts for the best run (`2026-04-01_05-04-39-156834_653c34`) are as follows:

| Training phase | Dead fraction (all states) | Avg dead actions / state | Avg live actions / state | % states with ≥1 dead |
|---|---|---|---|---|
| ~110k steps | 0.467 | **3.06** | 3.50 | 85.1% |
| ~221k steps | 0.661 | **4.34** | 2.22 | 97.7% |
| ~442k steps | 0.759 | **4.98** | 1.58 | 98.6% |
| ~1.1M steps | 0.794 | **5.21** | 1.35 | 98.4% |
| ~5M steps | 0.801 | **5.26** | 1.30 | 97.9% |
| ~10M steps (final) | 0.790 | **5.18** | **1.38** | 98.0% |

Three distinct phases are visible:

**Phase 1 — rapid collapse (0–440k steps):** Dead actions nearly double from 3.1 to 5.0 per state. The fraction of states with any dead action jumps from 85% to 98.6% within the first 440k steps. This is the sparsemax finding a sparse support, largely within the first 4% of training.

**Phase 2 — plateau (440k–8M steps):** The system stabilises with approximately **5.2–5.3 dead actions per state**, leaving only ~1.3 live actions on average. Conditioned on states that have at least one dead action (which is 98%+ of states), ~5.35 of 6.56 legal actions are dead. The policy has converged to a near-deterministic support of roughly 1–2 actions per state.

**Phase 3 — slight relaxation (8M–10M steps):** Dead actions decrease marginally from ~5.3 to ~5.2, and live actions recover slightly from ~1.3 to ~1.4. This is consistent with the slow learning rate annealing and the epsilon mechanism sustaining a small number of on-policy explorations of near-dead actions.

**What this means for revival:** With only ~1.4 live actions remaining per state, the epsilon mechanism is trying to resuscitate an average of 5.2 dead actions per state against a policy that has concentrated nearly all probability mass onto 1–2 actions. The gradient signal required to revive a dead action must overcome not just the sparsemax projection threshold but the full weight of ~1.4 well-trained live actions simultaneously pushing probability mass away from the dead branch. This is a near-impossible task for a single epsilon-forced trajectory.

The plateau at ~1.3–1.4 live actions is itself diagnostic: **the policy has effectively become near-deterministic at most information states.** A Nash equilibrium for phantom TTT likely requires broader support than this at many game states, meaning the algorithm has converged to a suboptimal sparse support rather than the Nash support. The epsilon mechanism cannot correct this because it cannot distinguish between the Nash-necessary dead actions and the correctly-dead dominated actions.

### 2.3 Hyperparameter sensitivity

Sorted by average exploitability across all 36 runs:

| `eps_decay` | `eps_start` | `eps_end` | `clip_coef` | `lr` | `update_epochs` | Avg expl |
|---|---|---|---|---|---|---|
| 0.99 | 0.3 | 0.2 | 0.5 | 0.002 | 1 | **0.4197** |
| 0.99 | 0.3 | 0.2 | 0.4 | 0.002 | 1 | **0.4236** |
| 0.99 | 0.3 | 0.2 | 0.5 | 0.002 | 1 | 0.4522 |
| 0.99 | 0.3 | 0.2 | 0.4 | 0.002 | 1 | 0.4697 |
| ... | | | | | | |
| 0.9 | 0.3 | 0.2 | 0.4 | 0.002 | 1 | 0.6996 |
| 0.99 | 0.3 | 0.2 | 0.6 | 0.002 | 1 | 0.7511 |
| 0.9 | 0.3 | 0.2 | 0.4 | 0.002 | 1 | 0.7571 |
| 0.9 | 0.35 | 0.25 | 0.4 | 0.002 | 1 | 0.7636 |
| 0.99 | 0.4 | 0.3 | 0.5 | 0.002 | 1 | 0.7919 |
| 0.9 | 0.3 | 0.2 | 0.5 | 0.002 | 1 | **0.8155** |

The best setting achieves 0.42; the worst achieves 0.82. Even the best IEM-E-Tsallis run is worse than the median IEM-Tsallis run (0.17).

### 2.4 Player asymmetry in poor runs

Several failing runs exhibit extreme asymmetry between the two players' exploitabilities:

| Run | expl0 | expl1 | avg |
|---|---|---|---|
| `2026-04-01_02-06-19` | 1.097 | 0.430 | 0.764 |
| `2026-03-31_23-55-35` | 0.996 | 0.635 | 0.816 |
| `2026-04-01_19-48-31` | 0.973 | 0.338 | 0.655 |
| `2026-04-01_18-11-43` | 1.002 | 0.581 | 0.792 |

In a zero-sum game, `expl0 > 1.0` means Player 0's policy can be exploited for *more than the maximum game payoff* — the policy is not just sub-optimal but incoherent. These runs are not merely converging slowly; they are converging to the wrong thing for one player entirely. The epsilon revival mechanism is introducing a training signal that one player can overfit to while the other adapts poorly.

---

## 3. Why It Does Not Work: Theoretical Analysis

### 3.1 Sparsemax is structurally incompatible with epsilon-greedy revival

The sparsemax operator projects a logit vector $z \in \mathbb{R}^K$ onto the probability simplex:
$$\text{sparsemax}(z) = \arg\min_{p \in \Delta^K} \|p - z\|_2^2$$

Unlike softmax, this projection is **piecewise linear** and creates a hard boundary: actions with logits below a threshold $\tau(z)$ receive exactly zero probability. The threshold is determined by all other active actions collectively.

The critical implication is that **the gradient of sparsemax with respect to a zeroed-out action's logit is exactly zero**. When an action $a$ has $p_a = 0$ under sparsemax, there is no gradient flowing through the policy head to the logit of $a$. The only path for $a$'s logit to increase is through the advantage-weighted policy gradient, which requires $a$ to be sampled. This is precisely what epsilon-revival attempts — but one sample is not enough.

To revive action $a$, its logit must cross the projection threshold, which depends on all other active actions. After epsilon forces a sample of $a$, the resulting advantage signal updates $a$'s logit by roughly `lr * advantage * (1 - p_a) ≈ lr * advantage`. But on the very next update step, the policy gradient for the *existing* active actions also fires — and if those actions have been well-exploited, their advantages are much larger and more frequent than the one-off epsilon sample. The net effect is that the projection threshold shifts faster than the dead logit rises.

In contrast, softmax gives $p_a > 0$ for all actions at all times and provides non-zero gradient to all logits continuously. Revival is a matter of gradual adjustment, not crossing a hard threshold. This is why vanilla IEM-Tsallis with softmax exploration (via entropy bonus) does not exhibit the same revival failure.

### 3.2 Epsilon-greedy exploration conflicts with Nash equilibrium learning in IIGs

Classic epsilon-greedy exploration is designed for **single-agent MDPs** where the exploration-exploitation dilemma is well-posed: explore suboptimal actions to reduce uncertainty, exploit the best known action to maximise reward. More exploration is generally better early, less is needed later.

In **Imperfect Information Games (IIGs)**, the objective is not reward maximisation but convergence to a Nash equilibrium (NE). A Nash equilibrium may require playing actions that appear suboptimal in isolation — specifically, actions that must be played with non-zero probability to prevent the opponent from deterministically exploiting your strategy. The *value* of an action in an IIG is not its expected reward but its *counterfactual value* — what would happen if you played it given that your opponent knows you might.

Epsilon-greedy revival is blind to this distinction. It treats dead actions as underexplored and applies exploration pressure uniformly across all dead actions. But in an IIG:
- Some dead actions are legitimately dominated (correctly zero in the NE support) and should stay dead.
- Other dead actions are part of the NE support but were killed by premature policy collapse; these need revival.

Epsilon-greedy cannot distinguish between these two cases. It forces exploration of both, which wastes the training signal on dominated actions and corrupts the on-policy learning assumption for the ones worth reviving.

### 3.3 The IEM intrinsic reward is insufficient to sustain revival

The IEM (Information Entropy Module) provides an intrinsic reward signal based on the novelty of visited information states. The reward decays as states are visited more frequently. By late training, the IEM signal has decayed to approximately 0.002 (raw, before beta scaling):

| Training phase | IEM reward (p0) | IEM reward (p1) | Total IEM signal (β=1) |
|---|---|---|---|
| ~110k steps | ~0.015 | ~0.007 | ~0.022 |
| ~5M steps | ~0.005 | ~0.005 | ~0.010 |
| ~9M steps | ~0.002 | ~0.002 | **~0.004** |

The advantage magnitudes from the game itself are typically much larger than this. When the policy gradient says "action A gives +0.5 advantage and action B gives +0.001 advantage (IEM)", the network strongly updates toward A and B drifts toward zero under sparsemax. The IEM reward is not sustained enough to counteract this — it is specifically designed to decay as states are explored, which means by the time dead branches need long-term sustenance, the IEM has nothing left to offer.

This is a fundamental design tension: IEM rewards novelty, but maintaining a non-zero-probability dead action requires *sustained* reward incentive, not novelty incentive.

### 3.4 Off-policy contamination in PPO

PPO is an **on-policy algorithm**. Its theoretical correctness relies on collecting data from the current policy, computing importance sampling ratios between the old and new policy, and clipping these ratios to prevent large updates. The clipped surrogate objective is:
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

In IEM-E-Tsallis, the data-collection policy is not $\pi_{\theta_{old}}$ but a mixture:
$$\pi_{behaviour}(a|s) = (1-\varepsilon) \cdot \text{sparsemax}(z) + \varepsilon \cdot \text{uniform}_{dead}(a|s)$$

The log-probability stored at rollout time is $\log \pi_{behaviour}(a|s)$, not $\log \pi_{sparsemax}(a|s)$. When PPO re-evaluates actions at learning time, it uses $\log \pi_{sparsemax}(a|s)$ (the current base policy). For a revival action $a$ that was dead under the old policy but is now alive, $\pi_{sparsemax}^{new}(a|s)$ might be very small but non-zero, while $\pi_{behaviour}^{old}(a|s) \approx \varepsilon / N_{dead}$ (the revival probability). The ratio $r_t$ becomes:

$$r_t = \frac{\pi_{sparsemax}^{new}(a|s)}{\pi_{behaviour}^{old}(a|s)}$$

If $\pi_{sparsemax}^{new}(a|s) \approx 0$ (still dead) and $\pi_{behaviour}^{old}(a|s) = \varepsilon/N_{dead} \approx 0.2/5 = 0.04$ (moderate revival probability), the ratio can be very small ($\approx 0$), meaning the update is heavily down-weighted by the IS correction. **The very actions we most want to revive get the smallest learning signal after IS correction.** The clip threshold is irrelevant here because the ratio is far below 1, not above.

### 3.5 Policy collapse accelerates in the zero-sum setting

In a two-player zero-sum game, both players share the same network (or closely coupled networks). When the sparsemax collapses Player 0's policy onto a small support, Player 1's best response also concentrates. This creates a feedback loop:

1. Sparsemax collapses Player 0 onto actions $\{a_1, a_2\}$.
2. Player 1 learns a best response that exploits this predictability.
3. The policy gradient for Player 0 now heavily favours $\{a_1, a_2\}$ (they have the highest advantage vs. Player 1's current policy).
4. Dead actions for Player 0 have *negative* advantages (Player 1's strategy is tuned to punish them).
5. Epsilon revival forces Player 0 to play dead actions, collecting negative advantage signals.
6. These negative advantages reinforce the sparsemax zero — the dead actions become *more dead*.

This self-reinforcing dynamic is specific to the zero-sum setting and explains the acceleration in `eps_states_with_dead_fraction` over training. The opponent's adaptation turns the epsilon exploration signal into a *counter-productive* gradient for the dead actions.

---

## 4. What Works Within IEM-E-Tsallis

Despite the above, some hyperparameter combinations perform better than others. The best results (avg exploitability ~0.42–0.50) come from:

### 4.1 Best configuration found

```yaml
epsilon_start: 0.3
epsilon_end: 0.2
epsilon_decay_fraction: 0.99
epsilon_dead_tol: 0.1
clip_coef: 0.4–0.5
learning_rate: 0.002
update_epochs: 1
num_minibatches: 8
num_envs: 24
max_grad_norm: 3.0
```

Best run: avg exploitability = **0.4197** (expl0=0.4127, expl1=0.4267) — the only run to achieve balanced convergence for both players.

### 4.2 Key factors

| Factor | Good setting | Bad setting | Reason |
|---|---|---|---|
| `epsilon_decay_fraction` | 0.99 (slow) | 0.9 (fast) | Faster decay with `lr=0.002` causes instability from early over-committing to epsilon-contaminated gradients |
| `epsilon_end` | 0.2 | 0.3+ | Higher floor sustains more off-policy noise throughout training; diminishing returns for revival |
| `clip_coef` | 0.4–0.5 | 0.6+ | High clip allows large updates on low-quality revival gradients; `clip_coef=0.6` alone gives 0.75 exploitability |
| `update_epochs` | 1 | 4 | Multiple re-uses of epsilon-contaminated data amplifies IS correction errors; all `update_epochs=4` runs are in the bottom half |
| `epsilon_start` | 0.3 | 0.4+ | Very high initial epsilon generates too much off-policy data before the policy has learned anything useful |

### 4.3 What the best configuration actually achieves

The best configuration does not revive dead actions — the `eps_dead_action_reentered_fraction` still collapses to 0% just like all other runs. What it achieves is **damage control**: the epsilon exploration interferes least with PPO's on-policy updates, allowing the base IEM-Tsallis learning to proceed roughly undisturbed. The algorithm performs well *despite* the epsilon mechanism, not *because* of it.

---

## 5. Lessons Learned

### 5.1 Exploration strategies designed for MDPs do not transfer to IIGs

Epsilon-greedy was developed for single-agent reward maximisation. The notion of "dead actions" as underexplored is appropriate in an MDP where every action has some true Q-value to be discovered. In an IIG, the value of an action is not an absolute quantity but a function of the opponent's strategy. An action that is genuinely dominated against the current opponent may be crucial in the Nash equilibrium against a best-responding opponent. Epsilon-greedy exploration cannot make this distinction.

Future exploration strategies for IIGs need to be *opponent-aware*. The question is not "have we tried this action enough?" but "is this action part of the Nash support we should be playing?"

### 5.2 Sparsemax sparsity is a feature, not a problem to be circumvented

The appeal of sparsemax for IIGs is that Nash equilibria often have small support — the optimal mixed strategy only randomises over a subset of actions. Sparsemax's hard zeros are a natural inductive bias toward finding this support. Epsilon-revival is philosophically at odds with this: it assumes sparsity is harmful and tries to undo it.

The correct framing is not "how do we keep all actions alive?" but "how do we ensure the policy converges to the *right* sparse support?" These require very different interventions.

### 5.3 The IEM intrinsic reward needs to be counterfactual-aware

The IEM rewards novelty of visited information states regardless of game-theoretic content. This is appropriate for early exploration but becomes counterproductive late in training. What is needed late in training is not novelty reward (the agent has seen most states) but a signal that specifically incentivises actions whose absence *reduces the difficulty of the opponent's task*. This is closer to counterfactual regret than to novelty-based exploration.

### 5.4 PPO is a poor foundation for mixed-policy algorithms

PPO's clipped surrogate objective is specifically designed for small on-policy deviations. The epsilon mixture creates large, structured deviations (entire action distributions shifted to dead branches) that the clip mechanism is not designed to handle. Any future work combining PPO with explicit off-policy exploration should either use an off-policy algorithm (e.g., MCTS-guided self-play, CFR-based methods) or explicitly handle the IS correction for mixture policies.

---

## 6. Future Directions

The following directions are motivated directly by the failure modes identified above. They are roughly ordered from "minimal change to existing architecture" to "fundamental algorithmic change".

### 6.1 Logit-space noise injection (minimal change)

Instead of epsilon-greedy sampling, add stochastic perturbations to the logits *before* the sparsemax projection:
$$z' = z + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2 I)$$
then apply sparsemax to $z'$. This keeps the training procedure on-policy (the noisy sparsemax is still the policy, no behaviour/target split), and the Gaussian noise can push dead-action logits above the projection threshold naturally. The variance $\sigma$ can be annealed during training.

Advantage: no off-policy correction needed. The noise acts as a smooth analogue of temperature in softmax. The policy trained under noisy sparsemax is still a valid stochastic policy for PPO.

### 6.2 Sparsemax temperature annealing

Scale the logits by a temperature $T$ before applying sparsemax:
$$\pi(a|s) = \text{sparsemax}(z / T)$$
High $T$ makes sparsemax behave like softmax (full support); low $T$ concentrates onto fewer actions. Annealing $T$ from high to low during training starts with broad exploration and gradually commits to a sparse support, similar to simulated annealing. The key question is the annealing schedule and whether the support found at low $T$ matches the NE support.

This is straightforward to implement as a modification to the existing `sparsemax` function and connects to the broader literature on entropy regularisation schedules in deep RL.

### 6.3 Adaptive IEM beta scheduling

Rather than a fixed `beta`, make the IEM scaling coefficient adapt to the current dead-action fraction:
$$\beta(t) = \beta_0 \cdot (1 + \lambda \cdot f_{dead}(t))$$
where $f_{dead}(t)$ is `eps_states_with_dead_fraction` at time $t$. When many states have dead actions, the IEM reward is boosted to counteract the sparsemax collapse. This directly addresses the finding that the IEM signal decays to 0.002 precisely when revival is most needed.

A cruder but effective variant: set `beta` to be much higher (e.g., 3–5 instead of 1.0) across the board and observe whether the stronger IEM signal can sustain broader support. This has not been tried in the current experiments.

### 6.4 Counterfactual regret-based dead-action revival

Instead of random epsilon revival, use counterfactual regret minimisation (CFR) to identify which dead actions have positive counterfactual regret — meaning playing them would have been better than the current strategy in expectation across information sets. Only revive actions with positive CFR. This requires maintaining a regret buffer per information state, which is expensive for large games but principled.

A lightweight approximation: track the advantage of dead actions at the time they were last played (via the epsilon mechanism) and only keep those with positive advantage estimates. Discard revival attempts for actions that consistently receive negative advantages — those are genuinely dominated.

### 6.5 Policy mixture with a separate "support-maintenance" policy

Maintain two policies simultaneously: a main policy $\pi_{main}$ that pursues the current best strategy, and a support policy $\pi_{support}$ that explicitly maintains non-zero probability on all legal actions. Mix them:
$$\pi_{behaviour}(a|s) = (1-\alpha) \pi_{main}(a|s) + \alpha \pi_{support}(a|s)$$
Train both with separate objectives: $\pi_{main}$ with the standard IEM-Tsallis objective, $\pi_{support}$ with a uniform entropy maximisation objective. The mixing coefficient $\alpha$ determines how much exploration breadth is maintained.

This is conceptually similar to the regularisation in PSRO (Policy Space Response Oracles) and connects to the idea of maintaining a population of strategies.

### 6.6 Replace PPO with an off-policy algorithm

The IS correction problem (Section 3.4) is fundamental to using PPO with any mixed-policy exploration. Replacing PPO with an off-policy algorithm such as **Soft Actor-Critic (SAC)** or **IMPALA** would allow arbitrary mixing policies without IS bias. SAC's entropy regularisation also provides an alternative to the sparsemax/Tsallis approach.

More specifically for IIGs, **Neural Fictitious Self-Play (NFSP)** maintains an average strategy separately from the best-response strategy, which naturally handles the exploration-vs-exploitation decomposition. Replacing the PPO backbone with an NFSP-style framework may allow the IEM signal to be used more cleanly.

### 6.7 Better dead-action detection: counterfactual value thresholding

The current `epsilon_dead_tol` threshold is applied to *probabilities*. An action is "dead" if its probability under sparsemax is below 0.1. This mixes two different things:
- Actions with low probability because they are dominated
- Actions with low probability because the sparsemax projection happened to push them to zero, even though their true counterfactual value is positive

A better dead-action criterion: compute the counterfactual value of each action periodically (using a separate value head), and classify an action as "revivable" only if its counterfactual value exceeds the current expected value minus some threshold. This requires more compute but would allow the epsilon mechanism to target only genuinely useful dead actions.

### 6.8 Multi-seed evaluation and variance reduction

All current experiments use a single seed (seed=3). The IEM-Tsallis runs also show high variance (stdev=0.20 across 199 runs), suggesting seed sensitivity is significant. Before concluding that any modification to IEM-E-Tsallis is an improvement, multi-seed evaluation (at least 3–5 seeds) is necessary to distinguish genuine improvement from lucky convergence.

---

## 7. Summary

| Aspect | Finding |
|---|---|
| **Does epsilon-revival work?** | No. Revival rate → 0% at end of all runs. Dead action fraction grows to 98%. |
| **Why not?** | Sparsemax hard zeros + opponent adaptation → revival gradient is always outcompeted. IEM reward too weak late in training. Off-policy IS correction in PPO penalises the very revival actions we want to reinforce. |
| **Best achievable result** | avg exploitability ~0.42 (best IEM-Tsallis: 0.11, ~4× better) |
| **Best config** | `eps_decay=0.99, eps_end=0.2, clip_coef=0.4–0.5, update_epochs=1` |
| **Why best config works** | It minimises damage from epsilon contamination; the algorithm converges *despite* epsilon, not because of it. |
| **Key design conflict** | Epsilon-greedy is for MDP reward maximisation; IIGs require opponent-aware support identification. |
| **Most promising fix** | Logit-space noise injection (on-policy, no IS correction needed) or adaptive IEM beta. |
| **Baseline to beat** | IEM-Tsallis: `clip_coef=0.05, num_envs=8, num_minibatches=2` → avg expl 0.109. |

The central lesson from IEM-E-Tsallis is that **bolting exploration mechanisms designed for single-agent MDPs onto game-theoretic algorithms does not work**, and does so for fundamental rather than incidental reasons. The incompatibility is between the hard-zero structure of sparsemax, the opponent-aware nature of IIG value estimation, and the on-policy requirement of PPO. Future work should address at least one of these tensions directly rather than attempting to paper over them with epsilon-greedy.
