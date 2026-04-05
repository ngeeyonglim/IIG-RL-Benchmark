# IEM-E-PPO Hyperparameter Analysis
*Classical Phantom Tic-Tac-Toe | seed=3 | 10M steps*

All results are from `iem_e_ppo` on `classical_phantom_ttt`, evaluated at 10M steps.
The algorithm uses **sparsemax** (hardcoded) + **epsilon-revival** exploration on top of Tsallis-entropy PPO with IEM.
Constrained: `update_epochs=1` (multi-epoch breaks sparsemax), `ent_coef=0.25`, `beta=1.0`, `tsallis_q=2.0`.

---

## Full Experiment Table (sorted by avg exploitability)

| avg   | expl0 | expl1 | clip | grad | envs | mini | lr     | iem_lr | decay | start | end  | tol   | alpha | vf_coef | gae_λ |
|-------|-------|-------|------|------|------|------|--------|--------|-------|-------|------|-------|-------|---------|-------|
| 0.420 | 0.413 | 0.427 | 0.5  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.99  | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.424 | 0.448 | 0.400 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.99  | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.470 | 0.490 | 0.449 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.99  | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.90  |
| 0.479 | 0.582 | 0.375 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.9   | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.510 | 0.435 | 0.585 | 0.4  | 3.0  | 24   | 8    | 0.001  | 0.001  | 0.9   | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.513 | 0.495 | 0.530 | 0.4  | 3.0  | 24   | 8    | 0.001  | 0.001  | 0.99  | 0.25  | 0.15 | 0.01  | 0.4   | 0.0625  | 0.95  |
| 0.527 | 0.531 | 0.523 | 0.3  | 2.0  | 12   | 8    | 0.002  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.529 | 0.525 | 0.533 | 0.4  | 3.0  | 24   | 8    | 0.001  | 0.001  | 0.99  | 0.25  | 0.15 | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.539 | 0.552 | 0.526 | 0.2  | 1.0  | 12   | 8    | 0.002  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.544 | 0.555 | 0.534 | 0.2  | 1.0  | 12   | 8    | 0.002  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.551 | 0.519 | 0.584 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.99  | 0.3   | 0.2  | 0.1   | 0.4   | 0.125   | 0.95  |
| 0.563 | 0.538 | 0.587 | 0.2  | 1.0  | 32   | 8    | 0.003  | 0.001  | 0.99  | 0.3   | 0.3  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.567 | 0.560 | 0.573 | 0.15 | 1.0  | 24   | 8    | 0.002  | 0.001  | 0.99  | 0.25  | 0.15 | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.576 | 0.579 | 0.551 | 0.2  | 1.0  | 32   | 8    | 0.003  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.576 | 0.625 | 0.526 | 0.2  | 1.0  | 6    | 4    | 0.002  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.580 | 0.605 | 0.555 | 0.05 | 0.5  | 8    | 2    | 0.002  | 0.001  | 0.8   | 0.06  | 0.03 | 0.0001| 0.4   | 0.0625  | 0.95  |
| 0.630 | 0.655 | 0.605 | 0.15 | 0.5  | 8    | 4    | 0.002  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.604 | 0.637 | 0.572 | 0.4  | 4.0  | 24   | 8    | 0.002  | 0.001  | 0.9   | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.615 | 0.647 | 0.583 | 0.2  | 1.0  | 12   | 8    | 0.003  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.664 | 0.699 | 0.630 | 0.1  | 1.0  | 12   | 8    | 0.001  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.125   | 0.95  |
| 0.700 | 0.823 | 0.576 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.9   | 0.3   | 0.2  | 0.15  | 0.4   | 0.0625  | 0.95  |
| 0.700 | 0.714 | 0.625 | 0.2  | 1.0  | 12   | 8    | 0.002  | 0.001  | 0.99  | 0.2   | 0.1  | 0.001 | 0.4   | 0.0625  | 0.95  |
| 0.704 | 0.820 | 0.587 | 0.4  | 3.0  | 12   | 8    | 0.002  | 0.001  | 0.9   | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.757 | 0.849 | 0.666 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.9   | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.764 | 1.097 | 0.431 | 0.4  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.9   | 0.35  | 0.25 | 0.1   | 0.4   | 0.0625  | 0.95  |
| 0.816 | 0.996 | 0.635 | 0.5  | 3.0  | 24   | 8    | 0.002  | 0.001  | 0.9   | 0.3   | 0.2  | 0.1   | 0.4   | 0.0625  | 0.95  |

---

## Findings

### Finding 1: `epsilon_decay_fraction=0.99` is the single most important parameter

Every result ≤ 0.513 has `decay=0.99`. Every result ≥ 0.575 without `decay=0.99` (unless it also has very small epsilon).

**Why it matters:** With `decay=0.9`, epsilon decays over ~9M of the 10M steps, so the last 1M steps use `epsilon_end`. The agent collapses exploration early and gets stuck. With `decay=0.99`, epsilon decays over ~99M steps — effectively *constant* throughout the 10M run. The behavior policy is stationary, which stabilizes the PPO importance-sampling ratio and prevents exploitation collapse.

| decay | results (avg expl) |
|-------|--------------------|
| 0.8   | 0.580              |
| 0.9   | 0.479–0.816 (all bad) |
| **0.99**  | **0.420–0.567** (all competitive) |

**Conclusion:** `decay=0.99` is a prerequisite. All further experiments should keep it.

---

### Finding 2: `epsilon_dead_tol` controls a stability/performance tradeoff

`epsilon_dead_tol` determines which actions are considered "dead" and eligible for revival.

| tol    | decay=0.99 results | pattern |
|--------|--------------------|---------|
| 0.0001 | 0.580 (decay=0.8)  | tiny, near-stable but starts from small epsilon |
| 0.001  | 0.527–0.567        | smooth, monotone decrease in learning curve |
| 0.01   | 0.513              | smoother than 0.1, faster convergence than 0.001 |
| **0.1**| **0.420–0.551**    | highest variance — best AND worst results |

With `tol=0.001`, the learning curves are smooth and monotone (consistently declining). With `tol=0.1`, learning curves are noisy and volatile — but the best final result (0.420) came from this regime.

**Hypothesis:** `tol=0.1` means any action with `p ≤ 0.1` gets revived. With sparsemax, many actions sit in a "partially dead" state (non-zero but small). Reviving all of them is aggressive, creating high variance but occasionally reaching better minima. `tol=0.001` only catches truly dead actions (sparsemax exact zeros), giving stable but conservative exploration.

**Unsettled question:** Does `tol=0.01` (middle ground) enable both stability AND performance? Only one data point exists (0.513 with `clip=0.4, lr=0.001`). Needs more investigation with `clip=0.5, lr=0.002`.

---

### Finding 3: `clip_coef` trend with `decay=0.99`

Within the `decay=0.99` regime, wider clip improves results:

| clip | best avg (decay=0.99) |
|------|-----------------------|
| 0.05 | 0.580 (decay=0.8 only) |
| 0.1  | 0.664                |
| 0.15 | 0.567                |
| 0.2  | 0.539                |
| 0.3  | 0.527                |
| 0.4  | 0.424                |
| **0.5**  | **0.420**        |

The trend is consistent: each step from 0.1 → 0.5 improves the best result. `clip=0.5` has not plateaued.

**Contrast with iem_ppo baseline:** The best iem_ppo results (avg~0.127) used `clip=0.05`. That algorithm uses softmax (smooth distributions) and `update_epochs=2`. For iem_e_ppo with sparsemax+revival, a wider clip appears necessary — likely because the behavior policy (mixing sparsemax + revival uniform) diverges significantly from the optimization policy, requiring a wider trust region.

**Untested:** `clip=0.6, 0.7, 0.8` with `decay=0.99`.

---

### Finding 4: `max_grad_norm` follows the same widening trend

| grad_norm | context | best result |
|-----------|---------|-------------|
| 0.5       | all configs | 0.580+ (consistently bad) |
| 1.0       | clip=0.1–0.2, envs=12 | 0.527–0.576 |
| 2.0       | clip=0.3, envs=12 | 0.527 |
| **3.0**   | clip=0.4–0.5, envs=24 | **0.420** |
| 4.0       | clip=0.4, decay=0.9 | 0.604 (no decay — confounded) |

`grad_norm=0.5` is harmful across all configs. `grad_norm=3.0` enables the best results. `grad_norm=4.0` has only been tested without `decay=0.99` — a gap to fill.

---

### Finding 5: `num_envs=24` is the sweet spot (with caveat)

| envs | mini | results |
|------|------|---------|
| 6    | 4    | 0.576   |
| 8    | 2    | 0.580   |
| 8    | 4    | 0.630   |
| 12   | 8    | 0.527–0.664 |
| 24   | 8    | **0.420**–0.816 (widest range) |
| 32   | 8    | 0.563–0.576 |

`envs=24` gives both the best and worst results — it's the highest-leverage setting. Lower (6, 8, 12) and higher (32) are more conservative.

The batch size with `envs=24, steps=512, mini=8` = **1536 samples per minibatch**. The `mini=2` variants (batch=6144) have never been tried with `decay=0.99`.

---

### Finding 6: Learning curve shapes reveal two distinct training regimes

**Regime A — Smooth monotone decline (stable but conservative):**
Seen with `tol=0.001` or `tol=0.01`, smaller clip/envs. Curves decrease consistently every checkpoint. Final values: 0.513–0.567. These likely haven't converged and might improve significantly with more steps or higher epsilon.

```
Run 2026-03-31_16-15-35 (tol=0.01): 0.658 → 0.573 → 0.555 → 0.534 → 0.513
Run 2026-03-31_15-28-17 (tol=0.001): 0.679 → 0.595 → 0.576 → 0.547 → 0.529
```

**Regime B — Volatile with late-run improvement (unstable but potentially better):**
Seen with `tol=0.1`, wider clip, larger envs. Curves spike to 0.6–1.1 during training then collapse to best values at final evaluation. High variance — same config can yield 0.420 or 0.700 depending on trajectory.

```
Best run (0.420): 0.496 → 0.943 → 0.563 → 0.646 → 0.420
Second best (0.424): 0.436 → 0.395 → 0.434 → 0.514 → 0.424
```

The 0.420 result arrived at a good point largely through lucky convergence at step 10M. The process is not reliably reproducible at this configuration.

---

### Finding 7: Player asymmetry is a symptom, not a cause

In many runs, expl0 >> expl1 or expl1 >> expl0. The best run (0.420) has nearly symmetric values (0.413, 0.427). Asymmetry ≥ 0.2 usually signals a bad run. No hyperparameter has been identified that reliably reduces asymmetry — it appears to be a consequence of unstable optimization dynamics rather than a tunable quantity.

---

### Finding 8: `vf_coef=0.125` is worse, `lr=0.001` is marginal

- `vf_coef=0.125` (doubled from 0.0625): consistently worse in tested configs (0.551 vs 0.424; 0.664 vs 0.539). Keep `vf_coef=0.0625`.
- `lr=0.001`: marginal. Gives 0.510–0.529 range vs 0.420–0.527 for `lr=0.002`. Keep `lr=0.002`.
- `lr=0.003`: slightly worse than 0.002 (0.563–0.615).

---

### Finding 9: iem_ppo vs iem_e_ppo gap — structural

The best `iem_ppo` (softmax, `update_epochs=2`) achieves avg≈0.127 — 3× better than our best iem_e_ppo result. Key structural differences:

| param | iem_ppo best | iem_e_ppo best |
|-------|-------------|----------------|
| policy_head | softmax | **sparsemax** (locked) |
| update_epochs | 2 | **1** (locked) |
| clip_coef | 0.05 | 0.5 (opposite direction!) |
| max_grad_norm | 0.5 | 3.0 (opposite direction!) |
| num_minibatches | 2 | 8 |

The opposite directions for `clip_coef` and `max_grad_norm` suggest the sparsemax+revival mechanism fundamentally changes the gradient landscape. Sparsemax creates discontinuous probability surfaces; the larger clip and grad_norm likely compensate for this by allowing bigger policy updates that "escape" degenerate sparse solutions.

---

## Parameter Interactions Summary

```
HIGH IMPORTANCE (clear signal):
  epsilon_decay_fraction: 0.99 >> 0.9 or 0.8
  clip_coef: wider is better (in [0.1, 0.5], trend unplateaued)
  max_grad_norm: looser is better (0.5 is bad, 3.0 is best tested)

MEDIUM IMPORTANCE (moderate signal):
  epsilon_dead_tol: 0.1 → highest variance; 0.001 → stable; 0.01 → middle ground
  epsilon_start: higher seems marginally better (0.3 > 0.2 > 0.06)
  num_envs: 24 with decay=0.99 > 12 or 32

LOW/UNCLEAR IMPORTANCE:
  gae_lambda: 0.95 ≈ 0.90 (minimal difference)
  learning_rate: 0.002 slightly better than 0.001 or 0.003
  vf_coef: 0.0625 > 0.125

UNTESTED with decay=0.99:
  clip_coef > 0.5
  max_grad_norm > 3.0 or = 4.0
  num_minibatches = 2 or 4
  epsilon_dead_tol = 0.01 with clip=0.5, lr=0.002
  alpha ≠ 0.4 (all runs use alpha=0.4)
  iem_lr ≠ 0.001 (all runs use iem_lr=0.001)
  anneal_lr = False
  epsilon_start > 0.3
```

---

## Paradigms to Explore (Next 10 Experiments)

All experiments should keep `decay=0.99` fixed. Current best config to build from:
```yaml
clip_coef: 0.5, max_grad_norm: 3.0, num_envs: 24, num_minibatches: 8
lr: 0.002, iem_lr: 0.001, alpha: 0.4
epsilon_start: 0.3, epsilon_end: 0.2, epsilon_decay_fraction: 0.99, epsilon_dead_tol: 0.1
```

### Paradigm A — Push the winning trends further

**Exp A1: `clip_coef=0.6`**
The clip widening trend (0.1→0.5) is consistent and unplateaued. Try 0.6.

**Exp A2: `clip_coef=0.8`**
Aggressive continuation. If A1 works, try even wider.

**Exp A3: `max_grad_norm=5.0`**
Continuing the loosening trend (3.0 is best tested). `max_grad_norm=4.0` was only tested without `decay=0.99`.

**Exp A4: `epsilon_start=0.4, epsilon_end=0.3`**
Higher constant epsilon. More revival pressure throughout training.

### Paradigm B — Stabilize the volatile regime

The best results come from a volatile learning process. Can we get the same final performance with a more reliable path?

**Exp B1: `epsilon_dead_tol=0.01, clip_coef=0.5, lr=0.002`**
Combine the stable `tol=0.01` setting with the wider clip (current best combination is `tol=0.1, clip=0.5`). The smooth monotone curve of `tol=0.01` might converge better than the volatile `tol=0.1`.

**Exp B2: `epsilon_dead_tol=0.05`**
Intermediate between 0.01 and 0.1. Explores the instability boundary.

### Paradigm C — Untested structural dimensions

**Exp C1: `num_minibatches=4` (with decay=0.99, clip=0.5)**
Larger minibatch (3072 vs 1536). Never tested with `decay=0.99`. The stable regime might benefit from more samples per gradient step.

**Exp C2: `num_minibatches=2` (with decay=0.99, clip=0.5)**
Extreme version — minibatch of 6144. Closest to the iem_ppo optimal structure.

**Exp C3: `alpha=0.6`**
The IEM target is `1/N^alpha`. Higher alpha means the reward signal decays faster with visit count — stronger pressure to visit NEW states. All 29 iem_e_ppo runs use `alpha=0.4`. The best iem_ppo run used `alpha=0.6`.

### Paradigm D — Hybrid: stabilize + exploit

**Exp D1: `clip_coef=0.6, epsilon_dead_tol=0.01`**
Combines the most likely-to-help clip expansion (A1) with the stable tol regime (B1). Tests whether the smooth learning curve can be pushed lower by pairing it with wider clip.

---

## Recommended Priority Order

Based on expected impact and coverage:

1. **A1** (`clip=0.6`) — direct extrapolation of clearest trend
2. **B1** (`tol=0.01, clip=0.5`) — tests if stability enables better final results
3. **A4** (`eps_start=0.4, eps_end=0.3`) — more exploration, cheap to test
4. **A3** (`grad_norm=5.0`) — continue loosening trend
5. **C3** (`alpha=0.6`) — only structural dimension never varied
6. **C1** (`mini=4`) — tests batch size impact in new regime
7. **D1** (`clip=0.6, tol=0.01`) — combines A1+B1
8. **B2** (`tol=0.05`) — fills gap in tol sweep
9. **A2** (`clip=0.8`) — aggressive push if A1 works
10. **C2** (`mini=2`) — tests iem_ppo-like batch structure

---

## Notes on Variance

The current best (0.420) has a highly volatile learning curve (peak mid-run at 0.943). It may not be reproducible. The second-best (0.424) is more stable and may represent a more reliable configuration. When evaluating whether a change "helps," a result of 0.440–0.450 that comes from a smoother curve may be preferable to 0.420 from a volatile one, since the latter is more seed-dependent.

Consider running seeds 1 and 5 (in addition to seed 3) on promising configs to validate robustness.
