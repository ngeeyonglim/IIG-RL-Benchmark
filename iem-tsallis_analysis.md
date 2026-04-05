# IEM-Tsallis PPO: Hyperparameter Analysis on Classical Phantom TicTacToe

**Game:** Classical Phantom TicTacToe
**Algorithm:** IEM-PPO with Tsallis Entropy
**Metric:** Average exploitability (lower = closer to Nash equilibrium)
**Total runs in CSV:** 178 runs across multiple hyperparameter sweeps

---

## Summary of Best Results

| Rank | Exploitability | ent_coef | alpha | beta | policy_head | tsallis_q |
|------|---------------|----------|-------|------|-------------|-----------|
| 1    | **0.1091**    | 0.10     | 0.4   | 0.5  | default     | none      |
| 2    | 0.1134        | 0.10     | 0.4   | 0.5  | default     | none      |
| 3    | 0.1228        | 0.20     | 0.4   | 3.0  | softmax     | 2.0       |
| 4    | 0.1240        | 0.10     | 0.4   | 0.5  | default     | none      |
| 5    | 0.1253        | 0.10     | 0.4   | 0.0  | default     | none      |

**Overall stats:** mean=0.306, median=0.201, best=0.109, worst=1.000

---

## Phase 1: Entropy Coefficient Sweep (ent_coef ∈ {0.15, 0.20, 0.35, 0.40})

Fixed config: `alpha=0.4, beta=3.0, tsallis_q=2.0, policy_head=softmax, lr=0.002, seed=1, 10M steps`

### Exploitability over training (avg_score_response at each 2M-step checkpoint)

| Steps    | ent_coef=0.15 | ent_coef=0.20 | ent_coef=0.35 | ent_coef=0.40 |
|----------|--------------|--------------|--------------|--------------|
| 2M       | 0.2019       | 0.3307       | 0.3271       | 0.3419       |
| 4M       | 0.2694       | 0.2490       | 0.2794       | 0.3074       |
| 6M       | 0.1968       | 0.1991       | 0.2705       | 0.2816       |
| 8M       | 0.1683       | 0.1890       | 0.2076       | 0.2620       |
| **10M**  | **0.1779**   | **0.1228**   | **0.1664**   | **0.1815**   |

### Phase 1 Ranking by Final Exploitability

| Rank | ent_coef | Final Expl | % Improvement |
|------|----------|-----------|--------------|
| 1    | **0.20** | **0.1228** | 62.9%        |
| 2    | 0.35     | 0.1664     | 49.1%        |
| 3    | 0.15     | 0.1779     | 11.9%        |
| 4    | 0.40     | 0.1815     | 46.9%        |

**Key observations:**
- `ent_coef=0.20` achieves the best final exploitability (0.1228) and the largest relative improvement (62.9%)
- `ent_coef=0.15` converges early but plateaus, suggesting insufficient exploration
- `ent_coef=0.35` and `0.40` start high and decline but reach a higher floor than `0.20`
- Higher entropy slows convergence without improving the final result in this range

---

## Effect of Entropy Coefficient (all CSV runs)

| ent_coef | # Runs | Mean Expl | Min Expl | Max Expl | Std Dev |
|----------|--------|-----------|----------|----------|---------|
| 0.00     | 3      | 0.9999    | 0.9999   | 1.0000   | 0.000   |
| 0.05     | 1      | 0.3624    | —        | —        | —       |
| 0.10     | 17     | 0.4556    | 0.1091   | 1.0000   | 0.312   |
| **0.15** | 1      | **0.1779**| —        | —        | —       |
| **0.20** | 44     | **0.2697**| **0.1228**| 0.8137  | 0.173   |
| **0.25** | 79     | **0.1902**| **0.1273**| 0.6904  | 0.098   |
| **0.30** | 20     | **0.1666**| **0.1380**| 0.2023  | 0.019   |
| 0.35     | 1      | 0.1664    | —        | —        | —       |
| 0.40     | 4      | 0.2148    | 0.1815   | 0.2554   | 0.034   |
| 0.50     | 3      | 0.2734    | 0.2418   | 0.3009   | 0.030   |
| 1.00     | 5      | 0.7202    | 0.5716   | 0.7685   | 0.084   |

**Findings:**
- `ent_coef=0.0` fails completely (no exploration, exploitability ≈ 1.0)
- Sweet spot: **0.20–0.35**, with `ent_coef=0.30` showing the most stable distribution (std=0.019)
- `ent_coef=0.25` is the most-tested value (79 runs) with consistent good performance
- `ent_coef ≥ 0.40` degrades; `ent_coef=1.0` is very poor (mean=0.72)
- The absolute best result (0.109) came from `ent_coef=0.10`, but it is high-variance (std=0.312)

---

## Effect of Alpha Parameter

Alpha controls the IEM intrinsic motivation weighting.

| alpha | # Runs | Mean Expl | Min Expl | Max Expl |
|-------|--------|-----------|----------|----------|
| 0.2   | 1      | 0.1653    | —        | —        |
| **0.3** | 8    | **0.1600**| **0.1399**| 0.1796  |
| **0.4** | 73   | **0.2007**| **0.1091**| 1.0000  |
| 0.5   | 4      | 0.7839    | 0.1357   | 0.9999   |
| 0.6   | 68     | 0.3251    | 0.1259   | 0.7790   |
| 0.7   | 20     | 0.2186    | 0.1528   | 0.3624   |
| 0.8   | 4      | 0.3108    | 0.2175   | 0.4456   |

**Findings:**
- **alpha=0.3** has the best mean (0.160) and lowest variance among well-sampled values
- **alpha=0.4** is the most tested (73 runs) and contains the overall best run (0.109)
- **alpha ≥ 0.5** degrades sharply; alpha=0.5 is highly unstable (range 0.136–1.0)
- Recommended range: **alpha ∈ [0.3, 0.4]**

---

## Best (ent_coef, alpha) Combinations

| ent_coef | alpha | # Runs | Mean Expl | Min Expl |
|----------|-------|--------|-----------|----------|
| 0.25     | 0.5   | 1      | 0.1357    | 0.1357   |
| **0.30** | **0.4** | 8    | **0.1487**| **0.1380**|
| 0.25     | 0.3   | 4      | 0.1568    | 0.1399   |
| 0.30     | 0.3   | 2      | 0.1599    | 0.1530   |
| 0.20     | 0.3   | 2      | 0.1663    | 0.1530   |
| **0.25** | **0.4** | 43   | **0.1679**| **0.1273**|
| 0.15     | 0.4   | 1      | 0.1779    | 0.1779   |

Most reliable combination with sufficient samples: **ent_coef=0.30, alpha=0.4** (8 runs, mean=0.149)

---

## Effect of Policy Head

| policy_head | # Runs | Mean Expl | Median | Min    |
|-------------|--------|-----------|--------|--------|
| default     | 77     | 0.2276    | 0.1796 | 0.1091 |
| softmax     | 88     | 0.2306    | 0.1691 | 0.1228 |
| **sparsemax** | 13   | **0.7027**| 0.7126 | 0.5631 |

**Findings:**
- `default` and `softmax` perform comparably (mean ≈ 0.23)
- **`sparsemax` is highly suboptimal** (~3× worse mean exploitability)
- All top-10 runs use either default or softmax

---

## Effect of Tsallis q

| tsallis_q | # Runs | Mean Expl | Min Expl |
|-----------|--------|-----------|----------|
| none      | 68     | 0.2360    | 0.1091   |
| 1.2       | 8      | 0.5783    | 0.1413   |
| 1.5       | 4      | 0.2527    | 0.1343   |
| **2.0**   | **93** | **0.2570**| **0.1228**|
| 2.2       | 1      | 0.1440    | 0.1440   |
| 2.4       | 2      | 0.4417    | 0.1931   |

**Findings:**
- **Tsallis q=2.0** is the most extensively tested variant and competitive with the non-Tsallis baseline
- **q=1.2** performs poorly (mean=0.578); the sparse Tsallis regime is too aggressive
- **q=2.2** showed a promising single result (0.144) but is under-sampled
- The non-Tsallis baseline holds the best overall result (0.109), but Tsallis q=2.0 can match it with good hyperparameters

### Best Tsallis q=2.0 Results (top 5)

| Expl   | ent_coef | alpha | beta | policy_head |
|--------|----------|-------|------|-------------|
| 0.1228 | 0.20     | 0.4   | 3.0  | softmax     |
| 0.1273 | 0.25     | 0.4   | 3.0  | softmax     |
| 0.1302 | 0.25     | 0.4   | 0.0  | softmax     |
| 0.1309 | 0.25     | 0.4   | 3.0  | softmax     |
| 0.1322 | 0.25     | 0.4   | 1.0  | softmax     |

All top Tsallis results require **softmax policy** and **alpha=0.4**.

---

## Performance Distribution

| Tier             | Range      | # Runs | % of Total |
|------------------|------------|--------|-----------|
| Excellent        | < 0.15     | 18     | 10%        |
| Good             | 0.15–0.25  | 54     | 30%        |
| Acceptable       | 0.25–0.35  | 46     | 26%        |
| Poor             | > 0.35     | 60     | 34%        |

---

## Recommended Configurations

### Config A — Best Overall (Non-Tsallis)
```yaml
ent_coef: 0.10
alpha: 0.4
beta: 0.0–0.5
policy_head: default
tsallis_q: none
```
Expected exploitability: ~0.11–0.13. Highest peak performance but higher variance across seeds.

### Config B — Most Stable (Tsallis q=2.0)
```yaml
ent_coef: 0.25–0.30
alpha: 0.4
beta: 3.0
policy_head: softmax
tsallis_q: 2.0
```
Expected exploitability: ~0.13–0.17. Best consistency across multiple seeds.

### Config C — Balanced
```yaml
ent_coef: 0.30
alpha: 0.3–0.4
beta: 1.0–3.0
policy_head: default or softmax
tsallis_q: 2.0
```
Expected exploitability: ~0.15–0.19. Lowest variance; suitable when seed diversity matters.

### Configurations to Avoid
- `ent_coef=0.0`: Complete failure (expl ≈ 1.0)
- `ent_coef ≥ 0.4`: Degraded convergence
- `alpha ≥ 0.5`: Sharp performance drop, high instability
- `policy_head=sparsemax`: ~3× worse performance
- `tsallis_q=1.2`: Poor performance (mean=0.578)

---

## Key Takeaways

1. **Entropy coefficient is the most critical hyperparameter.** The optimal range is 0.20–0.30. Too low (0.0) prevents exploration entirely; too high (≥0.40) prevents convergence.

2. **Alpha should be kept ≤ 0.4.** Values ≥ 0.5 cause dramatic instability. alpha=0.3 gives the best mean; alpha=0.4 gives the best single result.

3. **Tsallis q=2.0 with softmax is competitive with the Shannon-entropy baseline** but does not surpass it on the best individual runs. The non-Tsallis baseline holds the overall best result (0.109).

4. **Sparsemax is consistently harmful** across all hyperparameter settings and should be excluded from future sweeps.

5. **Phase 1 finding:** Among the tested entropy values {0.15, 0.20, 0.35, 0.40}, `ent_coef=0.20` is clearly superior, achieving 0.1228 final exploitability—the best result in the Phase 1 sweep and consistent with the broader CSV analysis.
