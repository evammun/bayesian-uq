# Analytical Approximation for Dirichlet Exceedance Probability

**Status:** Implemented (2026-04-21)

## Context

The exceedance probability P(leader is the true mode) is the confidence measure for Dirichlet posteriors in GMbench's Bayesian aggregation. It's used as the stopping criterion in policy simulation and as a calibration diagnostic.

For K=2 (boolean questions), exceedance has an exact analytical solution via the regularized incomplete beta function: `P(theta_1 > 0.5) = 1 - I_{0.5}(alpha_1, alpha_2)`. For K>=3 (categorical questions, or boolean with distinct-na mode), the current implementation uses Monte Carlo: sample 10k Gamma variates and count how often the leader wins. This is fast and deterministic (fixed seed), but an analytical approximation would be cleaner — no stochastic element, potentially faster in batch, and more suitable for differentiation if needed for optimization.

## Approach: Gaussian copula with damped first-order correction

### Setup

Define Z_j = theta_1 - theta_j for each non-leader category j. The exceedance probability is P(Z_2 > 0, Z_3 > 0, ..., Z_K > 0) — the leader beats every competitor simultaneously.

**Key insight:** Each pairwise exceedance P(theta_1 > theta_j) is exact under the full Dirichlet. The ratio theta_i / (theta_i + theta_j) is marginally Beta(alpha_i, alpha_j) regardless of the other categories. So:

```
p_j = P(theta_1 > theta_j) = 1 - I_{0.5}(alpha_1, alpha_j)
```

These pairwise probabilities are not approximations — they are exact.

### Product approximation (lower bound)

If the events {theta_1 > theta_j} were independent, the exceedance would be the product of pairwise probabilities:

```
P_product = prod(p_j)
```

But the events are positively correlated under the Dirichlet: all are driven by theta_1 being large, which is a shared random quantity. By the Sidak inequality for positively dependent events:

```
prod(p_j) <= P(exceedance) <= min(p_j)
```

The product underestimates (conservative for stopping criteria), with errors of 5-15% in typical scenarios.

### First-order Gaussian copula correction

Approximate the dependence structure of the Z_j variables as multivariate Gaussian. The Taylor expansion of the multivariate normal orthant probability around independence (all correlations zero) gives a first-order correction:

```
P_1st = prod(p_j) + sum_{j<k} rho_{jk} * phi(a_j) * phi(a_k) * prod_{l != j,k}(p_l)
```

Where:
- a_j = Phi^{-1}(p_j) — probit transform of the exact pairwise exceedance
- phi(a_j) — standard normal PDF evaluated at the probit
- rho_{jk} — correlation between Z_j and Z_k under the Dirichlet

The correction adds C(K-1, 2) pair terms, each weighted by the pairwise correlation and the product of the remaining marginal probabilities.

### Dirichlet correlation structure

The correlations rho_{jk} are computed from the Dirichlet's moments. With S = sum(alpha):

```
Var(Z_j) = [alpha_1 * (S - alpha_1) + alpha_j * (S - alpha_j) + 2 * alpha_1 * alpha_j] / (S^2 * (S + 1))

Cov(Z_j, Z_k) = [alpha_1 * (S + alpha_j + alpha_k - alpha_1) - alpha_j * alpha_k] / (S^2 * (S + 1))

rho_{jk} = Cov(Z_j, Z_k) / sqrt(Var(Z_j) * Var(Z_k))
```

These are closed-form — no sampling or numerical integration.

### Non-Gaussianity and why higher-order terms don't help

The first-order correction overshoots the true Dirichlet exceedance by a K-dependent amount. This is because the Gaussian copula itself overestimates: the Dirichlet's simplex constraint (theta_1 + ... + theta_K = 1) creates harder boundaries than Gaussian tails, and the positive correlation between {Z_j > 0} events is weaker than the Gaussian predicts.

Adding more terms in the tetrachoric series (the second-order term is (rho^2/2) * a_j * a_k * phi(a_j) * phi(a_k), etc.) converges toward the exact MVN copula value — which is the wrong target. Empirically, both the 2nd-order and 3rd-order corrections perform *worse* than the 1st-order alone, because they push closer to the MVN copula value that overshoots the true Dirichlet.

The first-order truncation benefits from a lucky error cancellation: the truncation error (which would push the result higher) partially cancels the non-Gaussianity error (which pulls the true value below the copula).

### Damping factor

To account for the systematic overcorrection, apply a damping factor to the correction term:

```
P_damped = prod(p_j) + damping(K) * sum_{j<k} rho_{jk} * phi(a_j) * phi(a_k) * prod_{l != j,k}(p_l)
```

The damping is K-dependent, calibrated by minimizing squared error against Monte Carlo ground truth (200-500k samples) across 60-80+ random alpha vectors per K. It compensates for two effects:
1. Missing negative cross-pair terms in the expansion (pairs sharing an index contribute negative second-order corrections that the first-order ignores)
2. Increasing non-Gaussianity of the Dirichlet at higher K

The per-K optimal damping follows a decaying exponential that converges to an asymptote:

```
damping(K) = 0.637 + 0.206 * exp(-0.587 * (K - 3))
```

| K | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11+ |
|---|---|---|---|---|---|---|---|---|---|
| Damping | 0.843 | 0.752 | 0.701 | 0.673 | 0.657 | 0.648 | 0.643 | 0.641 | 0.637 |

The damping drops sharply from K=3 to K=8, then effectively plateaus at ~0.637. For K >= 11, the curve is flat to three decimal places. K=2 uses the exact beta formula (no correction needed). In practice K rarely exceeds 5 (boolean with distinct-na gives K=3; the largest categorical answer sets are ~5-10 options). An implementation needs only a short lookup table for K=3..10 and a constant 0.637 for K >= 11.

## Results

### Error decomposition

Systematic comparison of the product, first-order correction, higher-order corrections, and exact MVN copula CDF against Monte Carlo truth (500k samples). The "Gap" column is the true correction needed (MC - Product), and "Ratio" is how much of that gap the 1st-order captures.

| Case | MC | Product | Gap | 1st-order | Ratio | MVN copula | Non-Gaussianity |
|---|---|---|---|---|---|---|---|
| K=3 clear (5,2,2) | 0.812 | 0.793 | +0.019 | +0.023 | 1.22 | 0.831 | -0.018 |
| K=3 moderate (3,2,2) | 0.537 | 0.473 | +0.064 | +0.074 | 1.15 | 0.556 | -0.019 |
| K=3 near-prior (1.5,1.2,1) | 0.449 | 0.378 | +0.071 | +0.083 | 1.16 | 0.468 | -0.018 |
| K=3 high conc (10,5,3) | 0.898 | 0.893 | +0.006 | +0.005 | 0.91 | 0.904 | -0.005 |
| K=4 clear (5,3,2,1) | 0.707 | 0.667 | +0.040 | +0.057 | 1.42 | 0.736 | -0.028 |
| K=4 moderate (3,2,2,2) | 0.444 | 0.325 | +0.119 | +0.153 | 1.29 | 0.479 | -0.035 |
| K=4 high conc (8,4,3,2) | 0.841 | 0.822 | +0.019 | +0.023 | 1.20 | 0.860 | -0.019 |
| K=5 clear (5,3,2,1,1) | 0.699 | 0.647 | +0.052 | +0.078 | 1.50 | 0.734 | -0.036 |
| K=5 moderate (3,2,2,2,2) | 0.382 | 0.223 | +0.158 | +0.211 | 1.33 | 0.427 | -0.045 |
| K=5 tight (4,3,2.5,2,1.5) | 0.476 | 0.346 | +0.130 | +0.190 | 1.46 | 0.525 | -0.049 |

Key observations:
- The 1st-order correction overshoots by a ratio that grows from ~1.14 (K=3) to ~1.43 (K=5)
- The MVN copula (exact Gaussian dependence, exact marginals) consistently overshoots the true Dirichlet — non-Gaussianity error of 2-5%
- Adding 2nd/3rd-order Gaussian terms converges toward MVN, making errors *worse*

### Damped correction vs alternatives

Comparison across a curated test suite:

| Case | MC | Product | 1st-order | Damped | Prod err | 1st err | Damp err |
|---|---|---|---|---|---|---|---|
| K=3 clear (5,2,2) | 0.812 | 0.793 | 0.817 | 0.814 | -0.020 | +0.004 | +0.001 |
| K=3 moderate (3,2,2) | 0.537 | 0.473 | 0.546 | 0.537 | -0.064 | +0.010 | +0.001 |
| K=3 near-prior (1.5,1.2,1) | 0.449 | 0.378 | 0.461 | 0.451 | -0.071 | +0.012 | +0.001 |
| K=3 high conc (10,5,3) | 0.898 | 0.893 | 0.898 | 0.897 | -0.006 | -0.001 | -0.001 |
| K=4 clear (5,3,2,1) | 0.707 | 0.667 | 0.724 | 0.712 | -0.040 | +0.017 | +0.004 |
| K=4 moderate (3,2,2,2) | 0.444 | 0.325 | 0.478 | 0.444 | -0.119 | +0.034 | +0.001 |
| K=4 tight (3,2.5,2,1.5) | 0.431 | 0.319 | 0.466 | 0.433 | -0.112 | +0.035 | +0.002 |
| K=5 clear (5,3,2,1,1) | 0.699 | 0.647 | 0.725 | 0.702 | -0.052 | +0.026 | +0.003 |
| K=5 moderate (3,2,2,2,2) | 0.382 | 0.223 | 0.434 | 0.372 | -0.158 | +0.052 | -0.010 |
| K=5 tight (4,3,2.5,2,1.5) | 0.476 | 0.346 | 0.536 | 0.480 | -0.130 | +0.060 | +0.004 |

### Realistic trajectory (distinct-na mode, K=3)

Simulating accumulating evidence for a boolean question in distinct-na mode (TRUE, FALSE, N/A):

| Obs | Alphas | MC | Damped | Error |
|---|---|---|---|---|
| 0 | (1.0, 1.0, 1.0) | 0.333 | 0.320 | -0.013 |
| 1 | (2.0, 1.0, 1.0) | 0.613 | 0.619 | +0.007 |
| 2 | (3.0, 1.0, 1.0) | 0.789 | 0.790 | +0.001 |
| 3 | (3.0, 2.0, 1.0) | 0.639 | 0.644 | +0.005 |
| 5 | (5.0, 2.0, 1.0) | 0.871 | 0.870 | -0.001 |
| 8 | (7.0, 2.0, 1.5) | 0.950 | 0.949 | -0.001 |
| 10 | (9.0, 2.0, 1.5) | 0.984 | 0.984 | +0.000 |
| 15 | (14.0, 2.0, 1.5) | 0.999 | 0.999 | +0.000 |

After just 2-3 observations, the error is < 0.5%.

### Random sweep validation (80+ random alpha vectors per K)

Alpha vectors generated by starting from the uniform prior and adding random observations with realistic weights (direct=1, CoT=2, N/A=0.5), plus structured cases (clear leader, moderate leader).

| K | Product RMSE | Damped RMSE | Improvement | Max |err| |
|---|---|---|---|---|
| 3 | 0.040 | 0.005 | 8x | 0.016 |
| 5 | 0.105 | 0.010 | 10x | 0.023 |
| 8 | 0.156 | 0.022 | 7x | 0.054 |
| 10 | 0.170 | 0.031 | 5x | 0.066 |
| 12 | 0.193 | 0.039 | 5x | 0.098 |
| 15 | 0.209 | 0.044 | 5x | 0.099 |
| 20 | 0.214 | 0.072 | 3x | 0.129 |
| 25 | 0.220 | 0.073 | 3x | 0.123 |

The approximation is most effective for K=3..10 (5-10x improvement, max errors under 7%). For K=12-15 it remains useful (5x, max error ~10%). At K=20-25 (the practical ceiling for extreme cases) it still provides 3x improvement.

Worst cases occur at the uniform prior (all alpha = 1) where the Dirichlet is maximally non-Gaussian. At uniform prior, exceedance = 1/K — well below any stopping threshold — so the error doesn't affect decisions.

## Analysis

### Where the approximation is weakest

1. **Uniform prior (all alpha = 1).** The Dirichlet degenerates to the uniform distribution on the simplex. The Gaussian copula is a poor fit. Error: 1.3% (K=3), 3.3% (K=4), 5.4% (K=5).

2. **Very tight races at high concentration.** Many observations with no clear leader. Error: 2-3% for K=5, growing with K.

In both cases the exceedance is far below stopping thresholds (0.20-0.33 vs thresholds of 0.60-0.90), so the approximation error cannot cause an incorrect stopping decision.

### Where the approximation is strongest

1. **Leader pulling ahead.** After 3+ concordant observations, error drops below 0.5%. This is the regime where stopping decisions happen.

2. **High concentration with a clear leader.** Error < 0.1%.

3. **K=3 generally.** This is the primary use case (boolean with distinct-na mode), and the approximation is excellent across the full range.

### Comparison to MC at 10k samples

The current MC implementation uses 10k samples with fixed seed. Its inherent sampling noise is on the order of 1/sqrt(10000) ~ 1% (for probabilities near 0.5). The damped analytical approximation has comparable or lower error — it matches MC accuracy without any sampling.

### Computational considerations

The analytical approximation requires:
- K-1 calls to `scipy.special.betainc` (pairwise exceedances)
- K-1 calls to `scipy.stats.norm.ppf` and `norm.pdf` (probit and PDF)
- C(K-1, 2) correlation computations (closed-form arithmetic)

For K=3 (the primary case): 2 betainc + 2 ppf + 2 pdf + 1 correlation. Compared to the MC path: 10k Gamma samples on a (10000, K) array + argmax + mean. The analytical path is lighter per call but the MC path is already fast (~50 microseconds for K=3). The advantage is in batch mode (many posteriors at once) and in eliminating the stochastic element.

## The damping curve

The damping was calibrated per-K by minimizing total squared error against Monte Carlo ground truth across 60-80+ random alpha vectors per K, covering concentrations from 1 (prior only) to ~50 (many observations). The raw optimal damping values for K=3..50 were fitted to a three-parameter exponential:

```
damping(K) = d_inf + (d_3 - d_inf) * exp(-lambda * (K - 3))

d_inf  = 0.637    (asymptotic damping for large K)
d_3    = 0.843    (damping at K=3)
lambda = 0.587    (decay rate)
```

The fit residuals have std = 0.03 for K=3..10.

**Why the damping converges.** At larger K, most of the C(K-1, 2) pair corrections involve weak competitors with low pairwise exceedance — their phi(a_j) terms are small, so they contribute little to the total correction. The dominant contribution remains the leader-vs-runner-up pair and its immediate neighbors. The ratio of "overcorrection to true correction" stabilizes because the structure of the dominant terms doesn't change with K — additional weak competitors add negligible corrections.

**Handling K > 10.** Tested a "split" strategy: product over all K-1 competitors, correction over only the top 9. This performs worse than using all competitors in both product and correction — the per-K damping already accounts for the overcorrection, and dropping weak competitors from the correction throws away useful information. Their individual corrections are small but collectively they help. The full correction with per-K damping is the right approach for all K.

**Practical implications for implementation.** A lookup table for K=3..10 plus a constant 0.637 for K >= 11 covers all cases. The full correction uses C(K-1, 2) pair terms, which for the practical ceiling of K~10 is 36 pairs — trivial. For K=2, exceedance is exact via the beta function (no correction needed). In GMbench, K rarely exceeds 5 (boolean with distinct-na gives K=3; the largest categorical answer sets have ~5-10 options).

## Decision

Implemented as the default path for K≥3 exceedance on 2026-04-21 — batched exceedance became load-bearing for the Adaptive Stopping analysis (`simulate_adaptive_stopping` calls `exceedance_probability_batched` on a `(R, n_max, K)` trajectory tensor, which the MC path cannot service efficiently). Determinism across platforms was a secondary win.

The approximation covers the full K range analytically with a single approach: exact pairwise product + damped first-order correction over all competitors. Improvement over product: 7-11x for K=3-10, 4-5x for K=12-15, still 3x at K=20-25.

## Implementation notes

- Code lives in [`src/gmbench/bayesian.py`](../../src/gmbench/bayesian.py):
  - `_damping_factor(K)` — closed-form damping curve (constants `_DAMPING_D3 = 0.843`, `_DAMPING_D_INF = 0.637`, `_DAMPING_LAMBDA = 0.587`).
  - `_exceedance_damped_copula(alpha)` — vectorized K≥3 kernel. Input shape `(..., K)`, output shape `(...)`.
  - `exceedance_probability_batched(alpha, approx="damped")` — batched entry point. `approx="mc"` falls back to a per-row loop over the scalar MC helper (slow, for parity/debugging only).
  - `exceedance_probability(posterior, approx="damped")` — scalar entry point. Default changed from MC to damped for K≥3; K=2 still uses exact beta regardless of `approx`.
- Integration: [`simulate_adaptive_stopping`](../../src/gmbench/bayesian.py) (and through it the new [`adaptive_stopping.py`](../../src/gmbench/adaptive_stopping.py) analysis script) rely on the batched entry. Existing `aggregate.py` calls (`calibration_report`) continue to use the scalar entry and now implicitly use the damped default.

## Validation

Test suite: [`tests/test_exceedance_approximation.py`](../../tests/test_exceedance_approximation.py) covers:

- Damping lookup table (K=3..10 within 1e-3 of tabulated values; large-K plateau).
- Devlog table cases for K=3, K=4, K=5 (tolerance 0.02 vs MC ground truth).
- Batch vs per-row parity (tolerance 1e-10).
- Leading-dim broadcasting `(7, 11, K) -> (7, 11)`.
- Scalar/batched agreement (tolerance 1e-10).
- K=2 exact beta path (approx kwarg ignored).
- Random sweep RMSE bounds: K=3 ≤ 0.02, K=5 ≤ 0.04, K=8 ≤ 0.07 (loosened by ~0.01 vs devlog to absorb 10k-MC noise).
- `simulate_adaptive_stopping` correctness (K=1 edge case, P=1 determinism, high-separation early stop, uniform-pool cap rate, avg_N monotone in τ, MAP equals posterior argmax at stop).

Existing tests in `tests/test_bayesian.py` (K≥3 exceedance at tolerance abs=3e-2) continue to pass with the default switched from MC to damped, since the per-case RMSE ≤ 0.01 is well inside the existing tolerance.
