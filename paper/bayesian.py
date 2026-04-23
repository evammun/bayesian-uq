"""Bayesian inference library for GMbench.

Pure math module with no I/O.  Computes posteriors from weighted observations,
resolves point estimates, measures posterior confidence, and supports subsampling
over collected data.

Consumed by analysis scripts operating offline over JSONL records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.special
import scipy.stats

from .types import SampleType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRICHLET_ALPHA_0: float = 1.0
NIG_KAPPA_0: float = 0.0
NIG_MU_0: float = 0.5
NIG_ALPHA_0: float = 0.5
NIG_BETA_0: float = 1 / 24  # Var(Uniform[0,1]) * alpha_0 = (1/12) * 0.5

DEFAULT_NA_WEIGHT: float = 0.5
DEFAULT_INTERVAL_L: float = 0.25  # range / 4


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalSpec:
    """Specification for a categorical (Dirichlet) problem."""

    categories: list[str]


@dataclass(frozen=True)
class NumericSpec:
    """Specification for a numeric (NIG) problem."""

    range_min: float
    range_max: float
    answer_options: dict[str, float]


ProblemSpec = CategoricalSpec | NumericSpec


@dataclass(frozen=True)
class WeightConfig:
    """Parameterizable observation weight configuration.

    Args:
        na_weight: Base weight for N/A observations.
    """

    na_weight: float = DEFAULT_NA_WEIGHT


@dataclass(frozen=True)
class DirichletPosterior:
    """Dirichlet posterior over K categories.

    Args:
        alpha: Mapping from category name to alpha parameter.
    """

    # WARNING: dict contents are mutable despite frozen=True.
    # Do not mutate alpha in place -- always create a new dict via dirichlet_update.
    alpha: dict[str, float]


@dataclass(frozen=True)
class NIGPosterior:
    """Normal-Inverse-Gamma posterior for numeric variables.

    Args:
        kappa: Pseudo-count controlling prior influence on mean.
        mu: Posterior mean (in normalized [0,1] space).
        alpha: Shape parameter (increases by w/2 per observation).
        beta: Rate parameter (accumulates variance information).
    """

    kappa: float
    mu: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class Sample:
    """A single observation for posterior updating.

    Args:
        value: Category label (str) for Dirichlet, or numeric value (float) for NIG.
        is_na: Whether the model answered N/A (reduces weight).
        sample_type: DIRECT or COT.
    """

    value: str | float
    is_na: bool
    sample_type: SampleType


# ---------------------------------------------------------------------------
# Dirichlet distribution
# ---------------------------------------------------------------------------


def dirichlet_prior(categories: list[str]) -> DirichletPosterior:
    """Create a uniform Dirichlet prior over the given categories.

    Args:
        categories: Category names (e.g. ["TRUE", "FALSE"] for boolean).

    Returns:
        DirichletPosterior with alpha_0 = 1.0 per category.
    """
    return DirichletPosterior(alpha={cat: DIRICHLET_ALPHA_0 for cat in categories})


def dirichlet_update(
    posterior: DirichletPosterior, category: str, weight: float,
) -> DirichletPosterior:
    """Update a Dirichlet posterior by adding weight to a category.

    Args:
        posterior: Current posterior.
        category: Observed category.
        weight: Effective weight to add.

    Returns:
        New DirichletPosterior with updated alpha for the category.
    """
    new_alpha = dict(posterior.alpha)
    new_alpha[category] = new_alpha[category] + weight
    return DirichletPosterior(alpha=new_alpha)


def dirichlet_map(posterior: DirichletPosterior) -> str:
    """Return the MAP (maximum a-posteriori) category.

    Ties are broken by dict iteration order.

    Args:
        posterior: Dirichlet posterior.

    Returns:
        Category name with the highest alpha.
    """
    return max(posterior.alpha, key=lambda k: posterior.alpha[k])


# ---------------------------------------------------------------------------
# Dirichlet MLE (Minka's fixed-point iteration)
# ---------------------------------------------------------------------------


def _trigamma(x: np.ndarray) -> np.ndarray:
    """Fast trigamma ψ₁(x) via recurrence + asymptotic series.

    Avoids the overhead of scipy.special.polygamma(1, x) which routes
    through the general-purpose zeta function.  Pure numpy operations.

    Uses the recurrence ψ₁(x) = ψ₁(x+1) + 1/x² to shift x ≥ 7,
    then the asymptotic expansion (Abramowitz & Stegun 6.4.12).
    Max 7 fixed shifts (x is clamped ≥ 1e-8 by callers).
    """
    x = np.array(x, dtype=np.float64)
    result = np.zeros_like(x)
    # Unrolled recurrence: 7 shifts guarantees x ≥ 7 for any x ≥ 0.01
    for _ in range(7):
        small = x < 7.0
        result += np.where(small, 1.0 / (x * x), 0.0)
        x = np.where(small, x + 1.0, x)
    # Asymptotic series: 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - ...
    ix = 1.0 / x
    ix2 = ix * ix
    result += ix + ix2 * (0.5 + ix * (1.0 / 6.0 + ix2 * (-1.0 / 30.0 + ix2 * (1.0 / 42.0))))
    return result


def _inverse_digamma(y: np.ndarray, max_iter: int = 8) -> np.ndarray:
    """Vectorized inverse digamma via Newton's method.

    Computes x such that digamma(x) = y for each element of y.
    Operates element-wise on arrays of any shape.

    Args:
        y: Target digamma values (any shape).
        max_iter: Maximum Newton iterations (8 is sufficient for 1e-8 precision
            with good initialization).

    Returns:
        Array of same shape with x satisfying digamma(x) ≈ y.
    """
    # Initial estimate: Eq. 24 from Minka (2003)
    # For y >= -2.22: x = exp(y) + 0.5
    # For y < -2.22:  x = -1 / (y - digamma(1))
    psi_1 = float(scipy.special.digamma(1))
    denom = y - psi_1
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    x = np.where(
        y >= -2.22,
        np.exp(y) + 0.5,
        -1.0 / denom,
    )
    x = np.maximum(x, 1e-8)

    for _ in range(max_iter):
        dx = (scipy.special.digamma(x) - y) / _trigamma(x)
        x = np.maximum(x - dx, 1e-8)

    return x


def _minka_iteration(
    suff_stats: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> np.ndarray:
    """Core Minka fixed-point iteration for Dirichlet MLE.

    Operates on precomputed sufficient statistics s_k = mean(log(p_k)).
    When suff_stats is (M, K), fits M independent Dirichlets in parallel.

    Args:
        suff_stats: (K,) or (M, K) array of sufficient statistics.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on max |alpha_new - alpha_old|.

    Returns:
        Fitted alpha array, same shape as input.
    """
    is_1d = suff_stats.ndim == 1
    if is_1d:
        suff_stats = suff_stats[np.newaxis, :]  # (1, K)

    K = suff_stats.shape[1]

    M = suff_stats.shape[0]

    # Initialize via Minka's method: the MLE satisfies ψ(α_k) = s_k + ψ(Σα).
    # Seed with Σα ≈ K, using ln(K) - 1/(2K) as a cheap proxy for ψ(K).
    psi_K_approx = math.log(K) - 0.5 / K
    alpha = np.maximum(_inverse_digamma(suff_stats + psi_K_approx), 0.1)  # (M, K)

    # Per-row convergence masking: stop processing rows once they converge.
    # Some rows (especially K=2 with extreme observations) oscillate and
    # never converge — masking prevents them from forcing full iteration
    # on the entire batch.
    active = np.arange(M)  # indices of unconverged rows

    for _ in range(max_iter):
        if len(active) == 0:
            break

        a = alpha[active]
        a_sum = a.sum(axis=1, keepdims=True)
        a_new = _inverse_digamma(
            scipy.special.digamma(a_sum) + suff_stats[active]
        )
        a_new = np.maximum(a_new, 1e-8)

        diffs = np.max(np.abs(a_new - a), axis=1)
        alpha[active] = a_new
        active = active[diffs >= tol]

    if is_1d:
        return alpha[0]  # (K,)
    return alpha


def dirichlet_mle(
    observations: list[dict[str, float]],
    prior_strength: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> DirichletPosterior:
    """Fit a Dirichlet distribution to observed probability vectors via MLE.

    Uses Minka's fixed-point iteration with a uniform pseudo-observation
    for regularization.

    Args:
        observations: List of probability dicts (label → probability).
            All dicts must share the same key set.
        prior_strength: Weight of the uniform pseudo-observation.
            Acts as regularization — stabilizes low-N fits.
        max_iter: Maximum Minka iterations.
        tol: Convergence tolerance.

    Returns:
        DirichletPosterior with fitted alpha parameters.
    """
    if not observations:
        raise ValueError("observations must be non-empty")

    categories = list(observations[0].keys())
    K = len(categories)

    if K == 0:
        raise ValueError("observations must have at least one category")

    # Convert dicts to (N, K) array
    obs_array = np.array(
        [[obs[c] for c in categories] for obs in observations],
        dtype=np.float64,
    )

    # Label smoothing: mix with uniform to prevent log(0) in MLE.
    # p_smooth = eps*(1/K) + (1-eps)*p, ensuring min prob = eps/K.
    eps = 1e-3
    obs_array = eps * (1.0 / K) + (1.0 - eps) * obs_array
    log_obs = np.log(obs_array)  # (N, K)

    N = len(observations)

    if prior_strength > 0:
        pseudo_log = np.full(K, np.log(1.0 / K))  # (K,)
        suff_stats = (log_obs.sum(axis=0) + prior_strength * pseudo_log) / (N + prior_strength)
    else:
        suff_stats = log_obs.mean(axis=0)  # (K,)

    alpha = _minka_iteration(suff_stats, max_iter=max_iter, tol=tol)

    return DirichletPosterior(alpha=dict(zip(categories, alpha.tolist())))


def dirichlet_sum(
    observations: list[dict[str, float]],
    prior_strength: float = 1.0,
    multiplier: float = 1.0,
) -> DirichletPosterior:
    """Fit a Dirichlet via weighted summation of probability vectors.

    Accumulates probability vectors as fractional pseudo-counts:
        alpha_k = prior_strength * DIRICHLET_ALPHA_0 + multiplier * sum(p_{ik})

    Each observation distributes `multiplier` total pseudo-counts proportionally
    to its probability vector.  No iterative solver — pure linear accumulation.

    Args:
        observations: List of probability dicts (label → probability).
            All dicts must share the same key set.
        prior_strength: Scales the uniform prior (default 1.0).
        multiplier: Pseudo-counts per observation (default 1.0).

    Returns:
        DirichletPosterior with accumulated alpha parameters.
    """
    if not observations:
        raise ValueError("observations must be non-empty")

    categories = list(observations[0].keys())
    K = len(categories)

    if K == 0:
        raise ValueError("observations must have at least one category")

    # Start from uniform prior scaled by prior_strength
    alpha = {cat: prior_strength * DIRICHLET_ALPHA_0 for cat in categories}

    # Accumulate: each observation distributes `multiplier` pseudo-counts
    for obs in observations:
        for cat in categories:
            alpha[cat] += multiplier * obs[cat]

    return DirichletPosterior(alpha=alpha)


# ---------------------------------------------------------------------------
# NIG (Normal-Inverse-Gamma) distribution
# ---------------------------------------------------------------------------


def nig_prior() -> NIGPosterior:
    """Create the default NIG prior.

    Returns:
        NIGPosterior with kappa=0, mu=0.5, alpha=0.5, beta=1/24.
    """
    return NIGPosterior(
        kappa=NIG_KAPPA_0, mu=NIG_MU_0, alpha=NIG_ALPHA_0, beta=NIG_BETA_0,
    )


def nig_update(
    posterior: NIGPosterior, x: float, weight: float,
) -> NIGPosterior:
    """Update a NIG posterior with a new observation.

    Args:
        posterior: Current posterior.
        x: Observed value (in normalized [0,1] space).
        weight: Effective weight for this observation.

    Returns:
        New NIGPosterior with updated parameters.
    """
    kappa, mu, alpha, beta = posterior.kappa, posterior.mu, posterior.alpha, posterior.beta

    kappa_new = kappa + weight
    mu_new = (kappa * mu + weight * x) / kappa_new
    alpha_new = alpha + weight / 2.0

    # When kappa == 0, there is no prior influence on mean, so the shift term is 0.
    if kappa == 0.0:
        beta_new = beta
    else:
        beta_new = beta + (kappa * weight * (x - mu) ** 2) / (2.0 * kappa_new)

    return NIGPosterior(kappa=kappa_new, mu=mu_new, alpha=alpha_new, beta=beta_new)


def nig_mean(posterior: NIGPosterior) -> float:
    """Return the posterior mean.

    Args:
        posterior: NIG posterior.

    Returns:
        Posterior mean mu.
    """
    return posterior.mu


def normalize_value(value: float, range_min: float, range_max: float) -> float:
    """Normalize a value to [0, 1] given a range.

    Args:
        value: Raw value.
        range_min: Minimum of the range.
        range_max: Maximum of the range.

    Returns:
        Normalized value in [0, 1].  Returns 0.5 if range is degenerate.
    """
    if range_min == range_max:
        return 0.5
    return (value - range_min) / (range_max - range_min)


def denormalize_value(normalized: float, range_min: float, range_max: float) -> float:
    """Denormalize a [0, 1] value back to the original range.

    Args:
        normalized: Value in [0, 1].
        range_min: Minimum of the range.
        range_max: Maximum of the range.

    Returns:
        Value in [range_min, range_max].  Returns range_min if range is degenerate.
    """
    if range_min == range_max:
        return range_min
    return range_min + normalized * (range_max - range_min)


# ---------------------------------------------------------------------------
# Observation processing
# ---------------------------------------------------------------------------


def compute_weight(
    is_na: bool, config: WeightConfig,
) -> float:
    """Compute the effective weight for an observation.

    Weight = config.na_weight if N/A, else 1.0.  All sample types get
    weight 1.0 (the former CoT multiplier was removed with the sample
    type redesign).

    Args:
        is_na: Whether the model answered N/A.
        config: Weight configuration.

    Returns:
        Effective observation weight.
    """
    return config.na_weight if is_na else 1.0


def process_observation(
    posterior: DirichletPosterior | NIGPosterior,
    value: str | float,
    is_na: bool,
    config: WeightConfig,
) -> DirichletPosterior | NIGPosterior:
    """Update a posterior with a single observation.

    Routes to dirichlet_update or nig_update based on posterior type.

    Args:
        posterior: Current Dirichlet or NIG posterior.
        value: Category label (str for Dirichlet) or normalized float (NIG).
        is_na: Whether the model answered N/A.
        config: Weight configuration.

    Returns:
        Updated posterior (new instance).
    """
    w = compute_weight(is_na, config)
    if isinstance(posterior, DirichletPosterior):
        if not isinstance(value, str):
            raise TypeError(f"Dirichlet requires str value, got {type(value).__name__}")
        return dirichlet_update(posterior, value, w)
    else:
        if not isinstance(value, (int, float)):
            raise TypeError(f"NIG requires numeric value, got {type(value).__name__}")
        return nig_update(posterior, float(value), w)


def process_split_observation(
    posterior: DirichletPosterior | NIGPosterior,
    splits: list[tuple[str | float, float]],
) -> DirichletPosterior | NIGPosterior:
    """Update a posterior with a split observation.

    Each (value, sub_weight) pair is processed as a separate update.
    All sample types get weight 1.0 (no type-based multiplier).

    Args:
        posterior: Current Dirichlet or NIG posterior.
        splits: List of (value, sub_weight) pairs.

    Returns:
        Updated posterior (new instance).
    """
    for value, sub_weight in splits:
        if isinstance(posterior, DirichletPosterior):
            if not isinstance(value, str):
                raise TypeError(f"Dirichlet requires str value, got {type(value).__name__}")
            posterior = dirichlet_update(posterior, value, sub_weight)
        else:
            if not isinstance(value, (int, float)):
                raise TypeError(f"NIG requires numeric value, got {type(value).__name__}")
            posterior = nig_update(posterior, float(value), sub_weight)
    return posterior


# ---------------------------------------------------------------------------
# Exceedance probability (Dirichlet confidence)
# ---------------------------------------------------------------------------


def exceedance_probability(
    posterior: DirichletPosterior,
    approx: Literal["damped", "mc"] = "damped",
) -> float:
    """Compute the exceedance probability for the leading category.

    P(the leading category is the true mode) -- the probability that the MAP
    category would be selected if we sampled from the Dirichlet posterior
    infinitely many times.

    K=2: exact analytical solution via the regularized incomplete beta function
    (``approx`` ignored).
    K>=3: damped Gaussian-copula approximation by default (deterministic,
    closed-form).  Pass ``approx="mc"`` to fall back to the Monte Carlo path.

    Args:
        posterior: Dirichlet posterior.
        approx: Approximation method for K>=3 (``"damped"`` or ``"mc"``).

    Returns:
        Exceedance probability in [0, 1].
    """
    alpha = posterior.alpha
    K = len(alpha)

    if K < 2:
        return 1.0

    if K == 2:
        values = list(alpha.values())
        a, b = max(values), min(values)
        return _exceedance_k2(a, b)

    if approx == "mc":
        return _exceedance_monte_carlo(alpha)

    # Default: damped Gaussian-copula approximation
    alpha_vec = np.asarray(list(alpha.values()), dtype=np.float64)
    return float(_exceedance_damped_copula(alpha_vec[None, :]).item())


def _exceedance_k2(a: float, b: float) -> float:
    """Analytical exceedance probability for K=2 (binary).

    Uses the regularized incomplete beta function:
        P(theta_1 > 0.5) = 1 - I_{0.5}(a, b)

    NOTE: scipy.special.betainc takes (a, b, x), not (x, a, b).

    Args:
        a: Alpha for the leading category.
        b: Alpha for the other category.

    Returns:
        Exceedance probability.
    """
    return float(1.0 - scipy.special.betainc(a, b, 0.5))


def _exceedance_monte_carlo(
    alpha: dict[str, float],
    n_samples: int = 10_000,
) -> float:
    """Monte Carlo exceedance using the actual posterior alphas.

    Samples K independent Gamma(alpha_i, 1) variates (unnormalized Dirichlet)
    and checks if the leader's sample exceeds all others.  Deterministic
    (fixed seed per call) -- same alpha always gives same result.

    Args:
        alpha: Mapping from category name to alpha parameter.
        n_samples: Number of Monte Carlo iterations.

    Returns:
        Estimated exceedance probability.
    """
    rng = np.random.default_rng(42)
    alpha_vec = np.array(list(alpha.values()))
    # Sample K independent Gamma(alpha_i, 1) -- unnormalized Dirichlet
    gammas = rng.gamma(alpha_vec, 1.0, size=(n_samples, len(alpha_vec)))
    # Leader is the category with max alpha (MAP)
    leader_idx = int(np.argmax(alpha_vec))
    leader_samples = gammas[:, leader_idx]
    # Max of all non-leader samples
    other_samples = np.delete(gammas, leader_idx, axis=1)
    max_others = other_samples.max(axis=1)
    return float(np.mean(leader_samples > max_others))


# ---------------------------------------------------------------------------
# Damped Gaussian-copula exceedance approximation (K>=3)
# ---------------------------------------------------------------------------
#
# Analytical, deterministic, vectorizable approximation to
# P(leader > all competitors) under a Dirichlet posterior.  Replaces the
# Monte Carlo path as the default for K>=3 in both the scalar and batched
# entry points.  See docs/dev/2026-04-01-exceedance-analytical-approximation.md
# for derivation and validation.
#
# Typical accuracy (RMSE vs MC 500k):
#   K=3: 0.005    K=5: 0.010   K=8: 0.022   K=10: 0.031
#   K=12: 0.039   K=15: 0.044  K=20: 0.072  K=25: 0.073
#
# Worst cases occur at the uniform prior, where exceedance is 1/K far below
# any stopping threshold -- the approximation error cannot flip a decision.

# Damping curve constants (fit to MC ground truth across K=3..50).
_DAMPING_D3: float = 0.843       # damping(K=3)
_DAMPING_D_INF: float = 0.637    # asymptotic damping for large K
_DAMPING_LAMBDA: float = 0.587   # decay rate


def _damping_factor(K: int) -> float:
    """K-dependent damping factor for the first-order copula correction.

    damping(K) = d_inf + (d_3 - d_inf) * exp(-lambda * (K - 3))

    Tabulated:
      K=3  -> 0.843
      K=4  -> 0.752
      K=5  -> 0.701
      K=6  -> 0.673
      K=7  -> 0.657
      K=8  -> 0.648
      K=9  -> 0.643
      K=10 -> 0.641
      K>=11 plateaus at ~0.637.
    """
    if K <= 2:
        return 0.0  # not used (K=2 handled exactly)
    return _DAMPING_D_INF + (_DAMPING_D3 - _DAMPING_D_INF) * math.exp(
        -_DAMPING_LAMBDA * (K - 3),
    )


def _exceedance_damped_copula(alpha: np.ndarray) -> np.ndarray:
    """Damped Gaussian-copula exceedance approximation for K>=3.

    Vectorized over leading dimensions.  For K<3 returns an array of ones
    (single-category degenerate case).

    Algorithm:
      1. Identify the leader (argmax of alpha) per row and gather its alpha.
      2. Compute pairwise exceedance p_j = P(theta_1 > theta_j) via the
         exact marginal Beta: p_j = 1 - I_{0.5}(alpha_1, alpha_j).
      3. Compute probit transforms a_j = Phi^-1(p_j) and densities phi_j.
      4. Compute Dirichlet variances Var(Z_j) and covariances Cov(Z_j, Z_k),
         where Z_j = theta_1 - theta_j, and derive the correlation rho_{jk}.
      5. Assemble the product approximation prod(p_j) plus a damped sum of
         pair corrections rho_{jk} * phi_j * phi_k * prod_{l!=j,k}(p_l).

    Args:
        alpha: Array of shape (..., K) with non-negative Dirichlet alpha
            parameters.  K must be >= 2 to produce a meaningful result; K=2
            also works and returns the exact beta value.

    Returns:
        Array of shape (...) with exceedance probabilities in [0, 1].
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0 or alpha.shape[-1] < 1:
        raise ValueError("alpha must have at least one trailing dimension of size >= 1")

    K = alpha.shape[-1]
    leading_shape = alpha.shape[:-1]

    if K == 1:
        return np.ones(leading_shape, dtype=np.float64)

    # Defensive clamp: scipy.special.betainc(a, b, x) returns NaN when a<=0 or
    # b<=0, so callers that pass raw posteriors with zero / near-zero alphas
    # (e.g. a malformed prior, or direct use of this helper without the
    # Dirichlet-prior floor) would otherwise propagate NaN through the whole
    # computation.  1e-12 keeps betainc / log numerically well-behaved while
    # being far below any realistic alpha (the Dirichlet prior ALPHA_0 = 1.0).
    alpha = np.maximum(alpha, 1e-12)

    # Flatten leading dims to (M, K) for the main computation
    flat = alpha.reshape(-1, K)
    M = flat.shape[0]

    # Leader per row (argmax of alpha)
    leader_idx = np.argmax(flat, axis=-1)  # (M,)
    row_idx = np.arange(M)
    alpha_lead = flat[row_idx, leader_idx]  # (M,)

    # Competitors: everything except the leader column.
    # Build a (M, K) bool mask and select -> (M, K-1).
    col_idx = np.arange(K)
    competitor_mask = col_idx[None, :] != leader_idx[:, None]  # (M, K)
    alpha_comp = flat[competitor_mask].reshape(M, K - 1)  # (M, K-1)

    # K=2 special case: exact beta (approx ignored).  Vectorized.
    if K == 2:
        a = alpha_lead
        b = alpha_comp[:, 0]
        # betainc(a, b, 0.5) with a >= b (by argmax selection)
        out = 1.0 - scipy.special.betainc(a, b, 0.5)
        return out.reshape(leading_shape).astype(np.float64)

    # --- Step 1: pairwise exceedance p_j ------------------------------------
    # p_j = P(theta_lead > theta_j) = 1 - I_{0.5}(alpha_lead, alpha_j).
    # With alpha_lead >= alpha_j (by argmax), p_j is in [0.5, 1].
    p_j = 1.0 - scipy.special.betainc(
        alpha_lead[:, None], alpha_comp, 0.5,
    )  # (M, K-1)
    # Numerical safety: clip to an open interval so ppf stays finite.
    p_j = np.clip(p_j, 1e-15, 1.0 - 1e-15)

    # --- Step 2: probit transform and density -------------------------------
    a_j = scipy.stats.norm.ppf(p_j)        # (M, K-1)
    phi_j = scipy.stats.norm.pdf(a_j)      # (M, K-1)

    # --- Step 3: Dirichlet moments Var(Z_j), Cov(Z_j, Z_k) ------------------
    S = flat.sum(axis=-1)                  # (M,)
    S_col = S[:, None]                     # (M, 1)
    al_col = alpha_lead[:, None]           # (M, 1)
    denom = S_col * S_col * (S_col + 1.0)  # (M, 1)
    # Numerical safety: S could be 0 if all alpha are 0 (pathological).
    denom = np.maximum(denom, 1e-300)

    # Var(Z_j) = [alpha_1*(S - alpha_1) + alpha_j*(S - alpha_j) + 2*alpha_1*alpha_j] / (S^2 * (S+1))
    var_j = (
        al_col * (S_col - al_col)
        + alpha_comp * (S_col - alpha_comp)
        + 2.0 * al_col * alpha_comp
    ) / denom  # (M, K-1)
    var_j = np.maximum(var_j, 1e-300)

    # Cov(Z_j, Z_k) = [alpha_1*(S + alpha_j + alpha_k - alpha_1) - alpha_j*alpha_k] / (S^2 * (S+1))
    aj_i = alpha_comp[:, :, None]          # (M, K-1, 1)
    ak_i = alpha_comp[:, None, :]          # (M, 1, K-1)
    S_mat = S[:, None, None]               # (M, 1, 1)
    al_mat = alpha_lead[:, None, None]     # (M, 1, 1)
    denom_mat = np.maximum(S_mat * S_mat * (S_mat + 1.0), 1e-300)
    cov_jk = (
        al_mat * (S_mat + aj_i + ak_i - al_mat) - aj_i * ak_i
    ) / denom_mat  # (M, K-1, K-1)

    # rho_{jk} = Cov(Z_j, Z_k) / sqrt(Var(Z_j) * Var(Z_k))
    var_prod = var_j[:, :, None] * var_j[:, None, :]  # (M, K-1, K-1)
    rho_jk = cov_jk / np.sqrt(var_prod)               # (M, K-1, K-1)

    # --- Step 4: product over all p_j and per-pair remainders ---------------
    # log-space for numerical stability: prod_{l!=j,k}(p_l) = exp(sum_log_p - log_p_j - log_p_k).
    log_p = np.log(p_j)                              # (M, K-1)
    log_prod_all = log_p.sum(axis=-1)                # (M,)
    prod_all = np.exp(log_prod_all)                  # (M,)

    # log_prod_remainder[i, j, k] = sum_l log(p_l) - log(p_j) - log(p_k)
    log_prod_remainder = (
        log_prod_all[:, None, None]
        - log_p[:, :, None]
        - log_p[:, None, :]
    )  # (M, K-1, K-1)
    prod_remainder = np.exp(log_prod_remainder)       # (M, K-1, K-1)

    # --- Step 5: upper-triangular pair sum ---------------------------------
    # term_{jk} = rho_{jk} * phi_j * phi_k * prod_remainder_{jk}
    phi_mat = phi_j[:, :, None] * phi_j[:, None, :]   # (M, K-1, K-1)
    term = rho_jk * phi_mat * prod_remainder          # (M, K-1, K-1)

    # Sum only the strict upper triangle (pairs j < k)
    triu_mask = np.triu(
        np.ones((K - 1, K - 1), dtype=bool), k=1,
    )                                                  # (K-1, K-1)
    # Select and sum: term[:, triu_mask] -> (M, n_pairs)
    correction_sum = term[:, triu_mask].sum(axis=-1)   # (M,)

    damping = _damping_factor(K)
    P = prod_all + damping * correction_sum             # (M,)
    P = np.clip(P, 0.0, 1.0)
    return P.reshape(leading_shape)


def exceedance_probability_batched(
    alpha: np.ndarray,
    approx: Literal["damped", "mc"] = "damped",
) -> np.ndarray:
    """Vectorized exceedance probability over batched alpha vectors.

    Accepts alpha of shape ``(..., K)`` and returns an array of shape
    ``(...)`` with per-row exceedance probabilities.

    Paths:
      - K=2: exact beta via vectorized ``betainc`` (``approx`` ignored).
      - K>=3 with ``approx="damped"``: damped Gaussian-copula (default).
      - K>=3 with ``approx="mc"``: per-row loop over the scalar Monte Carlo
        path (slow, for parity/debugging only).

    Args:
        alpha: Array of shape ``(..., K)`` with Dirichlet alphas.
        approx: Approximation method for K>=3.

    Returns:
        Array of shape ``(...)`` with values in [0, 1].
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    K = alpha.shape[-1]
    leading_shape = alpha.shape[:-1]

    if K < 2:
        return np.ones(leading_shape, dtype=np.float64)

    if K == 2:
        # Exact beta, vectorized.  max/min handle the (a >= b) convention.
        a = alpha.max(axis=-1)
        b = alpha.min(axis=-1)
        return (1.0 - scipy.special.betainc(a, b, 0.5)).astype(np.float64)

    if approx == "damped":
        return _exceedance_damped_copula(alpha)

    # approx == "mc": per-row loop.  Not a hot path; used only for parity
    # tests and debugging.  The scalar MC path is deterministic (fixed seed)
    # so the result is reproducible.
    flat = alpha.reshape(-1, K)
    out = np.empty(flat.shape[0], dtype=np.float64)
    for i in range(flat.shape[0]):
        alpha_dict = {f"c{j}": float(v) for j, v in enumerate(flat[i])}
        out[i] = _exceedance_monte_carlo(alpha_dict)
    return out.reshape(leading_shape)


# ---------------------------------------------------------------------------
# Adaptive stopping simulation
# ---------------------------------------------------------------------------


def simulate_adaptive_stopping(
    probs_pool: np.ndarray,
    thresholds: np.ndarray,
    n_max: int,
    n_iterations: int,
    rng: np.random.Generator,
    prior_strength: float = 1.0,
    multiplier: float = 1.0,
    *,
    cot_pool: np.ndarray | None = None,
    escalation_weight: float | None = None,
    escalation_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Replay adaptive stopping over a probability-vector pool.

    For each of ``n_iterations`` bootstrap rows, draw ``n_max`` probability
    vectors with replacement from ``probs_pool``, build the cumulative
    Dirichlet-sum posterior trajectory, compute exceedance probability at
    every step, and derive the stop index for every threshold in a single
    pass.

    When ``cot_pool`` and ``escalation_weight`` are both supplied, the
    kernel additionally simulates an escalation branch for capped rows.
    For every bootstrap row, draw one CoT vector from ``cot_pool``, add
    ``escalation_weight × cot_vector`` to the alpha at the budget cap
    (step ``n_max - 1``), and take the argmax of the augmented posterior
    as ``map_idx_escalated``.  This MAP is τ-invariant by construction —
    it is a property of the row's direct trajectory + the drawn CoT
    vector, not of any threshold — so callers select between the direct
    ``map_idx[t, r]`` and the escalated MAP per-cell using the
    ``capped[t, r]`` mask.

    Args:
        probs_pool: (P, K) array of probability vectors (rows sum to 1).
        thresholds: (T,) array of confidence thresholds.
        n_max: Maximum number of samples (budget cap).  Used as the stop
            index for rows that never cross any threshold.
        n_iterations: Bootstrap iterations R.
        rng: NumPy random generator (callers should use a fixed seed for
            reproducibility).
        prior_strength: Weight of the uniform prior in the Dirichlet sum
            (default 1.0).
        multiplier: Multiplier applied to each observation's probability
            vector (default 1.0).
        cot_pool: (P_cot, K) array of CoT probability vectors used as
            the escalation source.  Must share K with ``probs_pool``.
            Set together with ``escalation_weight``; ``None`` disables
            escalation and yields ``map_idx_escalated=None``.
        escalation_weight: Non-negative scalar weight applied to a single
            CoT vector when augmenting the budget-cap alpha.  Set
            together with ``cot_pool``.
        escalation_rng: Optional separate generator used only for the CoT
            bootstrap draws.  Keeps the direct ``rng`` stream byte-stable
            relative to a no-escalation run.  Falls back to ``rng`` when
            ``None``.

    Returns:
        Tuple ``(stop_idx, map_idx, capped, map_idx_escalated)``.  The
        first three have shape ``(T, R)``; ``map_idx_escalated`` has
        shape ``(R,)`` and dtype ``int64`` when escalation is active,
        else ``None``:

          * ``stop_idx[t, r]`` is the 0-based sample index where
            threshold ``t`` was first met on iteration ``r``; ``n_max - 1``
            if the threshold was never met within the budget.
          * ``map_idx[t, r]`` is the argmax category index of the posterior
            at the stop step (the MAP decision that would have been made).
          * ``capped[t, r]`` is True when threshold ``t`` was never met on
            iteration ``r`` — the row would have kept sampling if not for
            the budget cap.  Distinct from ``stop_idx == n_max - 1``: a
            row whose exceedance first crosses ``tau`` exactly at step
            ``n_max - 1`` also has ``stop_idx == n_max - 1`` but is
            *committed*, not capped.
          * ``map_idx_escalated[r]`` is the argmax of
            ``alpha_at_cap + escalation_weight × cot_vector`` for row
            ``r``, or ``None`` when escalation is off (no CoT pool, or
            empty CoT pool, or K=1 degenerate).

    Raises:
        ValueError: if exactly one of ``cot_pool`` / ``escalation_weight``
            is provided; if ``escalation_weight`` is negative; if
            ``cot_pool`` is not 2-D or its second dimension differs from
            ``probs_pool``'s.
    """
    # XOR check on the two escalation params — both or neither.
    has_cot = cot_pool is not None
    has_weight = escalation_weight is not None
    if has_cot != has_weight:
        raise ValueError(
            "cot_pool and escalation_weight must both be provided or both be None"
        )
    if has_weight and not math.isfinite(escalation_weight):
        raise ValueError(
            f"escalation_weight must be finite, got {escalation_weight!r}"
        )
    if has_weight and escalation_weight < 0:
        raise ValueError("escalation_weight must be >= 0")

    probs_pool = np.asarray(probs_pool, dtype=np.float64)
    if probs_pool.ndim != 2:
        raise ValueError("probs_pool must be a (P, K) array")
    P, K = probs_pool.shape
    if P == 0:
        raise ValueError("probs_pool is empty")

    # Validate cot_pool shape against probs_pool's K.  Empty pool -> disable.
    escalation_active = has_cot and has_weight
    if escalation_active:
        cot_pool = np.asarray(cot_pool, dtype=np.float64)
        if cot_pool.ndim != 2:
            raise ValueError("cot_pool must be a (P_cot, K) array")
        if cot_pool.shape[1] != K:
            raise ValueError(
                f"cot_pool K={cot_pool.shape[1]} must match probs_pool K={K}"
            )
        if cot_pool.shape[0] == 0:
            escalation_active = False

    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    T = thresholds.shape[0]
    R = int(n_iterations)

    # K=1 degenerate: only one category, everything goes there and MAP is 0.
    # Exceedance is trivially 1.0 so no row is ever capped.  Escalation has
    # nothing to flip; return None for map_idx_escalated for consistency.
    if K == 1:
        stop_idx = np.zeros((T, R), dtype=np.int64)
        map_idx = np.zeros((T, R), dtype=np.int64)
        capped = np.zeros((T, R), dtype=bool)
        return stop_idx, map_idx, capped, None

    # Bootstrap: draw n_max indices per row with replacement.
    indices = rng.integers(0, P, size=(R, n_max))  # (R, n_max)
    sampled = probs_pool[indices]                   # (R, n_max, K)

    # Alpha trajectory: prior + multiplier * cumulative sum of prob vectors.
    alpha_traj = (
        prior_strength * DIRICHLET_ALPHA_0
        + multiplier * np.cumsum(sampled, axis=1)
    )                                               # (R, n_max, K)
    # Safety clamp: keep alphas strictly positive for betainc / log.
    alpha_traj = np.maximum(alpha_traj, 1e-9)

    # Exceedance trajectory via the batched damped path.
    excd = exceedance_probability_batched(alpha_traj)  # (R, n_max)

    # Per-threshold stop index: first step where excd >= tau; else n_max - 1.
    met = excd[None, :, :] >= thresholds[:, None, None]   # (T, R, n_max)
    stop_idx = met.argmax(axis=-1).astype(np.int64)        # (T, R), 0 if never
    capped = ~met.any(axis=-1)                             # (T, R)
    stop_idx[capped] = n_max - 1

    # Gather the posterior alphas at the stop step for each (T, R) cell.
    r_idx = np.broadcast_to(np.arange(R)[None, :], (T, R))  # (T, R)
    alpha_at_stop = alpha_traj[r_idx, stop_idx, :]          # (T, R, K)
    map_idx = alpha_at_stop.argmax(axis=-1).astype(np.int64)  # (T, R)

    # Escalation branch: τ-invariant MAP from alpha_at_cap + weight*cot_vec.
    # Drawn from a separate generator when supplied so the direct rng stream
    # is unaffected by the presence/absence of escalation.
    map_idx_escalated: np.ndarray | None
    if escalation_active:
        cot_gen = escalation_rng if escalation_rng is not None else rng
        P_cot = cot_pool.shape[0]
        cot_idx = cot_gen.integers(0, P_cot, size=R)  # (R,)
        cot_vectors = cot_pool[cot_idx]                # (R, K)
        alpha_at_cap = alpha_traj[:, -1, :]            # (R, K)
        alpha_escalated = alpha_at_cap + float(escalation_weight) * cot_vectors
        map_idx_escalated = alpha_escalated.argmax(axis=-1).astype(np.int64)
    else:
        map_idx_escalated = None

    return stop_idx, map_idx, capped, map_idx_escalated


# ---------------------------------------------------------------------------
# Interval probability (NIG confidence)
# ---------------------------------------------------------------------------


def interval_probability(
    posterior: NIGPosterior, L: float = DEFAULT_INTERVAL_L,
) -> float:
    """Compute interval probability P(true mean within L of posterior mean).

    Uses the Student-t predictive distribution derived from the NIG posterior.

    Args:
        posterior: NIG posterior.
        L: Half-width of the interval in normalized [0,1] space.

    Returns:
        Probability in [0, 1].
    """
    kappa, mu, alpha, beta = posterior.kappa, posterior.mu, posterior.alpha, posterior.beta

    if kappa <= 0:
        return 0.0  # No observations yet

    scale = math.sqrt(beta * (kappa + 1) / (alpha * kappa))
    df = 2.0 * alpha

    if scale <= 0 or not math.isfinite(scale):
        return 1.0  # Extremely tight posterior

    lower = max(0.0, mu - L)
    upper = min(1.0, mu + L)

    # Student-t CDF evaluations
    interval_mass = (
        scipy.stats.t.cdf((upper - mu) / scale, df)
        - scipy.stats.t.cdf((lower - mu) / scale, df)
    )
    total_mass = (
        scipy.stats.t.cdf((1.0 - mu) / scale, df)
        - scipy.stats.t.cdf((0.0 - mu) / scale, df)
    )

    if total_mass <= 0:
        return 0.0

    return float(min(1.0, interval_mass / total_mass))


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------


def snap_to_label(value: float, answer_options: dict[str, float]) -> str:
    """Find the answer label whose numeric value is closest to the given value.

    Args:
        value: Resolved numeric value (in original range).
        answer_options: Mapping of label -> numeric value.

    Returns:
        Label of the closest answer option.
    """
    return min(answer_options, key=lambda label: abs(answer_options[label] - value))


def make_prior(
    spec: ProblemSpec,
) -> DirichletPosterior | NIGPosterior:
    """Create a fresh prior based on problem type.

    Args:
        spec: Problem specification.

    Returns:
        DirichletPosterior or NIGPosterior.
    """
    if isinstance(spec, CategoricalSpec):
        return dirichlet_prior(spec.categories)
    return nig_prior()


def resolve_value(
    posterior: DirichletPosterior | NIGPosterior,
    spec: ProblemSpec,
) -> str | float:
    """Resolve the posterior to a point estimate.

    Args:
        posterior: Dirichlet or NIG posterior.
        spec: Problem specification.

    Returns:
        Category label (str) or denormalized numeric value (float).
    """
    if isinstance(posterior, DirichletPosterior):
        return dirichlet_map(posterior)
    return denormalize_value(nig_mean(posterior), spec.range_min, spec.range_max)


def score_resolved(
    resolved: str | float,
    spec: ProblemSpec,
    scores: dict[str, float],
) -> float:
    """Map a resolved value to its score via the resolve->snap->label->score pipeline.

    Args:
        resolved: Resolved value from posterior (category label or denormalized float).
        spec: Problem specification.
        scores: Mapping from label to credit value (e.g. 1.0, 0.5, 0.0).

    Returns:
        Score (float credit) for the resolved value.
    """
    if isinstance(spec, CategoricalSpec):
        return scores[resolved]  # resolved is already a label
    label = snap_to_label(resolved, spec.answer_options)
    return scores[label]


def resolve_and_score(
    posterior: DirichletPosterior | NIGPosterior,
    spec: ProblemSpec,
    scores: dict[str, float],
) -> float:
    """Full pipeline: posterior -> resolved value -> score.

    Args:
        posterior: Dirichlet or NIG posterior.
        spec: Problem specification.
        scores: Mapping from label to credit value.

    Returns:
        Score (float credit) for the posterior's resolved value.
    """
    return score_resolved(
        resolve_value(posterior, spec),
        spec, scores,
    )



# ---------------------------------------------------------------------------
# Confidence dispatch
# ---------------------------------------------------------------------------


def compute_confidence(
    posterior: DirichletPosterior | NIGPosterior,
) -> float:
    """Compute the appropriate confidence measure for the posterior type.

    Args:
        posterior: Dirichlet or NIG posterior.

    Returns:
        Exceedance probability (Dirichlet) or interval probability (NIG).
    """
    if isinstance(posterior, DirichletPosterior):
        return exceedance_probability(posterior)
    return interval_probability(posterior)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def _subsample_iterate(
    direct_samples: list[Sample],
    cot_samples: list[Sample],
    n_direct: int,
    n_cot: int,
    weight_config: WeightConfig,
    spec: ProblemSpec,
    scores: dict[str, float],
    n_iterations: int,
    rng: np.random.Generator,
    *,
    need_confidence: bool = True,
):
    """Yield (confidence, score) for each resampling iteration.

    Shared iteration logic for subsample_calibration (and future callers).

    Args:
        need_confidence: If False, yield 0.0 for confidence to skip the
            expensive exceedance/interval computation.
    """
    if n_direct > len(direct_samples) or n_cot > len(cot_samples):
        raise ValueError(
            f"Cannot draw n_direct={n_direct} from {len(direct_samples)} "
            f"or n_cot={n_cot} from {len(cot_samples)}"
        )

    for _ in range(n_iterations):
        posterior = make_prior(spec)

        # Draw without replacement
        if n_direct > 0:
            d_indices = rng.choice(len(direct_samples), size=n_direct, replace=False)
            for idx in d_indices:
                s = direct_samples[idx]
                posterior = process_observation(
                    posterior, s.value, s.is_na, weight_config,
                )

        if n_cot > 0:
            c_indices = rng.choice(len(cot_samples), size=n_cot, replace=False)
            for idx in c_indices:
                s = cot_samples[idx]
                posterior = process_observation(
                    posterior, s.value, s.is_na, weight_config,
                )

        confidence = compute_confidence(posterior) if need_confidence else 0.0
        yield (
            confidence,
            resolve_and_score(posterior, spec, scores),
        )


def subsample_calibration(
    direct_samples: list[Sample],
    cot_samples: list[Sample],
    scores: dict[str, float],
    n_direct: int,
    n_cot: int,
    weight_config: WeightConfig,
    spec: ProblemSpec,
    n_iterations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Return per-iteration (confidence, score) pairs for calibration analysis.

    Args:
        direct_samples: Pool of direct samples.
        cot_samples: Pool of CoT samples.
        scores: Mapping from label to credit value.
        n_direct: Number of direct samples to draw.
        n_cot: Number of CoT samples to draw.
        weight_config: Observation weight configuration.
        spec: Problem specification.
        n_iterations: Number of resampling iterations.
        rng: Numpy random generator (created with default seed if None).

    Returns:
        List of (confidence, score) tuples, one per iteration.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    return list(_subsample_iterate(
        direct_samples, cot_samples, n_direct, n_cot,
        weight_config, spec, scores,
        n_iterations, rng,
    ))
