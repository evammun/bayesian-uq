"""
Dirichlet posterior utilities for Bayesian uncertainty quantification.

The Dirichlet distribution is the conjugate prior for categorical observations.
Starting from a uniform prior Dirichlet(1, 1, ..., 1), each observed answer
increments the corresponding pseudo-count by 1. The posterior is always Dirichlet.

Key quantities computed here:
  - Exceedance probability: P(leading answer has the highest probability)
    estimated via Monte Carlo using the Gamma sampling trick.
  - Posterior entropy: how spread out the posterior is (higher = more uncertain).
  - Posterior mean: expected probability for each answer choice.
"""

import numpy as np
from scipy.special import digamma, gammaln


def init_prior(num_choices: int = 4) -> np.ndarray:
    """
    Initialise a uniform Dirichlet prior with one pseudo-count per choice.

    Dirichlet(1, 1, ..., 1) represents maximum ignorance — all possible
    distributions over answers are equally likely a priori.

    Args:
        num_choices: Number of answer options (default 4 for A/B/C/D).

    Returns:
        Array of ones with shape (num_choices,).
    """
    return np.ones(num_choices, dtype=np.float64)


def update_posterior(alpha: np.ndarray, observation: int) -> np.ndarray:
    """
    Update the Dirichlet posterior by incrementing the pseudo-count for
    the observed answer choice.

    This is the conjugate update: if the prior is Dirichlet(alpha) and we
    observe category k, the posterior is Dirichlet(alpha') where
    alpha'[k] = alpha[k] + 1 and all other components stay the same.

    Args:
        alpha: Current pseudo-counts, shape (K,).
        observation: Index of the observed answer (0 to K-1).

    Returns:
        New pseudo-counts array (the input is NOT modified).
    """
    alpha_new = alpha.copy()
    alpha_new[observation] += 1.0
    return alpha_new


def exceedance_probability(
    alpha: np.ndarray,
    num_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo estimate of the exceedance probability for the leading answer.

    The exceedance probability is:
        P(theta_lead > theta_k for all k != lead)
    where theta ~ Dirichlet(alpha) and 'lead' is the answer with the highest
    pseudo-count.

    Uses the Gamma sampling trick: draw independent Gamma(alpha_k, 1) samples,
    then normalise to get a Dirichlet draw. This is exact — not an approximation
    of the Dirichlet distribution, but a mathematically equivalent way to sample
    from it.

    Args:
        alpha: Pseudo-counts, shape (K,).
        num_samples: Number of Monte Carlo draws (default 10,000).
        rng: NumPy random generator for reproducibility.

    Returns:
        Exceedance probability in [0, 1]. Values near 1.0 mean high confidence
        that the leading answer is correct.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw Gamma(alpha_k, 1) samples for each component — shape: (num_samples, K)
    gamma_samples = np.column_stack(
        [rng.gamma(a, 1.0, size=num_samples) for a in alpha]
    )

    # Normalise each row to get Dirichlet samples
    dirichlet_samples = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)

    # The leading answer is the one with the highest pseudo-count
    leading_idx = int(np.argmax(alpha))

    # Count how often the leading answer has the highest sampled probability
    winners = np.argmax(dirichlet_samples, axis=1)
    exceedance = np.mean(winners == leading_idx)

    return float(exceedance)


def posterior_entropy(alpha: np.ndarray) -> float:
    """
    Compute the differential entropy of a Dirichlet distribution.

    Formula:
        H(Dir(alpha)) = ln B(alpha) - (K - 1) * psi(alpha_0)
                        + sum_k (alpha_k - 1) * psi(alpha_k)

    where alpha_0 = sum(alpha), psi = digamma function, and
    B(alpha) = prod(Gamma(alpha_k)) / Gamma(alpha_0) is the multivariate Beta.

    Higher entropy → more uncertainty (posterior is spread across answers).
    Lower entropy → more concentration (posterior is peaked on one answer).

    Note: differential entropy can be negative for very concentrated distributions.
    This is normal and expected — it's not a probability.

    Args:
        alpha: Pseudo-counts, shape (K,).

    Returns:
        Differential entropy of the Dirichlet distribution.
    """
    alpha_0 = alpha.sum()
    K = len(alpha)

    # ln B(alpha) = sum(ln Gamma(alpha_k)) - ln Gamma(alpha_0)
    ln_beta = gammaln(alpha).sum() - gammaln(alpha_0)

    entropy = (
        ln_beta
        - (K - 1) * digamma(alpha_0)
        + ((alpha - 1) * digamma(alpha)).sum()
    )

    return float(entropy)


def posterior_mean(alpha: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the Dirichlet distribution.

    E[theta_k] = alpha_k / alpha_0, where alpha_0 = sum(alpha).

    This gives the expected probability for each answer choice under
    the current posterior. With a uniform prior Dirichlet(1,1,1,1) and
    3 observations of answer B, the mean would be [1/7, 4/7, 1/7, 1/7].

    Args:
        alpha: Pseudo-counts, shape (K,).

    Returns:
        Mean probabilities, shape (K,), summing to 1.0.
    """
    return alpha / alpha.sum()
