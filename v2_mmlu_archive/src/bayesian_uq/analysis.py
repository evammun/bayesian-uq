"""
Uncertainty metrics computed from raw logprob data.

These functions are NOT called during experiment runs — the pipeline stores
raw data only. These are for post-hoc analysis once we've verified the raw
logprob data looks correct.

All functions take lists of raw logprob values and return standard
information-theoretic uncertainty decomposition metrics.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon


def logprobs_to_probs(
    canonical_logprobs: dict[int, float],
    num_choices: int = 4,
    floor: float = 1e-10,
) -> list[float]:
    """Convert raw canonical logprobs to a normalised probability vector.

    Args:
        canonical_logprobs: {canonical_index: raw_logprob} — may be missing some indices.
        num_choices: Total number of answer choices (default 4).
        floor: Probability floor for missing choices (not in top_logprobs).

    Returns:
        List of num_choices probabilities summing to 1.0, in canonical order.
    """
    # Convert logprobs to raw probabilities
    raw_probs = []
    for i in range(num_choices):
        if i in canonical_logprobs:
            raw_probs.append(math.exp(canonical_logprobs[i]))
        else:
            raw_probs.append(floor)

    # Normalise to sum to 1.0
    total = sum(raw_probs)
    return [p / total for p in raw_probs]


def compute_question_metrics(canonical_probs_list: list[list[float]]) -> dict:
    """Compute uncertainty metrics from a list of probability vectors.

    Args:
        canonical_probs_list: List of N probability vectors, each [P(A), P(B), P(C), P(D)].
            These should already be normalised (sum to 1.0).

    Returns:
        Dict with:
        - mean_probs: mean probability vector across all queries
        - final_answer: argmax of mean_probs
        - mean_entropy: mean of per-query entropies (aleatoric uncertainty)
        - mean_of_dist_entropy: entropy of the mean distribution (total uncertainty)
        - epistemic_uncertainty: total - aleatoric (mutual information)
        - jsd: Jensen-Shannon divergence across all query distributions
        - agreement: fraction of queries where argmax matches overall argmax
    """
    probs = np.array(canonical_probs_list)  # shape: (N, K)

    # Mean probability vector
    mean_probs = probs.mean(axis=0)
    final_answer = int(np.argmax(mean_probs))

    # Per-query entropies (aleatoric uncertainty)
    per_query_entropies = [scipy_entropy(p, base=2) for p in probs]
    mean_entropy = float(np.mean(per_query_entropies))

    # Entropy of the mean distribution (total uncertainty)
    mean_of_dist_entropy = float(scipy_entropy(mean_probs, base=2))

    # Epistemic uncertainty = total - aleatoric (mutual information)
    epistemic_uncertainty = mean_of_dist_entropy - mean_entropy

    # Jensen-Shannon divergence across all query distributions
    # JSD is the average KL divergence from each distribution to the mean
    if len(probs) > 1:
        jsd = float(_jsd_multi(probs))
    else:
        jsd = 0.0

    # Agreement: how often does each query's argmax match the overall argmax?
    per_query_argmax = np.argmax(probs, axis=1)
    agreement = float(np.mean(per_query_argmax == final_answer))

    return {
        "mean_probs": mean_probs.tolist(),
        "final_answer": final_answer,
        "mean_entropy": mean_entropy,
        "mean_of_dist_entropy": mean_of_dist_entropy,
        "epistemic_uncertainty": epistemic_uncertainty,
        "jsd": jsd,
        "agreement": agreement,
    }


def _jsd_multi(probs: np.ndarray) -> float:
    """Compute the multi-distribution Jensen-Shannon divergence.

    JSD(P1, P2, ..., PN) = H(mean) - mean(H(Pi))

    This is equivalent to the mutual information between the paraphrase
    index and the output distribution, and equals epistemic_uncertainty
    computed above. We compute it here as a cross-check.

    Args:
        probs: Array of shape (N, K) where each row is a probability distribution.

    Returns:
        JSD value in bits (base 2).
    """
    mean_dist = probs.mean(axis=0)
    h_mean = scipy_entropy(mean_dist, base=2)
    mean_h = np.mean([scipy_entropy(p, base=2) for p in probs])
    return max(0.0, h_mean - mean_h)  # Clamp to 0 for numerical stability
