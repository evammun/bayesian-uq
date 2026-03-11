"""Bayesian UQ — Black-box uncertainty quantification for LLM outputs."""

from .config import (
    ANSWER_LETTERS,
    NUM_CHOICES,
    ExperimentConfig,
    ExperimentResult,
    ParaphraseRecord,
    QueryResult,
    QuestionRecord,
    QuestionResult,
)
from .dirichlet import (
    exceedance_probability,
    init_prior,
    posterior_entropy,
    posterior_mean,
    update_posterior,
)

__all__ = [
    # Constants
    "ANSWER_LETTERS",
    "NUM_CHOICES",
    # Config models
    "QuestionRecord",
    "ParaphraseRecord",
    "ExperimentConfig",
    "QueryResult",
    "QuestionResult",
    "ExperimentResult",
    # Dirichlet utilities
    "init_prior",
    "update_posterior",
    "exceedance_probability",
    "posterior_entropy",
    "posterior_mean",
]
