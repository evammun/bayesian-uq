"""
Data models for the Bayesian UQ framework.

All data structures used across the project are defined here using Pydantic
models. This provides validation, JSON serialisation, and clear type contracts
that the rest of the codebase depends on.

Three layers:
  1. Data layer   — QuestionRecord, ParaphraseRecord (what we're evaluating)
  2. Config layer — ExperimentConfig (how we're evaluating it)
  3. Result layer — QueryResult, QuestionResult, ExperimentResult (what happened)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_LETTERS = ["A", "B", "C", "D"]
NUM_CHOICES = 4


# ---------------------------------------------------------------------------
# Phase 1: Question database and paraphrase bank
# ---------------------------------------------------------------------------

class QuestionRecord(BaseModel):
    """A single question in the master question database (data/questions.json).

    Choices are stored as a list of strings in canonical order:
      index 0 = A, index 1 = B, index 2 = C, index 3 = D.
    Answer shuffling happens at query time, not here.
    """

    question_id: str
    question_text: str
    choices: list[str] = Field(..., min_length=NUM_CHOICES, max_length=NUM_CHOICES)
    correct_answer: int | None = Field(
        None, ge=0, lt=NUM_CHOICES,
        description="Index 0-3 for valid questions, None for broken-premise",
    )
    subject: str
    pair_id: str | None = None
    variant: Literal["valid", "broken"] | None = None
    break_type: Literal[
        "existence",
        "contradiction",
        "category_error",
        "temporal",
        "mathematical",
        "entity_mismatch",
    ] | None = None
    break_description: str | None = None
    source: str  # "mmlu_redux" or "original"


class ParaphraseRecord(BaseModel):
    """A single paraphrase of a question stem.

    The paraphrase is just the question text — answer choices are NOT included
    because answer ordering is handled separately at query time.
    """

    text: str
    embedding_similarity: float | None = None
    lexical_distance: float | None = None


# ---------------------------------------------------------------------------
# Phase 2: Experiment configuration
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run (loaded from YAML).

    Each YAML file in experiments/configs/ maps to one ExperimentConfig.
    The question_set field determines which questions from the database
    to include: "broken_pairs", "mmlu_standard", "pilot", "all",
    or a comma-separated list of question_ids.
    """

    run_name: str
    model: str
    think: bool = False
    question_set: str
    confidence_threshold: float = 0.95
    max_queries_per_question: int = 12
    max_questions: int | None = None  # if set, take a stratified sample of N questions across subjects
    seed: int = 42                    # random seed for stratified sampling and experiment RNGs
    shuffle_choices: bool = True      # if False, answer order stays A/B/C/D (no permutation)
    use_paraphrases: bool = True      # if False, re-ask original question text every query
    monte_carlo_samples: int = 10_000
    temperature: float = 0.7
    parallel_workers: int = 1  # concurrent question workers (1 = sequential)


# ---------------------------------------------------------------------------
# Phase 2: Result recording (one per query, one per question, one per run)
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """Record of a single model query within a question's evaluation.

    Stores everything needed to replay the posterior evolution without
    re-querying the model.
    """

    query_number: int
    paraphrase_index: int = Field(
        description="Index into the paraphrase list, or -1 for the original question",
    )
    answer_permutation: list[int] = Field(
        description=(
            "Mapping from display position to canonical index. "
            "e.g. [2, 0, 3, 1] means display slot A shows canonical choice 2."
        ),
    )
    raw_model_response: str  # the exact JSON string returned by the model
    thinking_trace: str = ""  # model's reasoning tokens (empty when think=False)
    canonical_answer: int = Field(
        ge=0, lt=NUM_CHOICES,
        description="Answer mapped back to canonical index (0-3)",
    )
    alpha_after: list[float]      # Dirichlet pseudo-counts after this update
    exceedance_after: float       # exceedance probability after this update
    entropy_after: float          # posterior entropy after this update


class QuestionResult(BaseModel):
    """Full result for one question across all its queries."""

    question_id: str
    query_log: list[QueryResult]
    final_answer: int = Field(
        ge=0, lt=NUM_CHOICES,
        description="Posterior mode — the answer with the highest pseudo-count",
    )
    final_alpha: list[float]
    final_exceedance: float
    final_entropy: float
    queries_used: int
    stopped_early: bool  # True if exceedance hit the threshold before budget ran out
    correct: bool | None = None  # None for broken-premise questions


class ExperimentResult(BaseModel):
    """Complete output of one experiment run.

    Saved as a single JSON file in results/, named by run_name and timestamp.
    """

    run_name: str
    config: ExperimentConfig
    timestamp: str  # ISO 8601 UTC
    question_results: list[QuestionResult]
