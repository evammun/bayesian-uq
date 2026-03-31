"""
Data models for the Bayesian UQ framework (v2 — logprob-based).

All data structures used across the project are defined here using Pydantic
models. This provides validation, JSON serialisation, and clear type contracts
that the rest of the codebase depends on.

Three layers:
  1. Data layer   — QuestionRecord, ParaphraseRecord (what we're evaluating)
  2. Config layer — ExperimentConfig (how we're evaluating it)
  3. Result layer — QueryResult, QuestionResult, ExperimentResult (what happened)

v2 changes from v1:
  - No more Dirichlet posterior, exceedance, or stopping criteria
  - QueryResult stores raw logprobs from Ollama (no normalisation)
  - QuestionResult stores raw answer counts (no computed metrics yet)
  - ExperimentConfig drops confidence_threshold, max_queries_per_question,
    monte_carlo_samples, parallel_workers; adds num_paraphrases
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
# Data layer: Question database and paraphrase bank
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
# Config layer: Experiment configuration
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run (loaded from YAML).

    Each YAML file in experiments/configs/ maps to one ExperimentConfig.
    The question_set field determines which questions from the database
    to include: "broken_pairs", "mmlu_standard", "pilot", "all",
    or a comma-separated list of question_ids.

    v2 simplification: every paraphrase is queried exactly once. No stopping
    criteria, no Dirichlet, no Monte Carlo. The query budget is fixed at
    num_paraphrases + 1 (original + paraphrases).
    """

    run_name: str
    model: str
    think: bool = False
    prompt_mode: Literal["direct", "cot", "cot_structured"] = "direct"
    question_set: str = "mmlu_standard"
    max_questions: int | None = None
    seed: int = 42
    shuffle_choices: bool = True
    use_paraphrases: bool = True
    temperature: float = 0.7
    num_paraphrases: int = 10  # max paraphrases to use per question


# ---------------------------------------------------------------------------
# Result layer: Raw data recording (one per query, one per question, one per run)
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """Record of a single model query within a question's evaluation.

    Stores both raw logprob data (audit trail) and normalised probabilities
    (what analysis uses). The raw values are kept exactly as Ollama returns
    them; the normalised values are derived from them in the pipeline.
    """

    query_number: int
    paraphrase_index: int = Field(
        description="Index into the paraphrase list, or -1 for the original question",
    )
    query_text: str  # the actual text sent to the model
    answer_permutation: list[int] = Field(
        description=(
            "Mapping from display position to canonical index. "
            "e.g. [2, 0, 3, 1] means display slot A shows canonical choice 2."
        ),
    )
    raw_response: str  # the full text the model output
    raw_logprobs: list[dict]  # the ENTIRE logprobs array from Ollama, unmodified

    # Per-letter raw data BEFORE any normalisation.
    # Keys are display letters ("A", "B", "C", "D"), values are raw logprobs from Ollama.
    # If a letter wasn't in top_logprobs, it's absent from the dict (NOT filled with a default).
    display_letter_logprobs: dict[str, float]

    # Same data mapped to canonical positions via answer_permutation.
    # Keys are canonical indices (0, 1, 2, 3), values are raw logprobs.
    canonical_logprobs: dict[int, float]

    # Normalised probabilities in canonical order [P(0), P(1), P(2), P(3)].
    # Derived from canonical_logprobs: exp(logprob) for each position,
    # exp(-30) for missing positions, then normalised to sum to 1.0.
    canonical_probs: list[float] = Field(
        default_factory=list,
        description="Normalised P(A), P(B), P(C), P(D) in canonical order, summing to ~1.0",
    )

    # What the model actually picked (highest prob token among A/B/C/D)
    display_answer: str  # the letter as displayed ("B")
    canonical_answer: int  # mapped to canonical index

    thinking_trace: str = ""

    # Post-commitment logprobs (think mode only): extracted from the streamed
    # response AFTER the model has already stated its answer. Compared with
    # canonical_probs (pre-commitment, from Pass 2) to quantify how much
    # stating the answer collapses the probability distribution.
    committed_canonical_probs: list[float] | None = None
    committed_display_letter_logprobs: dict[str, float] | None = None
    committed_canonical_logprobs: dict[int, float] | None = None
    committed_display_answer: str | None = None
    committed_canonical_answer: int | None = None


class QuestionResult(BaseModel):
    """Full result for one question across all its queries.

    Stores raw query logs plus lightweight aggregation (mean probabilities).
    Full uncertainty decomposition (entropy, JSD, epistemic) is deferred
    to analysis.py.
    """

    question_id: str
    query_log: list[QueryResult]
    num_queries: int
    correct: bool | None = None  # None for broken-premise questions

    # Aggregated probabilities: element-wise mean of canonical_probs across all queries.
    # mean_probs[i] = average probability assigned to canonical answer i.
    mean_probs: list[float] = Field(
        default_factory=list,
        description="Mean of canonical_probs across all queries",
    )

    # Final answer = argmax(mean_probs). Better than majority vote because
    # it accounts for confidence: a 60/40 query contributes less certainty
    # than a 99/1 query.
    final_answer: int

    # Vote counts kept for quick reference (how many times each canonical
    # answer was the argmax of an individual query's distribution)
    answer_counts: dict[int, int]


class ExperimentResult(BaseModel):
    """Complete output of one experiment run.

    Saved as a single JSON file in results/, named by run_name and timestamp.
    """

    run_name: str
    config: ExperimentConfig
    timestamp: str  # ISO 8601 UTC
    question_results: list[QuestionResult]
