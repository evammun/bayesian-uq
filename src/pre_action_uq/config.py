"""
Data models for the Pre-Action UQ framework (QuALITY-based).

Adapted from v2 MMLU config.py with changes:
  - QuestionRecord adapted for QuALITY dataset (article context, difficulty)
  - ExperimentConfig adds context_condition, model_path, n_ctx
  - QuestionResult includes QuALITY metadata (article_id, source, topic, difficult)
  - Removes MMLU-specific fields (subject, pair_id, variant, break_type)

Three layers:
  1. Config layer — ExperimentConfig (how we're evaluating)
  2. Result layer — QueryResult, QuestionResult, ExperimentResult (what happened)
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
# Config layer: Experiment configuration
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run (loaded from YAML).

    Each YAML file in experiments/configs/ maps to one ExperimentConfig.
    """

    run_name: str
    model_path: str = "auto"  # path to GGUF file, or "auto" to search standard locations
    model_name: str = "qwen3:8b-q4_K_M"  # model identifier (for logging/reference)
    think: bool = False
    prompt_mode: Literal["direct", "cot"] = "direct"
    dataset_file: str = "data/quality_all.jsonl"
    context_condition: Literal["sufficient", "insufficient"] = "sufficient"
    shuffle_options: bool = True  # shuffle A/B/C/D order per query; False = identity permutation
    max_questions: int | None = None
    seed: int = 42
    temperature: float = 0.7
    num_permutations: int = 10  # total queries per question (each with a different answer permutation)
    n_ctx: int = 12288  # 12K fits all QuALITY articles; 8192 overflows ~42% of them


# ---------------------------------------------------------------------------
# Result layer: Raw data recording
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """Record of a single model query within a question's evaluation.

    Same structure as v2 — stores raw logprobs plus normalised probs.
    """

    query_number: int
    paraphrase_index: int = Field(
        description="Index into paraphrase list, or -1 for original question",
    )
    paraphrase_category: str = Field(
        default="",
        description="Paraphrase category: 'A' (text-grounded), 'B' (pure rephrase), 'C' (angle shift), or '' (shuffle-only/no paraphrase)",
    )
    query_text: str  # the actual prompt sent to the model (truncated for storage)
    answer_permutation: list[int] = Field(
        description="Display position → canonical index mapping",
    )
    raw_response: str
    raw_logprobs: list[dict]

    # Per-letter logprobs BEFORE normalisation
    display_letter_logprobs: dict[str, float]
    canonical_logprobs: dict[int, float]

    # Normalised probabilities in canonical order [P(0), P(1), P(2), P(3)]
    canonical_probs: list[float] = Field(
        default_factory=list,
        description="Normalised P(A), P(B), P(C), P(D) in canonical order",
    )

    # Model's selected answer
    display_answer: str
    canonical_answer: int

    thinking_trace: str = ""

    # CoT Pass 1 answer capture — what the model freely chose before Pass 2
    pass1_answer: str = ""  # display letter (e.g. "B") or "" if not found
    pass1_canonical_answer: int = -1  # canonical index, or -1 if not found


class QuestionResult(BaseModel):
    """Full result for one question across all its queries.

    Includes QuALITY-specific metadata for downstream analysis.
    """

    question_id: str  # question_unique_id from QuALITY
    question_unique_id: str
    article_id: str
    question_text: str
    options: list[str]
    correct_answer: int  # 0-indexed
    difficult: bool

    # Context condition metadata
    context_condition: str  # "sufficient" or "insufficient"
    swapped_article_id: str | None = None  # for insufficient condition

    query_log: list[QueryResult]
    num_queries: int
    correct: bool | None = None  # None if skipped
    skipped: bool = False  # True if all queries failed (e.g. n_ctx overflow)

    # Aggregated probabilities
    mean_probs: list[float] = Field(
        default_factory=list,
        description="Mean of canonical_probs across all queries",
    )

    final_answer: int
    answer_counts: dict[int, int]


class ExperimentResult(BaseModel):
    """Complete output of one experiment run."""

    run_name: str
    config: ExperimentConfig
    timestamp: str  # ISO 8601 — experiment start
    completed_at: str | None = None  # ISO 8601 — experiment end (set on final save)
    question_results: list[QuestionResult]
