"""
Main experiment pipeline for Bayesian UQ v2 (logprob-based).

Orchestrates the experiment loop:
  1. Load questions and paraphrases from JSON files
  2. For each question: query the model once per paraphrase (+ original)
  3. Extract raw logprobs from each query
  4. Save all results incrementally to a timestamped JSON file

v2 simplification: no Dirichlet posterior, no stopping criteria, no Monte Carlo.
Every paraphrase is queried exactly once. The signal comes from the logprob
distributions, not from repeated categorical votes.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import yaml

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
from .query import OllamaClient, extract_answer_logprobs, generate_permutation


# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------

class _IncrementalWriter:
    """Efficiently writes ExperimentResult JSON by appending, not rewriting.

    The naive approach (model_dump_json on every save) re-serializes all
    question results each time. At 5,000 questions × 11 queries this takes
    ~160 seconds per save — O(n²) total serialization and O(n) I/O per save,
    completely dominating the actual inference time.

    This writer takes a different approach:
      1. On the first call, writes the full JSON header and opening bracket.
      2. On each subsequent call, seeks to just before the closing ']\n}'
         and appends only the new QuestionResult JSON.
      3. Each save is O(size_of_one_result) in both CPU and I/O.

    The output file is always valid JSON, readable by the dashboard at any
    point. The format is semantically identical to model_dump_json(indent=2).
    """

    def __init__(self) -> None:
        self._n_written: int = 0
        self._tail_len: int = 0  # byte length of the closing '\n  ]\n}'

    @staticmethod
    def _indent_result(json_str: str) -> str:
        """Indent a result JSON block by 4 spaces (matching indent=2 nesting)."""
        lines = json_str.split("\n")
        return "\n".join("    " + line for line in lines)

    def write(self, path: Path, data: ExperimentResult) -> None:
        """Append new question results to the JSON file on disk."""
        new_results = data.question_results[self._n_written:]
        if not new_results:
            return

        if self._n_written == 0:
            # First write: build full file from scratch
            # Serialize header (config, metadata) via an empty-results shell
            shell = data.model_copy(update={"question_results": []})
            shell_json = shell.model_dump_json(indent=2)
            marker = '"question_results": []'
            idx = shell_json.find(marker)
            if idx == -1:
                raise ValueError("Could not find question_results in serialized JSON")
            header_end = idx + len('"question_results": ')
            header = shell_json[:header_end]

            # Serialize all initial results
            parts = []
            for qr in new_results:
                parts.append(self._indent_result(qr.model_dump_json(indent=2)))

            tail = "\n  ]\n}"
            full_json = header + "[\n" + ",\n".join(parts) + tail
            self._tail_len = len(tail.encode("utf-8"))

            # Write atomically
            tmp_path = path.with_suffix(".tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(full_json)
                tmp_path.replace(path)
            except OSError:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(full_json)
        else:
            # Subsequent writes: in-place seek to overwrite the closing tail,
            # then append new results + new tail. Only writes the new bytes,
            # regardless of how large the file already is.
            new_parts = []
            for qr in new_results:
                new_parts.append(self._indent_result(qr.model_dump_json(indent=2)))

            tail = "\n  ]\n}"
            append_str = ",\n" + ",\n".join(new_parts) + tail
            append_bytes = append_str.encode("utf-8")

            with open(path, "r+b") as f:
                f.seek(-self._tail_len, 2)
                f.write(append_bytes)
                f.truncate()

            self._tail_len = len(tail.encode("utf-8"))

        self._n_written = len(data.question_results)


# ---------------------------------------------------------------------------
# Data loading helpers (reused from v1)
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[QuestionRecord]:
    """Load the master question database from data/questions.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [QuestionRecord(**q) for q in data]


def load_paraphrases(path: Path) -> dict[str, list[ParaphraseRecord]]:
    """Load the paraphrase bank from data/paraphrases.json.

    Returns a dict keyed by question_id, each value is a list of
    ParaphraseRecord objects. Returns an empty dict if the file
    doesn't exist (the pipeline will fall back to re-asking the
    original question with different answer permutations).

    Handles two formats:
      - Old format: {"qid": [{"text": "..."}, ...]}
      - New format: {"qid": {"original": "...", "paraphrases": ["...", ...]}}
    """
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, list[ParaphraseRecord]] = {}
    for qid, val in data.items():
        if isinstance(val, list):
            # Old format: list of {text: ...} dicts
            result[qid] = [ParaphraseRecord(**p) for p in val]
        elif isinstance(val, dict) and "paraphrases" in val:
            # New format: {original: ..., paraphrases: [str, ...]}
            result[qid] = [
                ParaphraseRecord(text=p) for p in val["paraphrases"]
            ]
        else:
            print(f"  Warning: unexpected format for paraphrases[{qid}], skipping")

    return result


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


# ---------------------------------------------------------------------------
# Question filtering (reused from v1)
# ---------------------------------------------------------------------------

def filter_questions(
    questions: list[QuestionRecord],
    question_set: str,
) -> list[QuestionRecord]:
    """Filter the question database based on the experiment's question_set field.

    Supported values:
      - "all"            -> every question
      - "broken_pairs"   -> only questions that have a pair_id (valid + broken)
      - "mmlu_standard"  -> MMLU questions without a broken variant
      - "pilot"          -> first 4 questions (quick sanity check)
      - comma-separated  -> specific question_ids
    """
    if question_set == "all":
        return questions
    elif question_set == "broken_pairs":
        return [q for q in questions if q.pair_id is not None]
    elif question_set == "mmlu_standard":
        return [q for q in questions if q.variant is None or q.variant == "valid"]
    elif question_set == "pilot":
        return questions[:4]
    else:
        ids = {qid.strip() for qid in question_set.split(",")}
        return [q for q in questions if q.question_id in ids]


# ---------------------------------------------------------------------------
# Stratified sampling (reused from v1)
# ---------------------------------------------------------------------------

def stratified_sample(
    questions: list[QuestionRecord],
    n: int,
    seed: int = 42,
) -> list[QuestionRecord]:
    """Take a stratified random sample of n questions, balanced across subjects.

    Ensures every subject gets at least 1 question (if possible), then
    distributes the remaining slots proportionally. Within each subject,
    questions are randomly sampled. The final list is shuffled.

    Args:
        questions: Full list of questions (already filtered by question_set).
        n: Target sample size.
        seed: Random seed for reproducibility.

    Returns:
        A list of n QuestionRecord objects (or fewer if the pool is smaller).
    """
    if n >= len(questions):
        return questions

    rng = random.Random(seed)

    # Group questions by subject
    by_subject: dict[str, list[QuestionRecord]] = {}
    for q in questions:
        by_subject.setdefault(q.subject, []).append(q)

    num_subjects = len(by_subject)
    subjects = sorted(by_subject.keys())

    if n < num_subjects:
        chosen_subjects = rng.sample(subjects, n)
        sampled = []
        for subj in chosen_subjects:
            sampled.append(rng.choice(by_subject[subj]))
        rng.shuffle(sampled)
        return sampled

    base = n // num_subjects
    remainder = n % num_subjects
    rng.shuffle(subjects)

    sampled = []
    for i, subj in enumerate(subjects):
        pool = by_subject[subj]
        take = base + (1 if i < remainder else 0)
        take = min(take, len(pool))
        sampled.extend(rng.sample(pool, take))

    if len(sampled) < n:
        used_ids = {q.question_id for q in sampled}
        leftover = [q for q in questions if q.question_id not in used_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Core pipeline: run one question
# ---------------------------------------------------------------------------

def _logprobs_to_canonical_probs(
    canonical_logprobs: dict[int, float],
    num_choices: int = NUM_CHOICES,
) -> list[float]:
    """Convert raw canonical logprobs to a normalised probability vector.

    For each canonical position 0..3: exp(logprob) if present, exp(-30) if
    missing (effectively zero but avoids division issues). Then normalise
    to sum to 1.0.

    Args:
        canonical_logprobs: {canonical_index: raw_logprob} — may be missing some.
        num_choices: Number of answer choices (default 4).

    Returns:
        List of num_choices probabilities summing to 1.0, in canonical order.
    """
    FLOOR_LOGPROB = -30.0  # exp(-30) ≈ 9.4e-14, effectively zero
    raw_probs = []
    for i in range(num_choices):
        logprob = canonical_logprobs.get(i, FLOOR_LOGPROB)
        raw_probs.append(math.exp(logprob))
    total = sum(raw_probs)
    return [p / total for p in raw_probs]


# ---------------------------------------------------------------------------
# Query execution strategies: parallel (direct mode) and sequential (CoT)
# ---------------------------------------------------------------------------

# Number of concurrent requests for direct mode. Matches OLLAMA_NUM_PARALLEL
# default (which is typically 1-4). 3 is a good balance: enough to keep the
# GPU busy between round-trips, not so many that we overwhelm the server.
DIRECT_MODE_WORKERS = 3


def _process_single_query(
    query_num: int,
    query_text: str,
    paraphrase_index: int,
    permutation: list[int],
    question: QuestionRecord,
    client: OllamaClient,
    config: ExperimentConfig,
    num_choices: int,
) -> QueryResult | None:
    """Send one query and process the result. Returns None on failure."""
    try:
        raw_response, all_logprobs, thinking_text = client.send_query(
            question_text=query_text,
            choices=question.choices,
            answer_permutation=permutation,
        )
    except (ValueError, requests.exceptions.RequestException):
        return None

    try:
        display_logprobs, canonical_logprobs, display_answer, canonical_answer, answer_idx = (
            extract_answer_logprobs(
                all_logprobs, permutation, prompt_mode=config.prompt_mode,
            )
        )
    except ValueError:
        return None

    # For CoT modes, keep only the answer token's logprobs entry to reduce
    # storage overhead (CoT responses have 100-300 token positions, but we
    # only need the one where the model states its answer).
    # Direct mode already stores just 1 entry (num_predict=1).
    # The streaming chat endpoint may return raw lists instead of dicts
    # with a "top_logprobs" key — wrap if needed for Pydantic validation.
    if config.prompt_mode != "direct":
        entry = all_logprobs[answer_idx]
        if isinstance(entry, list):
            entry = {"top_logprobs": entry}
        all_logprobs = [entry]

    canonical_probs = _logprobs_to_canonical_probs(canonical_logprobs, num_choices)

    return QueryResult(
        query_number=query_num,
        paraphrase_index=paraphrase_index,
        query_text=query_text,
        answer_permutation=permutation,
        raw_response=raw_response,
        raw_logprobs=all_logprobs,
        display_letter_logprobs=display_logprobs,
        canonical_logprobs=canonical_logprobs,
        canonical_probs=canonical_probs,
        display_answer=display_answer,
        canonical_answer=canonical_answer,
        thinking_trace=thinking_text,
    )


def _run_queries_parallel(
    query_texts: list[str],
    paraphrase_indices: list[int],
    permutations: list[list[int]],
    question: QuestionRecord,
    client: OllamaClient,
    config: ExperimentConfig,
    num_choices: int,
    global_query_count: list[int],
) -> tuple[list[QueryResult], int, int, bool]:
    """Run all queries for a question in parallel using ThreadPoolExecutor.

    Used for direct mode only (non-streaming, single-token responses).
    Sends up to DIRECT_MODE_WORKERS concurrent requests to Ollama.

    Returns:
        Tuple of (query_log, missing_letter_count, extraction_failures, any_verbose).
    """
    query_log: list[QueryResult] = []
    missing_letter_count = 0
    extraction_failures = 0
    any_verbose = False

    # Submit all queries concurrently
    results_by_num: dict[int, QueryResult | None] = {}

    with ThreadPoolExecutor(max_workers=DIRECT_MODE_WORKERS) as executor:
        futures = {
            executor.submit(
                _process_single_query,
                qn, query_texts[qn], paraphrase_indices[qn], permutations[qn],
                question, client, config, num_choices,
            ): qn
            for qn in range(len(query_texts))
        }
        for future in as_completed(futures):
            qn = futures[future]
            try:
                results_by_num[qn] = future.result()
            except Exception:
                results_by_num[qn] = None

    # Process results in query order (preserves deterministic ordering)
    for query_num in range(len(query_texts)):
        result = results_by_num.get(query_num)
        if result is None:
            extraction_failures += 1
            continue

        # Track missing letters
        missing = [
            l for l in ANSWER_LETTERS[:num_choices]
            if l not in result.display_letter_logprobs
        ]
        if missing:
            missing_letter_count += 1

        # Verbose output for first 3 queries globally
        verbose = global_query_count[0] < 3
        global_query_count[0] += 1

        if verbose:
            any_verbose = True
            if missing:
                print(
                    f"\n    [WARN] Letters {missing} not in top_logprobs "
                    f"for {question.question_id} query {query_num}",
                    end="", flush=True,
                )
            _print_verbose_query(
                query_num, result.display_letter_logprobs,
                result.canonical_logprobs, result.canonical_probs,
                result.answer_permutation, result.display_answer,
                result.canonical_answer, config.prompt_mode,
                result.raw_response, result.thinking_trace,
            )

        query_log.append(result)

    return query_log, missing_letter_count, extraction_failures, any_verbose


def _run_queries_sequential(
    query_texts: list[str],
    paraphrase_indices: list[int],
    permutations: list[list[int]],
    question: QuestionRecord,
    client: OllamaClient,
    config: ExperimentConfig,
    num_choices: int,
    global_query_count: list[int],
) -> tuple[list[QueryResult], int, int, bool]:
    """Run all queries for a question sequentially.

    Used for CoT modes (streaming, longer generations) and single-query cases.

    Returns:
        Tuple of (query_log, missing_letter_count, extraction_failures, any_verbose).
    """
    query_log: list[QueryResult] = []
    missing_letter_count = 0
    extraction_failures = 0
    any_verbose = False
    consecutive_failures = 0
    max_consecutive_failures = 3

    for query_num in range(len(query_texts)):
        permutation = permutations[query_num]
        verbose = global_query_count[0] < 3

        # Query the model
        try:
            raw_response, all_logprobs, thinking_text = client.send_query(
                question_text=query_texts[query_num],
                choices=question.choices,
                answer_permutation=permutation,
            )
        except (ValueError, requests.exceptions.RequestException) as e:
            consecutive_failures += 1
            if verbose:
                print(f"\n    [SKIP] query {query_num}: {e}", end="", flush=True)
            if consecutive_failures >= max_consecutive_failures:
                if verbose:
                    print(
                        f"\n    [BAIL] giving up after {consecutive_failures} "
                        "consecutive failures",
                        end="", flush=True,
                    )
                break
            continue

        consecutive_failures = 0
        global_query_count[0] += 1

        # Extract answer logprobs
        try:
            display_logprobs, canonical_logprobs, display_answer, canonical_answer, answer_idx = (
                extract_answer_logprobs(
                    all_logprobs, permutation, prompt_mode=config.prompt_mode,
                )
            )
        except ValueError as e:
            extraction_failures += 1
            if verbose:
                first_token = "?"
                if all_logprobs:
                    top = _peek_top_token(all_logprobs[0])
                    if top:
                        first_token = top
                print(
                    f"\n    [WARN] {question.question_id} query {query_num}: {e} "
                    f"(first token: {first_token!r})",
                    end="", flush=True,
                )
            continue

        # Track missing letters
        missing_letters = [
            l for l in ANSWER_LETTERS[:num_choices]
            if l not in display_logprobs
        ]
        if missing_letters:
            missing_letter_count += 1
            if verbose:
                print(
                    f"\n    [WARN] Letters {missing_letters} not in top_logprobs "
                    f"for {question.question_id} query {query_num}",
                    end="", flush=True,
                )

        # For CoT modes, keep only the answer token's logprobs entry to reduce
        # storage overhead (CoT responses have 100-300 token positions, but we
        # only need the one where the model states its answer).
        # The streaming chat endpoint may return raw lists instead of dicts
        # with a "top_logprobs" key — wrap if needed for Pydantic validation.
        if config.prompt_mode != "direct":
            entry = all_logprobs[answer_idx]
            if isinstance(entry, list):
                entry = {"top_logprobs": entry}
            stored_logprobs = [entry]
        else:
            stored_logprobs = all_logprobs

        canonical_probs = _logprobs_to_canonical_probs(canonical_logprobs, num_choices)

        query_log.append(QueryResult(
            query_number=query_num,
            paraphrase_index=paraphrase_indices[query_num],
            query_text=query_texts[query_num],
            answer_permutation=permutation,
            raw_response=raw_response,
            raw_logprobs=stored_logprobs,
            display_letter_logprobs=display_logprobs,
            canonical_logprobs=canonical_logprobs,
            canonical_probs=canonical_probs,
            display_answer=display_answer,
            canonical_answer=canonical_answer,
            thinking_trace=thinking_text,
        ))

        if verbose:
            any_verbose = True
            _print_verbose_query(
                query_num, display_logprobs, canonical_logprobs,
                canonical_probs, permutation, display_answer,
                canonical_answer, config.prompt_mode,
                raw_response, thinking_text,
            )

    return query_log, missing_letter_count, extraction_failures, any_verbose


def run_single_question(
    question: QuestionRecord,
    paraphrases: list[ParaphraseRecord],
    client: OllamaClient,
    config: ExperimentConfig,
    rng_shuffle: random.Random,
    question_index: int,
    total_questions: int,
    global_query_count: list[int] | None = None,
) -> QuestionResult:
    """
    Run the logprob extraction pipeline for a single question.

    Query schedule depends on experimental condition:
    - Paraphrases ON: original + N paraphrases = N+1 queries, each text once.
    - Paraphrases OFF, shuffle ON: original × (N+1) permutations. Each gives
      a genuinely different distribution because answer positions change.
    - Paraphrases OFF, shuffle OFF: 1 query only. Identical prompt gives
      identical logprobs, so repetition is pointless.

    Args:
        question: The question to evaluate.
        paraphrases: Pre-generated paraphrases for this question.
        client: Ollama client for sending queries.
        config: Experiment configuration.
        rng_shuffle: RNG for answer permutations.
        question_index: 0-based index for progress display.
        total_questions: Total number of questions for progress display.
        global_query_count: Mutable [int] tracking queries across all questions.
            First 3 queries globally get verbose output. Pass None to disable.

    Returns:
        A QuestionResult with the full query log and normalised probabilities.
    """
    if global_query_count is None:
        global_query_count = [999]  # disable verbose by default
    num_choices = len(question.choices)
    query_log: list[QueryResult] = []

    # Build query schedule: list of (query_text, paraphrase_index) pairs.
    # The number of queries depends on what sources of variation are enabled.
    query_texts: list[str] = []
    paraphrase_indices: list[int] = []
    schedule_description = ""

    if config.use_paraphrases and paraphrases:
        # Original + paraphrases: each text queried exactly once
        query_texts.append(question.question_text)
        paraphrase_indices.append(-1)
        for idx, para in enumerate(paraphrases[:config.num_paraphrases]):
            query_texts.append(para.text)
            paraphrase_indices.append(idx)
        n_para = len(query_texts) - 1
        schedule_description = f"{len(query_texts)} queries (original + {n_para} paraphrases)"

    elif config.use_paraphrases and not paraphrases:
        # Paraphrase mode on but none available — fall back based on shuffle
        print(
            f"\n    [WARN] {question.question_id}: no paraphrases found",
            end="", flush=True,
        )
        if config.shuffle_choices:
            num_queries = config.num_paraphrases + 1
            query_texts = [question.question_text] * num_queries
            paraphrase_indices = [-1] * num_queries
            schedule_description = f"{num_queries} queries (original x {num_queries} permutations, no paraphrases)"
        else:
            query_texts = [question.question_text]
            paraphrase_indices = [-1]
            schedule_description = "1 query (no variation sources, no paraphrases)"

    elif not config.use_paraphrases and config.shuffle_choices:
        # No paraphrases but shuffle is on: each permutation gives a different distribution
        num_queries = config.num_paraphrases + 1
        query_texts = [question.question_text] * num_queries
        paraphrase_indices = [-1] * num_queries
        schedule_description = f"{num_queries} queries (original x {num_queries} permutations)"

    else:
        # No paraphrases, no shuffle: identical prompt → identical logprobs. Query once.
        query_texts = [question.question_text]
        paraphrase_indices = [-1]
        schedule_description = "1 query (no variation sources)"

    missing_letter_count = 0  # accumulate for compact summary
    extraction_failures = 0
    any_verbose = False  # track if any query in this question was verbose

    # Pre-generate all permutations deterministically (same seed = same result)
    permutations = [
        generate_permutation(num_choices, rng_shuffle, shuffle=config.shuffle_choices)
        for _ in range(len(query_texts))
    ]

    # Direct mode: parallelise queries (Ollama can batch up to OLLAMA_NUM_PARALLEL).
    # CoT modes: keep sequential (streaming + longer generations).
    if config.prompt_mode == "direct" and len(query_texts) > 1:
        query_log, missing_letter_count, extraction_failures, any_verbose = (
            _run_queries_parallel(
                query_texts, paraphrase_indices, permutations,
                question, client, config, num_choices,
                global_query_count,
            )
        )
    else:
        query_log, missing_letter_count, extraction_failures, any_verbose = (
            _run_queries_sequential(
                query_texts, paraphrase_indices, permutations,
                question, client, config, num_choices,
                global_query_count,
            )
        )

    # Aggregate: mean probability vector across all queries
    if query_log:
        all_probs = np.array([qr.canonical_probs for qr in query_log])
        mean_probs = all_probs.mean(axis=0).tolist()
        final_answer = int(np.argmax(mean_probs))
    else:
        mean_probs = [1.0 / num_choices] * num_choices  # uniform if no data
        final_answer = 0

    # Vote counts (how many times each answer was argmax of individual queries)
    answer_counter = Counter(qr.canonical_answer for qr in query_log)
    answer_counts = dict(answer_counter)

    # Correctness check
    correct = None
    if question.correct_answer is not None:
        correct = (final_answer == question.correct_answer)

    # Build compact warning summary for non-verbose questions
    warnings: list[str] = []
    if missing_letter_count > 0:
        warnings.append(f"{missing_letter_count} missing letters")
    if extraction_failures > 0:
        warnings.append(f"{extraction_failures} extraction failures")

    # Print progress line
    if query_log:
        _print_progress(
            question_index, total_questions, question.question_id,
            schedule_description, len(query_log), final_answer,
            mean_probs, correct, any_verbose, warnings,
        )

    return QuestionResult(
        question_id=question.question_id,
        query_log=query_log,
        num_queries=len(query_log),
        correct=correct,
        mean_probs=mean_probs,
        final_answer=final_answer,
        answer_counts=answer_counts,
    )


def _peek_top_token(logprobs_entry: dict) -> str | None:
    """Get the top token from a logprobs entry for debug logging."""
    from .query import _get_top_logprobs
    top = _get_top_logprobs(logprobs_entry)
    if top:
        return top[0].get("token", None)
    return None


def _print_verbose_query(
    query_num: int,
    display_logprobs: dict[str, float],
    canonical_logprobs: dict[int, float],
    canonical_probs: list[float],
    permutation: list[int],
    display_answer: str,
    canonical_answer: int,
    prompt_mode: str,
    raw_response: str,
    thinking_trace: str,
) -> None:
    """Print detailed per-query output for the first 3 questions."""
    # Display logprobs line
    display_str = " ".join(
        f"{letter}:{display_logprobs.get(letter, '---'):>7.2f}"
        if letter in display_logprobs
        else f"{letter}:  ---  "
        for letter in ANSWER_LETTERS
    )
    # Canonical logprobs line
    canon_str = " ".join(
        f"{idx}:{canonical_logprobs.get(idx, '---'):>7.2f}"
        if idx in canonical_logprobs
        else f"{idx}:  ---  "
        for idx in range(4)
    )
    # Normalised probabilities line
    prob_str = " ".join(f"{idx}:{p:.3f}" for idx, p in enumerate(canonical_probs))

    print(f"\n    Query {query_num}:", flush=True)
    print(f"      Display logprobs: {display_str}", flush=True)
    print(f"      Canonical logprobs: {canon_str}", flush=True)
    print(f"      Canonical probs:  {prob_str}", flush=True)
    print(
        f"      Permutation: {permutation} | "
        f"Answer: {display_answer} (canonical: {canonical_answer})",
        flush=True,
    )

    # For CoT modes, show first 200 chars of reasoning
    if prompt_mode in ("cot", "cot_structured"):
        reasoning_preview = raw_response[:200].replace("\n", " ")
        print(f"      Reasoning: {reasoning_preview}...", flush=True)
    if thinking_trace:
        think_preview = thinking_trace[:200].replace("\n", " ")
        print(f"      Thinking: {think_preview}...", flush=True)


def _print_progress(
    question_index: int,
    total_questions: int,
    question_id: str,
    schedule_description: str,
    num_queries: int,
    final_answer: int,
    mean_probs: list[float],
    correct: bool | None,
    verbose: bool,
    warnings: list[str] | None = None,
) -> None:
    """Print one-line progress for a completed question."""
    correct_str = ""
    if correct is not None:
        correct_str = f" | {'CORRECT' if correct else 'WRONG'}"

    # Show mean probability of winning answer as a confidence indicator
    confidence = mean_probs[final_answer] if mean_probs else 0.0

    # Append warnings inline
    warn_str = ""
    if warnings:
        warn_str = f" [{', '.join(warnings)}]"

    # Add a newline before the progress line if we just printed verbose output
    prefix = "\n" if verbose else ""
    print(
        f"{prefix}  [{question_index + 1}/{total_questions}] "
        f"{question_id} ... [{schedule_description}] | "
        f"answer: {ANSWER_LETTERS[final_answer]} ({confidence:.3f}){correct_str}{warn_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Per-question deterministic RNGs
# ---------------------------------------------------------------------------

def _make_question_rngs(seed: int, question_index: int) -> random.Random:
    """Create a deterministic RNG for a single question.

    Derives a per-question seed from the global seed and question index,
    so each question gets the same randomness regardless of execution order.
    """
    q_seed = seed + question_index * 1000
    return random.Random(q_seed)


# ---------------------------------------------------------------------------
# Run a full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    questions: list[QuestionRecord],
    paraphrases: dict[str, list[ParaphraseRecord]],
    output_dir: Path,
    seed: int = 42,
    completed_ids: set[str] | None = None,
    carried_over_results: list[QuestionResult] | None = None,
) -> ExperimentResult:
    """
    Run a complete experiment: iterate over all questions, query the model,
    record results, and save to JSON.

    Sequential execution only (v2 simplification — logprob queries are fast
    enough that parallelism isn't needed for 100-question runs).

    Args:
        config: Experiment configuration (loaded from YAML).
        questions: Filtered list of questions to evaluate.
        paraphrases: Paraphrase bank keyed by question_id.
        output_dir: Directory to save the result JSON file.
        seed: Random seed for reproducibility.
        completed_ids: Question IDs already completed (for --resume). These
            questions will be skipped in the main loop.
        carried_over_results: QuestionResult objects from the partial file.
            Included in the output so the final file contains everything.

    Returns:
        The complete ExperimentResult (also saved to disk).

    Raises:
        ConnectionError: If Ollama is not reachable.
    """
    # Verify Ollama is running
    client = OllamaClient(
        model=config.model,
        think=config.think,
        prompt_mode=config.prompt_mode,
        temperature=config.temperature,
    )
    if not client.check_connection():
        raise ConnectionError(
            f"Cannot reach Ollama at {client.base_url}. "
            "Make sure Ollama is running (ollama serve) and the model is pulled."
        )

    # Determine output file path
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    filename = f"{config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file = output_dir / filename

    # Resume support: filter out already-completed questions
    if completed_ids:
        original_count = len(questions)
        questions = [q for q in questions if q.question_id not in completed_ids]
        skipped = original_count - len(questions)
        print(f"Resuming: skipping {skipped} already-completed questions, "
              f"{len(questions)} remaining")
        print(flush=True)

    # Print experiment header — compute expected queries per question
    if config.use_paraphrases:
        queries_per_q = config.num_paraphrases + 1
        schedule_note = f"original + {config.num_paraphrases} paraphrases"
    elif config.shuffle_choices:
        queries_per_q = config.num_paraphrases + 1
        schedule_note = f"original x {queries_per_q} permutations"
    else:
        queries_per_q = 1
        schedule_note = "1 query (no variation sources)"

    print(f"Starting experiment: {config.run_name}")
    print(f"  Model: {config.model} | Think: {config.think} | "
          f"Prompt: {config.prompt_mode} | Shuffle: {config.shuffle_choices} | "
          f"Paraphrases: {config.use_paraphrases}")
    parallel_note = f" | Workers: {DIRECT_MODE_WORKERS}" if config.prompt_mode == "direct" else ""
    print(f"  Questions: {len(questions)} | Queries/question: {queries_per_q} ({schedule_note}){parallel_note}")
    print(f"  Temperature: {config.temperature} | Seed: {seed}")
    if config.use_paraphrases:
        questions_with_paras = sum(
            1 for q in questions if q.question_id in paraphrases
        )
        print(f"  Paraphrases loaded for {questions_with_paras}/{len(questions)} questions")
    print(f"  Writing results to: {output_file}")
    print(flush=True)

    # Run each question sequentially — seed with carried-over results from resume
    question_results: list[QuestionResult] = list(carried_over_results or [])
    global_query_count = [0]  # mutable counter — first 3 queries get verbose output
    writer = _IncrementalWriter()

    for idx, question in enumerate(questions):
        rng_shuffle = _make_question_rngs(seed, idx)
        question_paraphrases = paraphrases.get(question.question_id, [])

        result = run_single_question(
            question=question,
            paraphrases=question_paraphrases,
            client=client,
            config=config,
            rng_shuffle=rng_shuffle,
            question_index=idx,
            total_questions=len(questions),
            global_query_count=global_query_count,
        )
        question_results.append(result)

        # Incremental save every 20 questions (skip Pydantic re-validation
        # via model_construct — the data was just built by us, not user input)
        if (idx + 1) % 20 == 0:
            experiment_result = ExperimentResult.model_construct(
                run_name=config.run_name,
                config=config,
                timestamp=timestamp,
                question_results=question_results,
            )
            writer.write(output_file, experiment_result)

    # Final write with full validation as a safety check
    experiment_result = ExperimentResult(
        run_name=config.run_name,
        config=config,
        timestamp=timestamp,
        question_results=question_results,
    )
    writer.write(output_file, experiment_result)

    print(f"\nResults saved to {output_file}")
    _print_summary(config.run_name, question_results)

    return experiment_result


def _print_summary(run_name: str, results: list[QuestionResult]) -> None:
    """Print a summary table after an experiment run."""
    total = len(results)
    correct_count = sum(1 for r in results if r.correct is True)
    valid_count = sum(1 for r in results if r.correct is not None)
    avg_queries = sum(r.num_queries for r in results) / max(total, 1)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {run_name}")
    print(f"{'=' * 60}")
    print(f"  Total questions:  {total}")
    if valid_count > 0:
        print(f"  Accuracy:         {correct_count}/{valid_count} "
              f"({correct_count / valid_count:.1%})")
    print(f"  Avg queries used: {avg_queries:.1f}")
