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

    consecutive_failures = 0
    max_consecutive_failures = 3
    missing_letter_count = 0  # accumulate for compact summary
    extraction_failures = 0
    any_verbose = False  # track if any query in this question was verbose

    for query_num in range(len(query_texts)):
        # Generate answer permutation
        permutation = generate_permutation(
            num_choices, rng_shuffle, shuffle=config.shuffle_choices,
        )

        # Per-query verbose: only for the first 3 queries globally
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

        # Extract answer logprobs from the raw data
        try:
            display_logprobs, canonical_logprobs, display_answer, canonical_answer = (
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

        # Track missing answer letters
        missing_letters = [l for l in ANSWER_LETTERS[:num_choices] if l not in display_logprobs]
        if missing_letters:
            missing_letter_count += 1
            if verbose:
                print(
                    f"\n    [WARN] Letters {missing_letters} not in top_logprobs "
                    f"for {question.question_id} query {query_num}",
                    end="", flush=True,
                )

        # Compute normalised probabilities from raw logprobs
        canonical_probs = _logprobs_to_canonical_probs(canonical_logprobs, num_choices)

        # Store the result
        query_log.append(QueryResult(
            query_number=query_num,
            paraphrase_index=paraphrase_indices[query_num],
            query_text=query_texts[query_num],
            answer_permutation=permutation,
            raw_response=raw_response,
            raw_logprobs=all_logprobs,
            display_letter_logprobs=display_logprobs,
            canonical_logprobs=canonical_logprobs,
            canonical_probs=canonical_probs,
            display_answer=display_answer,
            canonical_answer=canonical_answer,
            thinking_trace=thinking_text,
        ))

        # Verbose output for first 3 queries globally
        if verbose:
            any_verbose = True
            _print_verbose_query(query_num, display_logprobs, canonical_logprobs,
                                 canonical_probs, permutation, display_answer,
                                 canonical_answer, config.prompt_mode,
                                 raw_response, thinking_text)

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
    print(f"  Questions: {len(questions)} | Queries/question: {queries_per_q} ({schedule_note})")
    print(f"  Temperature: {config.temperature} | Seed: {seed}")
    if config.use_paraphrases:
        questions_with_paras = sum(
            1 for q in questions if q.question_id in paraphrases
        )
        print(f"  Paraphrases loaded for {questions_with_paras}/{len(questions)} questions")
    print(f"  Writing results to: {output_file}")
    print(flush=True)

    # Run each question sequentially
    question_results: list[QuestionResult] = []
    global_query_count = [0]  # mutable counter — first 3 queries get verbose output

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

        # Incremental save after every question
        experiment_result = ExperimentResult(
            run_name=config.run_name,
            config=config,
            timestamp=timestamp,
            question_results=question_results,
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(experiment_result.model_dump_json(indent=2))

    # Final write
    experiment_result = ExperimentResult(
        run_name=config.run_name,
        config=config,
        timestamp=timestamp,
        question_results=question_results,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(experiment_result.model_dump_json(indent=2))

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
