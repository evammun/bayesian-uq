"""
Main experiment pipeline for Bayesian UQ.

Orchestrates the sequential sampling loop:
  1. Load questions and paraphrases from JSON files
  2. For each question: query the model with paraphrases + shuffled answers
  3. Update the Dirichlet posterior after each query
  4. Stop early when exceedance probability exceeds the confidence threshold
  5. Save all results to a timestamped JSON file

The key design principle: record everything exhaustively so you never need
to re-run a model query just because you want a different analysis.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .config import (
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
    update_posterior,
)
from .query import OllamaClient, generate_permutation


# ---------------------------------------------------------------------------
# Data loading helpers
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
# Question filtering
# ---------------------------------------------------------------------------

def filter_questions(
    questions: list[QuestionRecord],
    question_set: str,
) -> list[QuestionRecord]:
    """Filter the question database based on the experiment's question_set field.

    Supported values:
      - "all"            → every question
      - "broken_pairs"   → only questions that have a pair_id (valid + broken)
      - "mmlu_standard"  → MMLU questions without a broken variant
      - "pilot"          → first 4 questions (quick sanity check)
      - comma-separated  → specific question_ids (e.g. "mmlu_redux_0042,broken_mito_01")
    """
    if question_set == "all":
        return questions
    elif question_set == "broken_pairs":
        return [q for q in questions if q.pair_id is not None]
    elif question_set == "mmlu_standard":
        # Questions that are either standalone or the valid variant of a pair
        return [q for q in questions if q.variant is None or q.variant == "valid"]
    elif question_set == "pilot":
        return questions[:4]
    else:
        # Treat as a comma-separated list of question IDs
        ids = {qid.strip() for qid in question_set.split(",")}
        return [q for q in questions if q.question_id in ids]


# ---------------------------------------------------------------------------
# Stratified sampling
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
        seed: Random seed for reproducibility — same seed gives same sample.

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
    subjects = sorted(by_subject.keys())  # sort for determinism before shuffling

    # Guarantee at least 1 per subject, then distribute the remainder
    # If n < num_subjects, we can only pick n subjects (each gets 1)
    if n < num_subjects:
        chosen_subjects = rng.sample(subjects, n)
        sampled = []
        for subj in chosen_subjects:
            sampled.append(rng.choice(by_subject[subj]))
        rng.shuffle(sampled)
        return sampled

    # Everyone gets at least 1; distribute remainder across subjects
    base = n // num_subjects           # guaranteed per subject (at least 1)
    remainder = n % num_subjects       # extra slots to distribute

    # Shuffle subjects so the "extra slot" assignment is random
    rng.shuffle(subjects)

    sampled = []
    for i, subj in enumerate(subjects):
        pool = by_subject[subj]
        # This subject gets base + 1 if it's in the first 'remainder' slots
        take = base + (1 if i < remainder else 0)
        # Can't take more than available
        take = min(take, len(pool))
        sampled.extend(rng.sample(pool, take))

    # If we're still short (some subjects had fewer questions than needed),
    # fill from the unused pool
    if len(sampled) < n:
        used_ids = {q.question_id for q in sampled}
        leftover = [q for q in questions if q.question_id not in used_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    # Final shuffle so subjects aren't clustered
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Core pipeline: run one question
# ---------------------------------------------------------------------------

def run_single_question(
    question: QuestionRecord,
    paraphrases: list[ParaphraseRecord],
    client: OllamaClient,
    config: ExperimentConfig,
    rng_shuffle: random.Random,
    rng_mc: np.random.Generator,
) -> QuestionResult:
    """
    Run the full Bayesian UQ pipeline for a single question.

    Queries the model with the original question first, then paraphrases
    in a random order. After each response, updates the Dirichlet posterior
    and checks whether the exceedance probability exceeds the confidence
    threshold. Stops early if it does.

    If no paraphrases are available, falls back to re-asking the original
    question text each time with a different answer permutation. This is
    the "same-prompt resampling" baseline (Adaptive-Consistency style).
    All such queries are logged with paraphrase_index=-1.

    Args:
        question: The question to evaluate.
        paraphrases: Pre-generated paraphrases for this question (can be empty).
        client: Ollama client for sending queries.
        config: Experiment configuration (thresholds, budget, etc.).
        rng_shuffle: RNG for answer permutations and paraphrase ordering.
        rng_mc: RNG for Monte Carlo exceedance estimation.

    Returns:
        A QuestionResult with the full query log and final posterior state.
    """
    num_choices = len(question.choices)
    alpha = init_prior(num_choices)
    query_log: list[QueryResult] = []

    # Build query schedule: a list of (query_text, paraphrase_index) pairs.
    # paraphrase_index = -1 means "original question, no paraphrase".
    #
    # When use_paraphrases is True and paraphrases are available:
    #   - Pool = original question (index -1) + up to 10 paraphrases
    #   - Shuffle the pool order using the experiment RNG
    #   - If the budget exceeds pool size, cycle through the pool again
    #     (each cycle gets a fresh shuffle AND fresh answer permutations)
    #
    # When use_paraphrases is False (or no paraphrases found):
    #   - Every query uses the original question text with different answer
    #     permutations. This is the "same-prompt resampling" baseline.

    query_texts: list[str] = []
    paraphrase_indices: list[int] = []

    if config.use_paraphrases and paraphrases:
        # Cap at 10 paraphrases to keep the pool manageable
        max_paraphrases = 10
        available_paraphrases = paraphrases[:max_paraphrases]

        # Build the pool: original question + available paraphrases
        pool_texts = [question.question_text]
        pool_indices = [-1]
        for idx, para in enumerate(available_paraphrases):
            pool_texts.append(para.text)
            pool_indices.append(idx)

        # Fill the query schedule by cycling through the pool.
        # Each cycle gets a fresh shuffle so repeated paraphrases don't
        # appear in the same order. Each reuse still gets a different
        # answer permutation (generated in the main loop below).
        while len(query_texts) < config.max_queries_per_question:
            # Create a shuffled ordering of the pool for this cycle
            cycle_order = list(range(len(pool_texts)))
            rng_shuffle.shuffle(cycle_order)

            for pool_pos in cycle_order:
                if len(query_texts) >= config.max_queries_per_question:
                    break
                query_texts.append(pool_texts[pool_pos])
                paraphrase_indices.append(pool_indices[pool_pos])

    elif config.use_paraphrases and not paraphrases:
        # Paraphrase mode is on but this question has no paraphrases —
        # fall back to original question text and log a warning
        print(f"\n    [WARN] {question.question_id}: no paraphrases found, "
              "falling back to original question text", end="", flush=True)
        query_texts = [question.question_text] * config.max_queries_per_question
        paraphrase_indices = [-1] * config.max_queries_per_question

    else:
        # Paraphrase mode is off — repeat the original question each time.
        # Answer permutations still vary (if shuffle_choices is on).
        query_texts = [question.question_text] * config.max_queries_per_question
        paraphrase_indices = [-1] * config.max_queries_per_question

    # Cap at the configured budget
    max_queries = config.max_queries_per_question
    stopped_early = False
    consecutive_failures = 0
    max_consecutive_failures = 3  # Give up on this question after 3 failures in a row

    for query_num in range(max_queries):
        # Generate answer permutation (identity if shuffle_choices is off)
        permutation = generate_permutation(
            num_choices, rng_shuffle, shuffle=config.shuffle_choices,
        )

        # Query the model — skip this query if the model returns garbage
        try:
            raw_response, canonical_answer, thinking_text = client.send_query(
                question_text=query_texts[query_num],
                choices=question.choices,
                answer_permutation=permutation,
            )
        except (ValueError, Exception) as e:
            consecutive_failures += 1
            print(f"\n    [SKIP] query {query_num}: {e}", end="", flush=True)
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n    [BAIL] giving up after {consecutive_failures} "
                      "consecutive failures", end="", flush=True)
                break
            continue

        # Reset the failure counter on success
        consecutive_failures = 0

        # Update the Dirichlet posterior
        alpha = update_posterior(alpha, canonical_answer)

        # Compute posterior statistics after this update
        exc = exceedance_probability(alpha, config.monte_carlo_samples, rng_mc)
        ent = posterior_entropy(alpha)

        # Record everything about this query
        query_log.append(QueryResult(
            query_number=query_num,
            paraphrase_index=paraphrase_indices[query_num],
            answer_permutation=permutation,
            raw_model_response=raw_response,
            thinking_trace=thinking_text,
            canonical_answer=canonical_answer,
            alpha_after=alpha.tolist(),
            exceedance_after=exc,
            entropy_after=ent,
        ))

        # Check stopping criterion
        if exc >= config.confidence_threshold:
            stopped_early = True
            break

    # Final answer = posterior mode (answer with the highest pseudo-count)
    final_answer = int(np.argmax(alpha))

    # Check correctness (only meaningful for valid questions with a known answer)
    correct = None
    if question.correct_answer is not None:
        correct = (final_answer == question.correct_answer)

    # Handle the case where all queries failed (empty log)
    if query_log:
        final_exc = query_log[-1].exceedance_after
        final_ent = query_log[-1].entropy_after
    else:
        final_exc = 1.0 / num_choices  # uniform — no information
        final_ent = posterior_entropy(alpha)

    return QuestionResult(
        question_id=question.question_id,
        query_log=query_log,
        final_answer=final_answer,
        final_alpha=alpha.tolist(),
        final_exceedance=final_exc,
        final_entropy=final_ent,
        queries_used=len(query_log),
        stopped_early=stopped_early,
        correct=correct,
    )


# ---------------------------------------------------------------------------
# Per-question deterministic RNGs
# ---------------------------------------------------------------------------

def _make_question_rngs(
    seed: int, question_index: int,
) -> tuple[random.Random, np.random.Generator]:
    """Create deterministic RNGs for a single question.

    Derives a per-question seed from the global seed and question index,
    so each question gets the same randomness regardless of execution
    order or degree of parallelism.
    """
    q_seed = seed + question_index * 1000
    return random.Random(q_seed), np.random.default_rng(q_seed)


# ---------------------------------------------------------------------------
# Run a full experiment (parallel-capable)
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

    Supports parallel execution via config.parallel_workers. Each worker
    gets its own OllamaClient (separate HTTP session) and per-question
    deterministic RNGs so results are reproducible regardless of execution
    order.

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
    # Verify Ollama is running before starting the experiment
    probe_client = OllamaClient(
        model=config.model,
        think=config.think,
        prompt_mode=config.prompt_mode,
        temperature=config.temperature,
    )
    if not probe_client.check_connection():
        raise ConnectionError(
            f"Cannot reach Ollama at {probe_client.base_url}. "
            "Make sure Ollama is running (ollama serve) and the model is pulled."
        )

    # Determine output file path up front so we can write incrementally
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    filename = f"{config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file = output_dir / filename

    workers = config.parallel_workers
    print(f"Starting experiment: {config.run_name}")
    print(f"  Model: {config.model} | Think: {config.think} | Prompt: {config.prompt_mode} | Shuffle: {config.shuffle_choices} | Paraphrases: {config.use_paraphrases}")
    print(f"  Questions: {len(questions)} | Max queries/question: {config.max_queries_per_question}")
    print(f"  Confidence threshold: {config.confidence_threshold} | Workers: {workers}")
    print(f"  Writing results to: {output_file}")
    print()

    # Shared state for thread-safe progress and incremental writes
    results: list[QuestionResult | None] = [None] * len(questions)
    write_lock = threading.Lock()
    print_lock = threading.Lock()
    completed_count = [0]  # mutable counter shared across threads

    def _run_one(idx: int) -> tuple[int, QuestionResult]:
        """Process a single question (called from thread pool)."""
        question = questions[idx]

        # Each worker gets its own client (separate HTTP session)
        client = OllamaClient(
            model=config.model,
            think=config.think,
            prompt_mode=config.prompt_mode,
            temperature=config.temperature,
        )

        # Per-question deterministic RNGs (same seed → same result)
        rng_shuffle, rng_mc = _make_question_rngs(seed, idx)

        # Get paraphrases for this question
        question_paraphrases = paraphrases.get(question.question_id, [])

        result = run_single_question(
            question=question,
            paraphrases=question_paraphrases,
            client=client,
            config=config,
            rng_shuffle=rng_shuffle,
            rng_mc=rng_mc,
        )
        return idx, result

    def _on_complete(idx: int, result: QuestionResult) -> None:
        """Handle a completed question: update results, print, write JSON."""
        question = questions[idx]
        results[idx] = result
        completed_count[0] += 1

        # Build status line
        status = "EARLY STOP" if result.stopped_early else "FULL BUDGET"
        correct_str = ""
        if result.correct is not None:
            correct_str = f" | {'CORRECT' if result.correct else 'WRONG'}"
        para_note = ""
        if not paraphrases.get(question.question_id):
            para_note = "[no paraphrases] "

        # Thread-safe progress print
        with print_lock:
            print(
                f"  [{completed_count[0]}/{len(questions)}] "
                f"{question.question_id} ... {para_note}"
                f"{status} ({result.queries_used}q, "
                f"exc={result.final_exceedance:.3f}){correct_str}"
            )

        # Thread-safe incremental JSON write (all completed results in
        # original question order so the dashboard can track progress)
        with write_lock:
            completed_results = [r for r in results if r is not None]
            experiment_result = ExperimentResult(
                run_name=config.run_name,
                config=config,
                timestamp=timestamp,
                question_results=completed_results,
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(experiment_result.model_dump_json(indent=2))

    # Dispatch all questions through the thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, idx): idx
            for idx in range(len(questions))
        }
        for future in as_completed(futures):
            idx, result = future.result()
            _on_complete(idx, result)

    # Build the final result with all questions in original order
    final_results = [r for r in results if r is not None]
    experiment_result = ExperimentResult(
        run_name=config.run_name,
        config=config,
        timestamp=timestamp,
        question_results=final_results,
    )

    # Final write to ensure complete file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(experiment_result.model_dump_json(indent=2))

    print(f"\nResults saved to {output_file}")

    # Print summary
    _print_summary(config.run_name, final_results)

    return experiment_result


def _print_summary(run_name: str, results: list[QuestionResult]) -> None:
    """Print a summary table after an experiment run."""
    total = len(results)
    stopped_early = sum(1 for r in results if r.stopped_early)
    correct_count = sum(1 for r in results if r.correct is True)
    valid_count = sum(1 for r in results if r.correct is not None)
    avg_queries = np.mean([r.queries_used for r in results])

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {run_name}")
    print(f"{'='*60}")
    print(f"  Total questions:  {total}")
    if valid_count > 0:
        print(f"  Accuracy:         {correct_count}/{valid_count} ({correct_count/valid_count:.1%})")
    print(f"  Stopped early:    {stopped_early}/{total} ({stopped_early/total:.1%})")
    print(f"  Avg queries used: {avg_queries:.1f}")
