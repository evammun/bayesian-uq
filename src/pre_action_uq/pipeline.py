"""
Experiment pipeline for Pre-Action UQ with QuALITY dataset.

Adapted from v2 pipeline.py with changes:
  - QuALITY JSONL loading (article + questions format)
  - Context condition construction (sufficient/insufficient)
  - Prompt building includes article passage
  - Uses LlamaCppClient instead of OllamaClient
  - All queries run sequentially (GPU is the bottleneck, not CPU)
  - Chat template applied for all modes (Qwen3 is chat-finetuned)
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import (
    ANSWER_LETTERS,
    NUM_CHOICES,
    ExperimentConfig,
    ExperimentResult,
    QueryResult,
    QuestionResult,
)
from .inference import LlamaCppClient, extract_answer_logprobs


# Module-level paraphrase bank (loaded once per experiment)
_paraphrase_bank: dict = {}


# ---------------------------------------------------------------------------
# Atomic incremental file writes (ported from v2 — O(1) per save)
# ---------------------------------------------------------------------------

class _IncrementalWriter:
    """Efficiently writes ExperimentResult JSON by appending, not rewriting.

    On first call, writes the full JSON header + opening bracket.
    On subsequent calls, seeks to just before the closing ']\\n}' and
    appends only new QuestionResult entries. Each save is O(size_of_one_result).
    """

    def __init__(self) -> None:
        self._n_written: int = 0
        self._tail_len: int = 0

    @staticmethod
    def _indent_result(json_str: str) -> str:
        """Indent a result JSON block by 4 spaces."""
        lines = json_str.split("\n")
        return "\n".join("    " + line for line in lines)

    def finalize(self, path: Path, completed_at: str) -> None:
        """Patch the completed_at field in the existing JSON file."""
        if self._n_written == 0:
            return
        raw = path.read_text(encoding="utf-8")
        raw = raw.replace('"completed_at": null', f'"completed_at": "{completed_at}"', 1)
        path.write_text(raw, encoding="utf-8")

    def write(self, path: Path, data: ExperimentResult) -> None:
        """Append new question results to the JSON file on disk."""
        new_results = data.question_results[self._n_written:]
        if not new_results:
            return

        if self._n_written == 0:
            # First write: build full file from scratch
            shell = data.model_copy(update={"question_results": []})
            shell_json = shell.model_dump_json(indent=2)
            marker = '"question_results": []'
            idx = shell_json.find(marker)
            if idx == -1:
                raise ValueError("Could not find question_results in serialized JSON")
            header_end = idx + len('"question_results": ')
            header = shell_json[:header_end]

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
            # Subsequent writes: seek to overwrite closing tail, append new
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
# QuALITY dataset loading
# ---------------------------------------------------------------------------

class QualityQuestion:
    """A single question extracted from the QuALITY JSONL dataset.

    Flattened from the nested article→questions structure for pipeline use.
    """

    def __init__(
        self,
        question_unique_id: str,
        article_id: str,
        question_text: str,
        options: list[str],
        correct_answer: int,  # 0-indexed
        difficult: bool,
        article_text: str,
        source: str,
        topic: str,
        title: str,
    ):
        self.question_unique_id = question_unique_id
        self.article_id = article_id
        self.question_text = question_text
        self.options = options
        self.correct_answer = correct_answer
        self.difficult = difficult
        self.article_text = article_text
        self.source = source
        self.topic = topic
        self.title = title


def load_quality_dataset(path: Path) -> list[QualityQuestion]:
    """Load a QuALITY JSONL file and flatten into individual questions.

    Each line in the JSONL is one article with multiple questions.
    gold_label is 1-indexed in the dataset — we convert to 0-indexed here.

    Args:
        path: Path to the JSONL file (e.g. data/quality_all.jsonl).

    Returns:
        List of QualityQuestion objects, one per question.
    """
    questions: list[QualityQuestion] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            article = json.loads(line.strip())
            article_id = article["article_id"]
            article_text = article["article"]
            source = article.get("source", "unknown")
            topic = article.get("topic", "unknown")
            title = article.get("title", "")

            for q in article["questions"]:
                # gold_label is 1-indexed in QuALITY — convert to 0-indexed
                correct_answer = q["gold_label"] - 1
                difficult = bool(q.get("difficult", 0))

                questions.append(QualityQuestion(
                    question_unique_id=q["question_unique_id"],
                    article_id=article_id,
                    question_text=q["question"],
                    options=q["options"],
                    correct_answer=correct_answer,
                    difficult=difficult,
                    article_text=article_text,
                    source=source,
                    topic=topic,
                    title=title,
                ))

    return questions


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    question_text: str,
    options: list[str],
    article_text: str,
    answer_permutation: list[int],
    prompt_mode: str = "direct",
    think: bool = False,
) -> str:
    """Construct an MCQ prompt with article context.

    For direct mode: ends with "Answer:" for single-token extraction.
    For CoT mode: adds reasoning instructions that explicitly forbid
    naming the answer letter in the reasoning (prevents leaking the
    answer to Pass 2 of the two-pass pipeline).
    For direct+think: tells the model to answer with a single letter
    only — all reasoning goes in the <think> block, visible output
    must be minimal to prevent answer leaking to Pass 2.

    Args:
        question_text: The question stem.
        options: Answer texts in canonical order (index 0-3).
        article_text: The passage to read.
        answer_permutation: Display position → canonical index.
        prompt_mode: "direct" or "cot".

    Returns:
        Formatted prompt ending with "Answer:".
    """
    # Build display choices using the permutation
    choice_lines = []
    for display_pos, canonical_idx in enumerate(answer_permutation):
        letter = ANSWER_LETTERS[display_pos]
        choice_lines.append(f"{letter}) {options[canonical_idx]}")

    if prompt_mode == "cot":
        # Natural CoT: model reasons freely, no anti-leak constraint.
        # Logprobs are extracted conditioned on the full reasoning.
        return (
            "Read the following passage and answer the question.\n\n"
            f"Passage:\n{article_text}\n\n"
            f"Question: {question_text}\n\n"
            + "\n".join(choice_lines)
            + "\n\n"
            "BE CONCISE. 3-4 bullet points of reasoning only.\n\n"
            "End with: Answer: X"
        )
    elif think:
        # Think mode: reasoning goes in <think> block via /think tag.
        # Logprobs are extracted conditioned on the full output
        # (including think block).
        return (
            "Read the following passage and answer the question.\n\n"
            f"Passage:\n{article_text}\n\n"
            f"Question: {question_text}\n\n"
            + "\n".join(choice_lines)
            + "\n\n"
            "Answer with a single letter.\n\n"
            "End with: Answer: X"
        )
    else:
        return (
            "Read the following passage and answer the question.\n\n"
            f"Passage:\n{article_text}\n\n"
            f"Question: {question_text}\n\n"
            + "\n".join(choice_lines)
            + "\n\nAnswer:"
        )


# ---------------------------------------------------------------------------
# Logprob → probability conversion (from v2)
# ---------------------------------------------------------------------------

def _logprobs_to_canonical_probs(
    canonical_logprobs: dict[int, float],
    num_choices: int = NUM_CHOICES,
) -> list[float]:
    """Convert raw canonical logprobs to a normalised probability vector.

    For each position 0..3: exp(logprob) if present, exp(-30) if missing.
    Then normalise to sum to 1.0.
    """
    FLOOR_LOGPROB = -30.0
    raw_probs = []
    for i in range(num_choices):
        logprob = canonical_logprobs.get(i, FLOOR_LOGPROB)
        raw_probs.append(math.exp(logprob))
    total = sum(raw_probs)
    return [p / total for p in raw_probs]


# ---------------------------------------------------------------------------
# Answer permutation generation (from v2)
# ---------------------------------------------------------------------------

def generate_permutation(
    num_choices: int = NUM_CHOICES,
    rng: random.Random | None = None,
) -> list[int]:
    """Generate a random answer permutation.

    Returns a list where permutation[display_pos] = canonical_index.
    """
    perm = list(range(num_choices))
    if rng is None:
        rng = random.Random()
    rng.shuffle(perm)
    return perm


# ---------------------------------------------------------------------------
# Insufficient context construction
# ---------------------------------------------------------------------------

def pick_swap_article(
    question: QualityQuestion,
    all_questions: list[QualityQuestion],
    rng: random.Random,
) -> tuple[str, str]:
    """Pick a different article's text for the insufficient condition.

    Prefers same topic if available, otherwise random. Returns
    (swap_article_text, swap_article_id).
    """
    # Collect unique articles
    articles: dict[str, QualityQuestion] = {}
    for q in all_questions:
        if q.article_id != question.article_id and q.article_id not in articles:
            articles[q.article_id] = q

    # Try same topic first
    same_topic = [q for q in articles.values() if q.topic == question.topic]
    if same_topic:
        swap = rng.choice(same_topic)
    else:
        swap = rng.choice(list(articles.values()))

    return swap.article_text, swap.article_id


# ---------------------------------------------------------------------------
# Query execution (sequential — GPU is the bottleneck)
# ---------------------------------------------------------------------------

def _process_single_query(
    query_num: int,
    prompt: str,
    paraphrase_index: int,
    permutation: list[int],
    client: LlamaCppClient,
    config: ExperimentConfig,
    question_text_used: str = "",
    paraphrase_category: str = "",
) -> QueryResult | None:
    """Send one query and process the result. Returns None on failure."""
    try:
        if config.think:
            result = client.generate_think(
                prompt=prompt,
                max_tokens=4096,
                temperature=config.temperature,
            )
        elif config.prompt_mode == "cot":
            result = client.generate_cot(
                prompt=prompt,
                max_tokens=2048,
                temperature=config.temperature,
            )
        else:
            # Direct mode — single-token logprob extraction
            result = client.generate_with_logprobs(
                prompt=prompt,
                think=False,
            )
    except Exception:
        return None

    # Extract answer logprobs
    extraction_mode = "cot" if (config.prompt_mode == "cot" or config.think) else "direct"
    try:
        display_lp, canonical_lp, display_ans, canonical_ans, answer_idx = (
            extract_answer_logprobs(
                result["logprobs"], permutation, mode=extraction_mode,
            )
        )
    except ValueError:
        return None

    canonical_probs = _logprobs_to_canonical_probs(canonical_lp)

    # For CoT, keep only the answer token's logprobs
    stored_logprobs = result["logprobs"]
    if extraction_mode == "cot" and len(stored_logprobs) > 1:
        stored_logprobs = [stored_logprobs[answer_idx]]

    # Store passage snippet (first 80 chars) for verification, not full article
    stored_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt

    # Map Pass 1 answer (CoT) to canonical index
    pass1_answer = result.get("pass1_answer", "")
    pass1_canonical = -1
    if pass1_answer and pass1_answer in ANSWER_LETTERS:
        display_pos = ANSWER_LETTERS.index(pass1_answer)
        pass1_canonical = permutation[display_pos]

    return QueryResult(
        query_number=query_num,
        paraphrase_index=paraphrase_index,
        paraphrase_category=paraphrase_category,
        query_text=stored_prompt,
        question_text_used=question_text_used,
        answer_permutation=permutation,
        raw_response=result["response_text"],
        raw_logprobs=stored_logprobs,
        display_letter_logprobs=display_lp,
        canonical_logprobs=canonical_lp,
        canonical_probs=canonical_probs,
        display_answer=display_ans,
        canonical_answer=canonical_ans,
        thinking_trace=result.get("thinking_trace", ""),
        pass1_answer=pass1_answer,
        pass1_canonical_answer=pass1_canonical,
        # Per-query timing and token counts from the inference layer
        inference_time_s=result.get("inference_time_s"),
        prompt_tokens=result.get("prompt_tokens"),
        output_tokens=result.get("output_tokens"),
    )


def _run_queries(
    prompts: list[str],
    paraphrase_indices: list[int],
    permutations: list[list[int]],
    client: LlamaCppClient,
    config: ExperimentConfig,
    global_query_count: list[int],
    question_texts_used: list[str] | None = None,
    paraphrase_categories: list[str] | None = None,
) -> tuple[list[QueryResult], int]:
    """Run all queries for a question sequentially."""
    query_log: list[QueryResult] = []
    extraction_failures = 0

    for qn in range(len(prompts)):
        q_text = question_texts_used[qn] if question_texts_used else ""
        q_cat = paraphrase_categories[qn] if paraphrase_categories else ""
        result = _process_single_query(
            qn, prompts[qn], paraphrase_indices[qn], permutations[qn],
            client, config, question_text_used=q_text, paraphrase_category=q_cat,
        )
        if result is None:
            extraction_failures += 1
            continue
        global_query_count[0] += 1
        query_log.append(result)

    return query_log, extraction_failures


# ---------------------------------------------------------------------------
# Run one question
# ---------------------------------------------------------------------------

def run_single_question(
    client: LlamaCppClient,
    question: QualityQuestion,
    config: ExperimentConfig,
    rng: random.Random,
    all_questions: list[QualityQuestion],
    question_index: int,
    total_questions: int,
    global_query_count: list[int],
) -> QuestionResult:
    """Run the logprob extraction pipeline for a single QuALITY question.

    Query schedule: 1 original + num_permutations answer permutations.
    Each gets a different random answer ordering.

    Args:
        client: LlamaCppClient for inference.
        question: The QuALITY question to evaluate.
        config: Experiment configuration.
        rng: Deterministic RNG for this question.
        all_questions: Full question list (for insufficient context swaps).
        question_index: 0-based index for progress.
        total_questions: Total questions for progress display.
        global_query_count: Mutable [int] counter for verbose output.

    Returns:
        QuestionResult with query log and aggregated probabilities.
    """
    num_choices = len(question.options)
    num_queries = config.num_permutations

    # Determine article context
    swapped_article_id = None
    if config.context_condition == "sufficient":
        article_text = question.article_text
    else:
        article_text, swapped_article_id = pick_swap_article(
            question, all_questions, rng,
        )

    # Build query schedule
    prompts: list[str] = []
    paraphrase_indices: list[int] = []
    permutations: list[list[int]] = []
    question_texts_used: list[str] = []
    paraphrase_categories: list[str] = []

    # Load paraphrases for this question if enabled
    para_texts: list[tuple[str, str]] = []  # (text, category)
    if config.use_paraphrases and _paraphrase_bank:
        entry = _paraphrase_bank.get(question.question_unique_id, {})
        for cat in ["A", "B", "C"]:
            for p in entry.get(cat, []):
                para_texts.append((p, cat))

    for qn in range(num_queries):
        if config.shuffle_options:
            perm = generate_permutation(num_choices, rng)
        else:
            perm = list(range(num_choices))  # identity permutation

        # Use paraphrase if available, otherwise original
        if para_texts and qn < len(para_texts):
            q_text, q_cat = para_texts[qn]
        else:
            q_text = question.question_text
            q_cat = ""

        prompt = build_prompt(
            q_text,
            question.options,
            article_text,
            perm,
            prompt_mode=config.prompt_mode,
            think=config.think,
        )
        prompts.append(prompt)
        paraphrase_indices.append(qn)
        permutations.append(perm)
        question_texts_used.append(q_text)
        paraphrase_categories.append(q_cat)

    # Execute queries sequentially (GPU is the bottleneck)
    query_log, failures = _run_queries(
        prompts, paraphrase_indices, permutations,
        client, config, global_query_count,
        question_texts_used=question_texts_used or [question.question_text] * len(prompts),
        paraphrase_categories=paraphrase_categories or [""] * len(prompts),
    )

    # Handle skipped questions (all queries failed, e.g. n_ctx overflow)
    if not query_log:
        print(
            f"  [{question_index + 1}/{total_questions}] "
            f"{question.question_unique_id[:30]} | "
            f"[SKIPPED - all queries failed]",
            flush=True,
        )
        return QuestionResult(
            question_id=question.question_unique_id,
            question_unique_id=question.question_unique_id,
            article_id=question.article_id,
            question_text=question.question_text,
            options=question.options,
            correct_answer=question.correct_answer,
            difficult=question.difficult,
            context_condition=config.context_condition,
            swapped_article_id=swapped_article_id,
            query_log=[],
            num_queries=0,
            correct=None,
            skipped=True,
            mean_probs=[],
            final_answer=-1,
            answer_counts={},
        )

    # Aggregate probabilities
    all_probs = np.array([qr.canonical_probs for qr in query_log])
    mean_probs = all_probs.mean(axis=0).tolist()
    final_answer = int(np.argmax(mean_probs))

    # Vote counts
    answer_counter = Counter(qr.canonical_answer for qr in query_log)
    answer_counts = dict(answer_counter)

    # Correctness
    correct = (final_answer == question.correct_answer)

    # Progress line
    confidence = mean_probs[final_answer] if mean_probs else 0.0
    correct_str = "CORRECT" if correct else "WRONG"
    warn_str = f" [{failures} failures]" if failures > 0 else ""
    diff_str = " [HARD]" if question.difficult else ""
    print(
        f"  [{question_index + 1}/{total_questions}] "
        f"{question.question_unique_id[:30]} | "
        f"answer: {ANSWER_LETTERS[final_answer]} ({confidence:.3f}) | "
        f"{correct_str}{diff_str}{warn_str}",
        flush=True,
    )

    return QuestionResult(
        question_id=question.question_unique_id,
        question_unique_id=question.question_unique_id,
        article_id=question.article_id,
        question_text=question.question_text,
        options=question.options,
        correct_answer=question.correct_answer,
        difficult=question.difficult,
        context_condition=config.context_condition,
        swapped_article_id=swapped_article_id,
        query_log=query_log,
        num_queries=len(query_log),
        correct=correct,
        skipped=False,
        mean_probs=mean_probs,
        final_answer=final_answer,
        answer_counts=answer_counts,
    )


# ---------------------------------------------------------------------------
# Per-question deterministic RNGs
# ---------------------------------------------------------------------------

def _make_question_rng(seed: int, question_index: int) -> random.Random:
    """Create a deterministic RNG for a single question."""
    return random.Random(seed + question_index * 1000)


# ---------------------------------------------------------------------------
# Run a full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    completed_ids: set[str] | None = None,
    carried_over_results: list[QuestionResult] | None = None,
    resume_file: Path | None = None,
) -> ExperimentResult:
    """Run a complete experiment on QuALITY questions.

    Args:
        config: Experiment configuration.
        output_dir: Directory to save results.
        completed_ids: Question IDs already done (for --resume).
        carried_over_results: QuestionResult objects from partial file.
        resume_file: If resuming, write back to this file instead of creating a new one.

    Returns:
        Complete ExperimentResult (also saved to disk).
    """
    # Load dataset
    dataset_path = Path(config.dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent.parent / dataset_path
    print(f"Loading dataset: {dataset_path}")
    all_questions = load_quality_dataset(dataset_path)
    print(f"  {len(all_questions)} questions loaded")

    # Load paraphrase bank if enabled
    global _paraphrase_bank
    if config.use_paraphrases:
        para_path = Path(config.paraphrases_file)
        if not para_path.is_absolute():
            para_path = Path(__file__).resolve().parent.parent.parent / para_path
        print(f"Loading paraphrases: {para_path}")
        _paraphrase_bank = json.loads(para_path.read_text(encoding="utf-8"))
        print(f"  {len(_paraphrase_bank)} questions with paraphrases")
    else:
        _paraphrase_bank = {}

    # Apply max_questions limit
    if config.max_questions and config.max_questions < len(all_questions):
        rng_sample = random.Random(config.seed)
        rng_sample.shuffle(all_questions)
        all_questions_full = list(all_questions)  # keep full list for swaps
        all_questions = all_questions[:config.max_questions]
        print(f"  Sampled {config.max_questions} questions")
    else:
        all_questions_full = all_questions

    # Load model
    from .inference import LlamaCppClient, find_model_path

    model_path = config.model_path
    if model_path == "auto":
        found = find_model_path(config.model_name)
        if found is None:
            raise FileNotFoundError(
                "No GGUF model found. Searched: UQ_MODEL_PATH env var, "
                "/workspace/models/, <project>/models/, ~/models/, "
                "~/.ollama/models/blobs/. Set UQ_MODEL_PATH or copy "
                "the GGUF to one of these locations."
            )
        model_path = str(found)

    print(f"Loading model: {Path(model_path).name[:50]}...")
    # Pass model_family from config so the correct chat template is used.
    # "auto" (the default) detects family from the GGUF filename.
    client = LlamaCppClient(
        model_path=model_path,
        n_ctx=config.n_ctx,
        seed=config.seed,
        verbose=False,
        model_family=config.model_family,
    )
    print(f"  Model loaded in {client.load_time:.1f}s")

    # Output file — reuse the same file on resume so we don't create duplicates
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    if resume_file is not None:
        output_file = resume_file
    else:
        filename = f"{config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = output_dir / filename

    # Build stable index map BEFORE filtering — so each question always gets
    # the same RNG seed regardless of whether we're resuming or running fresh.
    _original_index = {q.question_unique_id: i for i, q in enumerate(all_questions)}

    # Resume support
    if completed_ids:
        original_count = len(all_questions)
        all_questions = [q for q in all_questions if q.question_unique_id not in completed_ids]
        print(f"Resuming: {original_count - len(all_questions)} done, {len(all_questions)} remaining")

    # Print experiment header
    queries_per_q = config.num_permutations
    print(f"\nStarting experiment: {config.run_name}")
    print(f"  Model: {Path(model_path).name[:40]} | Think: {config.think} | "
          f"Prompt: {config.prompt_mode}")
    print(f"  Context: {config.context_condition} | Questions: {len(all_questions)} | "
          f"Queries/q: {queries_per_q}")
    print(f"  Temperature: {config.temperature} | Seed: {config.seed}")
    print(f"  Output: {output_file}")
    print(flush=True)

    # Run — seed with carried-over results, track IDs to prevent duplicates
    question_results: list[QuestionResult] = list(carried_over_results or [])
    completed_ids_set = set(completed_ids or [])
    global_query_count = [0]
    writer = _IncrementalWriter()

    for idx, question in enumerate(all_questions):
        # Double-check: skip if already in carried-over results (belt + suspenders)
        if question.question_unique_id in completed_ids_set:
            continue

        rng = _make_question_rng(config.seed, _original_index[question.question_unique_id])

        result = run_single_question(
            client=client,
            question=question,
            config=config,
            rng=rng,
            all_questions=all_questions_full,
            question_index=idx,
            total_questions=len(all_questions),
            global_query_count=global_query_count,
        )
        question_results.append(result)
        completed_ids_set.add(question.question_unique_id)

        # Incremental save: first question immediately, then every 10
        if idx == 0 or (idx + 1) % 10 == 0:
            experiment_result = ExperimentResult.model_construct(
                run_name=config.run_name,
                config=config,
                timestamp=timestamp,
                question_results=question_results,
            )
            writer.write(output_file, experiment_result)

    # Final save — stamp completion time
    completed_at = datetime.now(timezone.utc).isoformat()
    writer.finalize(output_file, completed_at)
    experiment_result = ExperimentResult(
        run_name=config.run_name,
        config=config,
        timestamp=timestamp,
        completed_at=completed_at,
        question_results=question_results,
    )

    # Summary
    total = len(question_results)
    correct_count = sum(1 for r in question_results if r.correct is True)
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {config.run_name}")
    print(f"{'=' * 60}")
    print(f"  Total questions:  {total}")
    print(f"  Accuracy:         {correct_count}/{total} ({correct_count / max(total, 1):.1%})")
    print(f"  Results saved to: {output_file}")

    return experiment_result
