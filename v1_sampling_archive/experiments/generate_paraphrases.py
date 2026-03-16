"""
Generate paraphrases for each question using the Anthropic API (Claude Sonnet).

One-time data preparation step that produces a reusable paraphrase bank
stored at data/paraphrases.json. Supports resumption — if the script is
interrupted, rerun it and it picks up where it left off.

Usage:
    python experiments/generate_paraphrases.py                 # full run
    python experiments/generate_paraphrases.py --dry-run       # inspect prompt
    python experiments/generate_paraphrases.py --max-batches 2 # test with 2 batches
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"
PARAPHRASES_PATH = PROJECT_ROOT / "data" / "paraphrases.json"

# ---------------------------------------------------------------------------
# Cost constants (Sonnet pricing as of 2025)
# ---------------------------------------------------------------------------

INPUT_COST_PER_MTOK = 3.0    # $3 per million input tokens
OUTPUT_COST_PER_MTOK = 15.0  # $15 per million output tokens

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful academic question rewriter. Your job is to rephrase "
    "exam questions while preserving their exact meaning. You return only "
    "valid JSON with no additional text."
)

USER_PROMPT_TEMPLATE = """\
Generate 10 paraphrases for each of the following exam questions.

Rules:
- Preserve ALL factual content exactly: every name, number, technical term, organism, date, and specific claim must appear in every paraphrase
- Vary sentence structure, vocabulary, and framing substantially — each paraphrase should read differently
- Each paraphrase must be a natural, standalone exam question
- Do NOT include answer choices — only rephrase the question stem
- Do NOT add information that isn't in the original
- Do NOT remove information that is in the original
- If the question contains a premise or setup paragraph, you may rephrase it but must keep all factual claims

Return a JSON object mapping each question_id to a list of 10 paraphrase strings.

Questions:

{questions_block}"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions() -> list[dict]:
    """Load all questions from data/questions.json."""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_existing_paraphrases() -> dict:
    """Load existing paraphrases from data/paraphrases.json if it exists.

    Returns a dict keyed by question_id. Handles both the old format
    (list of {text: ...} objects) and the new format (dict with
    'original' and 'paraphrases' keys).
    """
    if not PARAPHRASES_PATH.exists():
        return {}
    try:
        with open(PARAPHRASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: could not load existing paraphrases: {e}")
        return {}


def count_done(paraphrases: dict) -> int:
    """Count how many questions already have 10 paraphrases."""
    count = 0
    for qid, val in paraphrases.items():
        if isinstance(val, dict):
            # New format: {"original": ..., "paraphrases": [...]}
            if len(val.get("paraphrases", [])) >= 10:
                count += 1
        elif isinstance(val, list):
            # Old format: [{text: ...}, ...]
            if len(val) >= 10:
                count += 1
    return count


def is_question_done(paraphrases: dict, qid: str) -> bool:
    """Check if a question already has 10 paraphrases."""
    val = paraphrases.get(qid)
    if val is None:
        return False
    if isinstance(val, dict):
        return len(val.get("paraphrases", [])) >= 10
    if isinstance(val, list):
        return len(val) >= 10
    return False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_questions_block(batch: list[dict]) -> str:
    """Format a batch of questions into the prompt's question block."""
    parts = []
    for q in batch:
        parts.append(f'[question_id: "{q["question_id"]}"]\n{q["question_text"]}')
    return "\n\n".join(parts)


def build_user_prompt(batch: list[dict]) -> str:
    """Build the full user prompt for a batch of questions."""
    block = build_questions_block(batch)
    return USER_PROMPT_TEMPLATE.format(questions_block=block)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw_text: str, batch: list[dict]) -> dict:
    """Parse the model's JSON response into a dict of paraphrase lists.

    Handles models that wrap JSON in markdown code fences.
    Returns a dict mapping question_id -> list[str].
    """
    text = raw_text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        return _validate_parsed(parsed, batch)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        return _validate_parsed(parsed, batch)
    except json.JSONDecodeError as e:
        print(f"\n  ERROR: could not parse JSON response: {e}")
        print(f"  Raw response (first 500 chars): {raw_text[:500]}")
        return {}


def _validate_parsed(parsed: dict, batch: list[dict]) -> dict:
    """Validate and clean parsed paraphrases."""
    expected_ids = {q["question_id"] for q in batch}
    result = {}

    for qid in expected_ids:
        if qid not in parsed:
            print(f"\n  Warning: question {qid} missing from response")
            continue
        paras = parsed[qid]
        if not isinstance(paras, list):
            print(f"\n  Warning: question {qid} has non-list value, skipping")
            continue
        # Filter to non-empty strings
        clean = [p for p in paras if isinstance(p, str) and p.strip()]
        if len(clean) < 10:
            print(f"\n  Warning: question {qid} has {len(clean)}/10 paraphrases")
        result[qid] = clean

    return result


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_paraphrases(paraphrases: dict) -> None:
    """Write the full paraphrases dict to data/paraphrases.json."""
    with open(PARAPHRASES_PATH, "w", encoding="utf-8") as f:
        json.dump(paraphrases, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paraphrases for questions using the Anthropic API.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Questions per API call (default: 20)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Stop after N batches (default: unlimited, useful for testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel API workers (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the first batch prompt without calling the API",
    )
    args = parser.parse_args()

    # --- Check API key ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # --- Load data ---
    print("Loading questions...")
    questions = load_questions()
    paraphrases = load_existing_paraphrases()

    # Filter to questions that still need paraphrasing
    remaining = [q for q in questions if not is_question_done(paraphrases, q["question_id"])]

    n_done = count_done(paraphrases)
    print(f"  Total questions:    {len(questions)}")
    print(f"  Already done:       {n_done}")
    print(f"  Remaining:          {len(remaining)}")
    print()

    if not remaining:
        print("All questions already have 10 paraphrases. Nothing to do.")
        return

    # --- Batch the remaining questions ---
    batches = []
    for i in range(0, len(remaining), args.batch_size):
        batches.append(remaining[i : i + args.batch_size])

    if args.max_batches is not None:
        batches = batches[: args.max_batches]

    total_batches = len(batches)
    print(f"  Batches to process: {total_batches} (batch size: {args.batch_size})")
    print()

    # --- Dry run: just print the first prompt ---
    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — First batch prompt:")
        print("=" * 70)
        print()
        print(f"[System message]")
        print(SYSTEM_PROMPT)
        print()
        print(f"[User message]")
        print(build_user_prompt(batches[0]))
        print()
        print("=" * 70)
        print(f"Batch contains {len(batches[0])} questions.")
        print("No API call made.")
        return

    # --- Process batches in parallel ---
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = args.workers
    print(f"  Workers: {workers}")
    print()

    # Shared mutable state protected by locks
    write_lock = threading.Lock()
    print_lock = threading.Lock()
    total_input_tokens = 0
    total_output_tokens = 0
    questions_done_this_run = 0
    completed_batches = [0]  # mutable counter
    failed_batches = [0]
    start_time = time.monotonic()

    def _process_batch(batch_idx: int, batch: list[dict]) -> None:
        """Process one batch: call API, parse, merge results."""
        nonlocal total_input_tokens, total_output_tokens, questions_done_this_run

        batch_num = batch_idx + 1
        batch_ids = [q["question_id"] for q in batch]

        # Each worker gets its own client (separate HTTP connection)
        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=600.0,
        )

        # Build the prompt
        user_prompt = build_user_prompt(batch)

        # API call with retries
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32768,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                break
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)  # back off more with each retry
                with print_lock:
                    print(f"\n    [Batch {batch_num}] Rate limited, waiting {wait}s "
                          f"(retry {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    wait = 30
                    with print_lock:
                        print(f"\n    [Batch {batch_num}] Server error {e.status_code}, "
                              f"waiting {wait}s (retry {attempt + 1}/{max_retries})...",
                              flush=True)
                    time.sleep(wait)
                else:
                    with print_lock:
                        print(f"\n    [Batch {batch_num}] ERROR: API {e.status_code}: "
                              f"{e.message}", flush=True)
                    response = None
                    break
            except anthropic.APIConnectionError:
                wait = 30
                with print_lock:
                    print(f"\n    [Batch {batch_num}] Connection error, waiting {wait}s "
                          f"(retry {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(wait)

        if response is None:
            with print_lock:
                print(f"  [Batch {batch_num}] FAILED — skipping {len(batch)} questions",
                      flush=True)
            with write_lock:
                failed_batches[0] += 1
            return

        # Parse the response
        raw_text = response.content[0].text
        batch_result = parse_response(raw_text, batch)

        if not batch_result:
            with print_lock:
                print(f"  [Batch {batch_num}] PARSE FAILED — skipping {len(batch)} questions",
                      flush=True)
            with write_lock:
                failed_batches[0] += 1
            return

        # Thread-safe merge and save
        with write_lock:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            for q in batch:
                qid = q["question_id"]
                if qid in batch_result:
                    paraphrases[qid] = {
                        "original": q["question_text"],
                        "paraphrases": batch_result[qid],
                    }
                    questions_done_this_run += 1

            # Save incrementally
            save_paraphrases(paraphrases)

            completed_batches[0] += 1
            done_count = completed_batches[0]
            fail_count = failed_batches[0]
            new_done = count_done(paraphrases)

            # Progress stats
            elapsed = time.monotonic() - start_time
            elapsed_str = _fmt_seconds(elapsed)
            processed = done_count + fail_count
            avg_per_batch = elapsed / processed if processed > 0 else 0
            remaining_count = total_batches - processed
            eta_str = _fmt_seconds(avg_per_batch * remaining_count)
            pct = new_done / len(questions) * 100

            # Estimate cost so far
            cost_so_far = (
                total_input_tokens / 1_000_000 * INPUT_COST_PER_MTOK
                + total_output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK
            )

        with print_lock:
            print(
                f"  [{pct:5.1f}%] {new_done}/{len(questions)} questions | "
                f"Batch {done_count}/{total_batches} | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                f"~${cost_so_far:.2f}"
                + (f" | {fail_count} failed" if fail_count else ""),
                flush=True,
            )

    # Dispatch all batches through the thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_batch, idx, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            # Re-raise any unexpected exceptions
            exc = future.exception()
            if exc is not None:
                print(f"\n  UNEXPECTED ERROR in batch {futures[future]}: {exc}",
                      flush=True)

    # --- Final summary ---
    elapsed_total = time.monotonic() - start_time
    input_cost = total_input_tokens / 1_000_000 * INPUT_COST_PER_MTOK
    output_cost = total_output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK
    total_cost = input_cost + output_cost

    print()
    print("=" * 60)
    print("  PARAPHRASE GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Questions paraphrased this run: {questions_done_this_run}")
    print(f"  Total with paraphrases:         {count_done(paraphrases)}/{len(questions)}")
    print(f"  Time elapsed:                   {_fmt_seconds(elapsed_total)}")
    print(f"  Input tokens:                   {total_input_tokens:,}")
    print(f"  Output tokens:                  {total_output_tokens:,}")
    print(f"  Estimated cost:                 ${total_cost:.2f} "
          f"(${input_cost:.2f} input + ${output_cost:.2f} output)")
    print(f"  Saved to:                       {PARAPHRASES_PATH}")


def _fmt_seconds(sec: float) -> str:
    """Format seconds as a readable string."""
    if sec < 60:
        return f"{sec:.0f}s"
    m = int(sec // 60)
    s = int(sec % 60)
    if m < 60:
        return f"{m}m {s}s"
    h = m // 60
    m = m % 60
    return f"{h}h {m}m"


if __name__ == "__main__":
    main()
