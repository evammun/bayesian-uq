"""
Generate paraphrases for QuALITY questions using the Anthropic API (Sonnet).

Produces a reusable paraphrase bank at data/paraphrases.json with three
categories per question: A (text-grounded), B (pure rephrase), C (angle shift).
Supports resumption — rerun to pick up where it left off.

Usage:
    python experiments/generate_paraphrases.py                 # full run
    python experiments/generate_paraphrases.py --dry-run       # inspect prompt
    python experiments/generate_paraphrases.py --max-batches 2 # test with 2 batches
    python experiments/generate_paraphrases.py --max-questions 1000  # first 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "quality_all.jsonl"
PARAPHRASES_PATH = DATA_DIR / "paraphrases.json"

# ---------------------------------------------------------------------------
# Cost constants (Sonnet 4 pricing)
# ---------------------------------------------------------------------------

INPUT_COST_PER_MTOK = 3.0
OUTPUT_COST_PER_MTOK = 15.0

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful academic question rewriter. Your job is to rephrase "
    "reading comprehension questions while preserving their exact meaning. "
    "You return only valid JSON with no additional text."
)

USER_PROMPT_TEMPLATE = """\
Generate 10 paraphrases for each question below, organised into three categories.

**Category A — Text-grounded framing (3 paraphrases)**
Add a reference to the title or the passage the reader just read.
Examples: "According to the passage...", "In [Title]...", "Based on what you just read..."
Use ONLY the title and character names. Do NOT add plot details or information beyond what the original question already contains.

**Category B — Pure rephrasing (3 paraphrases)**
Rephrase with different vocabulary and sentence structure. No reference to the text or title.

**Category C — Perspective/angle shifts (4 paraphrases)**
Change the angle of approach. Use counterfactuals ("What would have had to be different for..."), perspective inversions, negation framing, specificity shifts, or other creative restructuring. Push hard for diversity — these should feel genuinely different from the original.

Rules:
- The correct answer to the original must ALSO be correct for every paraphrase
- Do NOT include answer options — only rephrase the question stem
- Do NOT add information that isn't in the original question
- Do NOT hint at or steer toward any particular answer
- Keep roughly similar length to the original

Return a JSON object where each question_id maps to an object with keys "A", "B", "C", each containing a list of paraphrase strings.

Example output format:
{{"example_id": {{"A": ["...", "...", "..."], "B": ["...", "...", "..."], "C": ["...", "...", "...", "..."]}}}}

Questions:

{questions_block}"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(max_questions: int | None = None) -> list[dict]:
    """Load questions from QuALITY JSONL, flattened with article titles."""
    questions = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            article = json.loads(line.strip())
            title = article.get("title", "").strip()
            for q in article["questions"]:
                questions.append({
                    "question_id": q["question_unique_id"],
                    "question_text": q["question"].strip(),
                    "title": title,
                })
                if max_questions and len(questions) >= max_questions:
                    return questions
    return questions


def load_existing() -> dict:
    """Load existing paraphrases if file exists."""
    if not PARAPHRASES_PATH.exists():
        return {}
    try:
        with open(PARAPHRASES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: could not load existing paraphrases: {e}")
        return {}


def is_done(paraphrases: dict, qid: str) -> bool:
    """Check if a question has all 10 paraphrases (3A + 3B + 4C)."""
    val = paraphrases.get(qid)
    if not isinstance(val, dict):
        return False
    a = val.get("A", [])
    b = val.get("B", [])
    c = val.get("C", [])
    return len(a) >= 3 and len(b) >= 3 and len(c) >= 4


def count_done(paraphrases: dict) -> int:
    return sum(1 for qid in paraphrases if is_done(paraphrases, qid))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_questions_block(batch: list[dict]) -> str:
    parts = []
    for q in batch:
        title_line = f'  Title: "{q["title"]}"' if q["title"] else ""
        parts.append(
            f'[question_id: "{q["question_id"]}"]'
            f"{title_line}\n"
            f"  {q['question_text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw_text: str, batch: list[dict]) -> dict:
    """Parse JSON response. Returns dict of qid -> {"A": [...], "B": [...], "C": [...]}."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n  ERROR: could not parse JSON: {e}")
        print(f"  Raw (first 500 chars): {raw_text[:500]}")
        return {}

    expected_ids = {q["question_id"] for q in batch}
    result = {}

    for qid in expected_ids:
        if qid not in parsed:
            print(f"\n  Warning: {qid} missing from response")
            continue
        val = parsed[qid]
        if not isinstance(val, dict):
            print(f"\n  Warning: {qid} has non-dict value")
            continue

        cats = {}
        for cat, expected_n in [("A", 3), ("B", 3), ("C", 4)]:
            items = val.get(cat, [])
            clean = [p for p in items if isinstance(p, str) and p.strip()]
            if len(clean) < expected_n:
                print(f"\n  Warning: {qid} cat {cat} has {len(clean)}/{expected_n}")
            cats[cat] = clean
        result[qid] = cats

    return result


def save_paraphrases(paraphrases: dict) -> None:
    with open(PARAPHRASES_PATH, "w", encoding="utf-8") as f:
        json.dump(paraphrases, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    m = int(sec // 60)
    s = int(sec % 60)
    if m < 60:
        return f"{m}m {s}s"
    return f"{m // 60}h {m % 60}m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paraphrases for QuALITY questions via Anthropic API.",
    )
    parser.add_argument("--batch-size", type=int, default=15,
                        help="Questions per API call (default: 15)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after N batches (for testing)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Only process first N questions")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first batch prompt without calling API")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # Load data
    print("Loading questions...")
    questions = load_questions(max_questions=args.max_questions)
    paraphrases = load_existing()

    remaining = [q for q in questions if not is_done(paraphrases, q["question_id"])]
    n_done = count_done(paraphrases)
    print(f"  Total questions:  {len(questions)}")
    print(f"  Already done:     {n_done}")
    print(f"  Remaining:        {len(remaining)}")
    print()

    if not remaining:
        print("All questions already paraphrased. Nothing to do.")
        return

    # Batch
    batches = [remaining[i:i + args.batch_size]
               for i in range(0, len(remaining), args.batch_size)]
    if args.max_batches:
        batches = batches[:args.max_batches]
    print(f"  Batches: {len(batches)} (size {args.batch_size})")

    # Dry run
    if args.dry_run:
        prompt = USER_PROMPT_TEMPLATE.format(
            questions_block=build_questions_block(batches[0]))
        print("\n" + "=" * 70)
        print("DRY RUN — First batch prompt:")
        print("=" * 70)
        print(f"\n[System]\n{SYSTEM_PROMPT}\n")
        print(f"[User]\n{prompt}\n")
        print(f"Batch has {len(batches[0])} questions. No API call made.")
        return

    # Process batches in parallel
    write_lock = threading.Lock()
    print_lock = threading.Lock()
    stats = {"input_tok": 0, "output_tok": 0, "done": 0, "failed": 0, "qs": 0}
    t0 = time.monotonic()

    def process_batch(batch_idx: int, batch: list[dict]) -> None:
        client = anthropic.Anthropic(api_key=api_key, timeout=600.0)
        prompt = USER_PROMPT_TEMPLATE.format(
            questions_block=build_questions_block(batch))

        response = None
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32768,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                with print_lock:
                    print(f"  [Batch {batch_idx+1}] Rate limited, waiting {wait}s...",
                          flush=True)
                time.sleep(wait)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                with print_lock:
                    print(f"  [Batch {batch_idx+1}] API error: {e}", flush=True)
                time.sleep(30)

        if response is None:
            with write_lock:
                stats["failed"] += 1
            return

        batch_result = parse_response(response.content[0].text, batch)
        if not batch_result:
            with write_lock:
                stats["failed"] += 1
            return

        with write_lock:
            stats["input_tok"] += response.usage.input_tokens
            stats["output_tok"] += response.usage.output_tokens

            for q in batch:
                qid = q["question_id"]
                if qid in batch_result:
                    paraphrases[qid] = {
                        "original": q["question_text"],
                        "title": q["title"],
                        **batch_result[qid],
                    }
                    stats["qs"] += 1

            save_paraphrases(paraphrases)
            stats["done"] += 1

            elapsed = time.monotonic() - t0
            processed = stats["done"] + stats["failed"]
            eta = _fmt_sec((elapsed / processed) * (len(batches) - processed)) if processed else "?"
            cost = (stats["input_tok"] / 1e6 * INPUT_COST_PER_MTOK
                    + stats["output_tok"] / 1e6 * OUTPUT_COST_PER_MTOK)

        with print_lock:
            total_done = count_done(paraphrases)
            print(f"  [{total_done}/{len(questions)}] "
                  f"Batch {stats['done']}/{len(batches)} | "
                  f"ETA: {eta} | ~${cost:.2f}"
                  + (f" | {stats['failed']} failed" if stats["failed"] else ""),
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_batch, i, b): i
                   for i, b in enumerate(batches)}
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"\n  ERROR in batch {futures[f]}: {exc}", flush=True)

    # Summary
    elapsed = time.monotonic() - t0
    cost = (stats["input_tok"] / 1e6 * INPUT_COST_PER_MTOK
            + stats["output_tok"] / 1e6 * OUTPUT_COST_PER_MTOK)
    print(f"\nDone. {stats['qs']} questions paraphrased in {_fmt_sec(elapsed)}. "
          f"~${cost:.2f}. Saved to {PARAPHRASES_PATH}")


if __name__ == "__main__":
    main()
