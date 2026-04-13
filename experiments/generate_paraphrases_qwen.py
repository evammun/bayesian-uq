"""
Generate paraphrases using Qwen locally (no API needed).

Runs on Mahti A100 for speed. Generates 5 Category A (text-grounded)
and 5 Category B (pure rephrase) per question. Saves every 10 questions.
Supports resume.

Usage:
    python experiments/generate_paraphrases_qwen.py
    python experiments/generate_paraphrases_qwen.py --max-questions 50  # test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASET_PATH = PROJECT_ROOT / "data" / "quality_all.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "paraphrases_qwen.json"

PROMPT_TEMPLATE = """\
Generate 10 paraphrases of the following reading comprehension question.

Category A (3 paraphrases): Add a reference to the title or passage.
Examples: "According to the passage...", "In {title}...", "Based on what you just read..."
Use ONLY the title and character names from the original question. Do NOT add plot details.

Category B (7 paraphrases): Rephrase with different vocabulary and structure. No reference to the title or text. Vary sentence structure substantially — each should read differently.

Rules:
- The correct answer must be identical for every paraphrase
- Do NOT include answer options
- Do NOT add information not in the original question
- Keep similar length

Title: "{title}"
Question: "{question}"

Return valid JSON only, no other text:
{{"A": ["...", "...", "..."], "B": ["...", "...", "...", "...", "...", "...", "..."]}}"""


def load_questions(max_questions: int | None = None) -> list[dict]:
    """Load questions from QuALITY JSONL."""
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
    if not OUTPUT_PATH.exists():
        return {}
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_done(data: dict, qid: str) -> bool:
    val = data.get(qid)
    if not isinstance(val, dict):
        return False
    return len(val.get("A", [])) >= 3 and len(val.get("B", [])) >= 7


def save(data: dict) -> None:
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_response(text: str) -> dict | None:
    """Try to extract JSON from model output."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:-3].strip()
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            a = [p for p in parsed.get("A", []) if isinstance(p, str) and p.strip()]
            b = [p for p in parsed.get("B", []) if isinstance(p, str) and p.strip()]
            if len(a) >= 2 and len(b) >= 5:  # accept partial
                return {"A": a, "B": b}
        except json.JSONDecodeError:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    from pre_action_uq.inference import LlamaCppClient, find_model_path

    # Find model
    model_name = "qwen3:8b-q4_K_M"
    model_path = os.environ.get("UQ_MODEL_PATH") or find_model_path(model_name)
    if not model_path:
        print("ERROR: No model found")
        sys.exit(1)

    # Load model
    print(f"Loading model: {Path(str(model_path)).name}")
    client = LlamaCppClient(str(model_path), n_ctx=4096, verbose=False)
    print(f"  Loaded in {client.load_time:.1f}s")

    # Load data
    questions = load_questions(max_questions=args.max_questions)
    existing = load_existing()
    remaining = [q for q in questions if not is_done(existing, q["question_id"])]

    print(f"Total: {len(questions)}, Done: {len(questions) - len(remaining)}, Remaining: {len(remaining)}")
    if not remaining:
        print("All done.")
        return

    t0 = time.monotonic()
    done = 0
    failed = 0

    for i, q in enumerate(remaining):
        prompt = PROMPT_TEMPLATE.format(title=q["title"], question=q["question_text"])
        chat_prompt = client._build_chat_prompt(prompt, think=False)

        try:
            gen = client.model.create_completion(
                prompt=chat_prompt,
                max_tokens=1024,
                temperature=0.0,
                stop=["<|im_end|>"],
                echo=False,
            )
            raw = gen["choices"][0]["text"]
            result = parse_response(raw)
        except Exception as e:
            print(f"  ERROR on {q['question_id']}: {e}")
            result = None

        if result:
            existing[q["question_id"]] = {
                "original": q["question_text"],
                "title": q["title"],
                **result,
            }
            done += 1
        else:
            failed += 1
            if i < 5:
                print(f"  PARSE FAIL {q['question_id']}: {raw[:200] if 'raw' in dir() else '?'}")

        # Save every 10
        if (i + 1) % 10 == 0 or i == len(remaining) - 1:
            save(existing)
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            total_done = sum(1 for qid in existing if is_done(existing, qid))
            print(f"  [{total_done}/{len(questions)}] {done} ok, {failed} failed | "
                  f"{rate:.1f} q/s | ETA: {eta/60:.0f}m", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nDone. {done} generated, {failed} failed in {elapsed/60:.1f}m. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
