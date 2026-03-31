"""
RGB Dataset Exploration & Profiling
====================================
Profiles all three English RGB dataset files:
  - rgb_en_refine.json (noise refinement)
  - rgb_en_int.json    (integration / multi-part)
  - rgb_en_fact.json   (factual robustness)

Prints full profile to stdout and saves summary to experiments/rgb_dataset_profile.md
"""

import json
import re
import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_MD = Path(__file__).resolve().parent / "rgb_dataset_profile.md"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FILES = {
    "refine": DATA_DIR / "rgb_en_refine.json",
    "int":    DATA_DIR / "rgb_en_int.json",
    "fact":   DATA_DIR / "rgb_en_fact.json",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, one JSON object per line."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Failed to parse line {i} in {path.name}: {e}")
    return entries


def word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def length_stats(values: list[int]) -> dict:
    """Compute min/max/mean/median for a list of ints."""
    arr = np.array(values)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


def fmt_stats(stats: dict) -> str:
    return f"min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}, median={stats['median']:.1f}, std={stats['std']:.1f}"


def classify_answer(answer_text: str) -> str:
    """Rough classification of answer type based on content."""
    answer_text = answer_text.strip()
    # Date patterns
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", answer_text, re.I):
        return "date"
    if re.search(r"\b\d{4}\b", answer_text) and len(answer_text) < 30:
        return "date/year"
    # Pure number
    if re.match(r"^[\d,.\-\+\$%]+$", answer_text):
        return "number"
    # Yes/No
    if answer_text.lower() in ("yes", "no"):
        return "yes/no"
    # Short phrase (1-3 words)
    wc = word_count(answer_text)
    if wc <= 3:
        return "short_phrase"
    if wc <= 8:
        return "medium_phrase"
    return "long_phrase"


def get_all_answer_strings(answer_field) -> list[str]:
    """
    Flatten the answer field into a list of string variants.
    Handles nested structures: str, list of str, list of list of str, mixed.
    """
    strings = []
    if isinstance(answer_field, str):
        strings.append(answer_field)
    elif isinstance(answer_field, list):
        for item in answer_field:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        strings.append(sub)
    return strings


# ── Per-file profiling ──────────────────────────────────────────────────────
def profile_file(name: str, entries: list[dict]) -> dict:
    """Profile a single dataset file. Returns a dict of stats."""
    print(f"\n{'='*70}")
    print(f"  {name.upper()} — {FILES[name].name} ({len(entries)} entries)")
    print(f"{'='*70}")

    # Keys present
    all_keys = set()
    for e in entries:
        all_keys.update(e.keys())
    print(f"\nFields: {sorted(all_keys)}")

    # Check required fields
    required = {"id", "query", "answer", "positive", "negative"}
    missing = required - all_keys
    extra = all_keys - required
    if missing:
        print(f"  MISSING required fields: {missing}")
    if extra:
        print(f"  Extra fields: {sorted(extra)}")

    # Query analysis
    q_words = [word_count(e["query"]) for e in entries]
    q_chars = [char_count(e["query"]) for e in entries]
    q_word_stats = length_stats(q_words)
    q_char_stats = length_stats(q_chars)
    print(f"\nQuery lengths (words): {fmt_stats(q_word_stats)}")
    print(f"Query lengths (chars): {fmt_stats(q_char_stats)}")

    # Answer analysis
    answer_variant_counts = []
    answer_types = Counter()
    for e in entries:
        variants = get_all_answer_strings(e["answer"])
        answer_variant_counts.append(len(variants))
        if variants:
            # Classify based on the first (canonical) variant
            answer_types[classify_answer(variants[0])] += 1

    av_stats = length_stats(answer_variant_counts)
    print(f"\nAnswer variants per question: {fmt_stats(av_stats)}")
    print(f"Answer type distribution:")
    for atype, count in answer_types.most_common():
        print(f"  {atype:>15s}: {count:4d} ({100*count/len(entries):.1f}%)")

    # Passage analysis
    pos_counts = [len(e.get("positive", [])) for e in entries]
    neg_counts = [len(e.get("negative", [])) for e in entries]
    pos_stats = length_stats(pos_counts)
    neg_stats = length_stats(neg_counts)
    print(f"\nPositive passages per entry: {fmt_stats(pos_stats)}")
    print(f"Negative passages per entry: {fmt_stats(neg_stats)}")

    # Passage lengths (sample to avoid slow-down on large nested lists)
    all_pos_lens = []
    all_neg_lens = []
    for e in entries:
        pos_list = e.get("positive", [])
        neg_list = e.get("negative", [])
        # Handle nested lists (en_int has list-of-lists for positive)
        for p in pos_list:
            if isinstance(p, str):
                all_pos_lens.append(word_count(p))
            elif isinstance(p, list):
                for sub in p:
                    if isinstance(sub, str):
                        all_pos_lens.append(word_count(sub))
        for n in neg_list:
            if isinstance(n, str):
                all_neg_lens.append(word_count(n))

    if all_pos_lens:
        print(f"Positive passage length (words): {fmt_stats(length_stats(all_pos_lens))} (n={len(all_pos_lens)})")
    if all_neg_lens:
        print(f"Negative passage length (words): {fmt_stats(length_stats(all_neg_lens))} (n={len(all_neg_lens)})")

    # Extra fields analysis (file-specific)
    if "fakeanswer" in all_keys:
        print(f"\n[fact-specific] Has 'fakeanswer' field (fake/wrong answers for robustness testing)")
        sample_fakes = [e["fakeanswer"] for e in entries[:3]]
        print(f"  Sample fake answers: {sample_fakes}")
    if "positive_wrong" in all_keys:
        pw_counts = [len(e.get("positive_wrong", [])) for e in entries]
        print(f"[fact-specific] 'positive_wrong' passages per entry: {fmt_stats(length_stats(pw_counts))}")
    if "asnwer1" in all_keys:
        print(f"\n[int-specific] Has 'asnwer1' (typo in original) and 'answer2' fields (multi-part answers)")
        # Check structure of positive field
        sample_pos = entries[0].get("positive", [])
        if isinstance(sample_pos, list) and len(sample_pos) > 0 and isinstance(sample_pos[0], list):
            print(f"  Positive field is nested: list of {len(sample_pos)} sub-lists (one per sub-question)")

    # Sample queries
    print(f"\n--- Sample queries (5 random) ---")
    sample_idx = random.sample(range(len(entries)), min(5, len(entries)))
    for idx in sample_idx:
        e = entries[idx]
        variants = get_all_answer_strings(e["answer"])
        short_ans = variants[0] if variants else "N/A"
        if len(short_ans) > 80:
            short_ans = short_ans[:77] + "..."
        print(f"  [{e['id']:3d}] Q: {e['query']}")
        print(f"        A: {short_ans}")

    return {
        "name": name,
        "count": len(entries),
        "fields": sorted(all_keys),
        "extra_fields": sorted(extra),
        "q_word_stats": q_word_stats,
        "q_char_stats": q_char_stats,
        "answer_variant_stats": av_stats,
        "answer_types": dict(answer_types.most_common()),
        "pos_passage_stats": pos_stats,
        "neg_passage_stats": neg_stats,
        "pos_len_stats": length_stats(all_pos_lens) if all_pos_lens else None,
        "neg_len_stats": length_stats(all_neg_lens) if all_neg_lens else None,
    }


# ── Cross-file comparison ──────────────────────────────────────────────────
def cross_file_comparison(all_data: dict[str, list[dict]], profiles: dict[str, dict]):
    print(f"\n{'='*70}")
    print(f"  CROSS-FILE COMPARISON")
    print(f"{'='*70}")

    # Size comparison
    print("\nEntry counts:")
    for name, entries in all_data.items():
        print(f"  {name:>8s}: {len(entries)}")

    # ID overlap
    print("\nID overlap check:")
    id_sets = {name: set(e["id"] for e in entries) for name, entries in all_data.items()}
    names = list(id_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = id_sets[names[i]] & id_sets[names[j]]
            print(f"  {names[i]} & {names[j]}: {len(overlap)} shared IDs")

    # Query overlap
    print("\nQuery overlap check:")
    query_sets = {name: set(e["query"] for e in entries) for name, entries in all_data.items()}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = query_sets[names[i]] & query_sets[names[j]]
            print(f"  {names[i]} & {names[j]}: {len(overlap)} shared queries")
            if overlap:
                print(f"    Examples: {list(overlap)[:3]}")

    # Structural differences
    print("\nStructural differences:")
    print(f"  refine: Standard QA with noisy retrieval. 5 fields.")
    print(f"  int:    Multi-part questions (asnwer1/answer2). Nested positive passages (grouped by sub-question). 7 fields.")
    print(f"  fact:   Factual robustness testing. Has fakeanswer + positive_wrong (passages with incorrect info). 7 fields.")

    # Query complexity comparison
    print("\nQuery complexity comparison (mean words):")
    for name, prof in profiles.items():
        print(f"  {name:>8s}: {prof['q_word_stats']['mean']:.1f} words, {prof['q_char_stats']['mean']:.1f} chars")

    # Answer type distribution comparison
    print("\nAnswer type distributions:")
    all_types = set()
    for prof in profiles.values():
        all_types.update(prof["answer_types"].keys())
    header = f"  {'type':>15s}" + "".join(f"  {n:>8s}" for n in profiles.keys())
    print(header)
    for atype in sorted(all_types):
        row = f"  {atype:>15s}"
        for prof in profiles.values():
            count = prof["answer_types"].get(atype, 0)
            row += f"  {count:>8d}"
        print(row)


# ── Markdown summary ────────────────────────────────────────────────────────
def write_summary_md(all_data: dict[str, list[dict]], profiles: dict[str, dict]):
    lines = []
    lines.append("# RGB Dataset Profile")
    lines.append(f"Generated by `rgb_exploration.py` | Seed={SEED}\n")

    # Overview table
    lines.append("## Overview")
    lines.append("| File | Entries | Fields | Query len (words) | Pos passages | Neg passages |")
    lines.append("|------|---------|--------|--------------------|--------------|--------------|")
    for name, prof in profiles.items():
        q = prof["q_word_stats"]
        lines.append(
            f"| {name} | {prof['count']} | {len(prof['fields'])} | "
            f"{q['mean']:.0f} ± {q['std']:.0f} | "
            f"{prof['pos_passage_stats']['mean']:.1f} | "
            f"{prof['neg_passage_stats']['mean']:.1f} |"
        )

    # Per-file details
    for name, prof in profiles.items():
        lines.append(f"\n## {name}")
        lines.append(f"- **Entries:** {prof['count']}")
        lines.append(f"- **Fields:** {', '.join(prof['fields'])}")
        if prof["extra_fields"]:
            lines.append(f"- **Extra fields:** {', '.join(prof['extra_fields'])}")
        lines.append(f"- **Query length (words):** {fmt_stats(prof['q_word_stats'])}")
        lines.append(f"- **Query length (chars):** {fmt_stats(prof['q_char_stats'])}")
        lines.append(f"- **Answer variants/question:** {fmt_stats(prof['answer_variant_stats'])}")
        lines.append(f"- **Answer types:** {prof['answer_types']}")
        lines.append(f"- **Positive passages/entry:** {fmt_stats(prof['pos_passage_stats'])}")
        lines.append(f"- **Negative passages/entry:** {fmt_stats(prof['neg_passage_stats'])}")
        if prof["pos_len_stats"]:
            lines.append(f"- **Positive passage length (words):** {fmt_stats(prof['pos_len_stats'])}")
        if prof["neg_len_stats"]:
            lines.append(f"- **Negative passage length (words):** {fmt_stats(prof['neg_len_stats'])}")

    # Cross-file
    lines.append("\n## Cross-file comparison")

    # ID/query overlap
    id_sets = {name: set(e["id"] for e in entries) for name, entries in all_data.items()}
    query_sets = {name: set(e["query"] for e in entries) for name, entries in all_data.items()}
    names = list(id_sets.keys())
    lines.append("### Overlap")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            id_ov = len(id_sets[names[i]] & id_sets[names[j]])
            q_ov = len(query_sets[names[i]] & query_sets[names[j]])
            lines.append(f"- {names[i]} & {names[j]}: {id_ov} shared IDs, {q_ov} shared queries")

    lines.append("\n### Key differences")
    lines.append("- **refine:** Standard QA with noisy retrieval (relevant + irrelevant passages). Tests ability to extract answers from mixed-quality context.")
    lines.append("- **int:** Multi-part questions requiring integration of info from multiple passages. Has sub-answers (asnwer1/answer2) and nested positive passages grouped by sub-question.")
    lines.append("- **fact:** Factual robustness testing. Includes `fakeanswer` and `positive_wrong` (passages containing plausible but incorrect information). Tests resistance to misinformation in retrieved context.")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary saved to {OUTPUT_MD}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("RGB Dataset Exploration")
    print("=" * 70)

    # Load all files
    all_data = {}
    for name, path in FILES.items():
        print(f"\nLoading {path.name}...")
        if not path.exists():
            print(f"  ERROR: {path} not found!")
            sys.exit(1)
        all_data[name] = load_jsonl(path)
        print(f"  Loaded {len(all_data[name])} entries")

    # Profile each file
    profiles = {}
    for name, entries in all_data.items():
        profiles[name] = profile_file(name, entries)

    # Cross-file comparison
    cross_file_comparison(all_data, profiles)

    # Write markdown summary
    write_summary_md(all_data, profiles)

    print("\nDone.")


if __name__ == "__main__":
    main()
