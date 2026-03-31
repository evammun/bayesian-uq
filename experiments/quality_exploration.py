"""
QuALITY dataset exploration and profiling.

Produces detailed stats on both train and dev splits, including:
  - Article/question counts, difficulty split
  - Article length analysis (token estimates, n_ctx fit check)
  - Question/option length stats
  - Gold label distribution
  - Sample questions for inspection
  - Cross-article analysis for insufficient context construction
  - Summary saved to experiments/quality_dataset_profile.md

Usage:
    python experiments/quality_exploration.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_quality_jsonl(path: Path) -> list[dict]:
    """Load a QuALITY JSONL file, returning list of article dicts."""
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line.strip()))
    return articles


def flatten_questions(articles: list[dict]) -> list[dict]:
    """Flatten articles into individual question dicts with metadata."""
    questions = []
    for article in articles:
        for q in article["questions"]:
            questions.append({
                "article_id": article["article_id"],
                "question_unique_id": q["question_unique_id"],
                "question": q["question"],
                "options": q["options"],
                "gold_label": q["gold_label"],  # 1-indexed
                "difficult": q.get("difficult", 0),
                "article_text": article["article"],
                "source": article.get("source", "unknown"),
                "topic": article.get("topic", "unknown"),
                "title": article.get("title", ""),
            })
    return questions


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def basic_stats(articles: list[dict], split_name: str) -> str:
    """Compute basic stats for a split."""
    lines = [f"\n## {split_name}\n"]

    n_articles = len(articles)
    questions = flatten_questions(articles)
    n_questions = len(questions)

    lines.append(f"- **Articles:** {n_articles}")
    lines.append(f"- **Questions:** {n_questions}")

    # Questions per article
    q_per_article = [len(a["questions"]) for a in articles]
    lines.append(f"- **Questions/article:** min={min(q_per_article)}, "
                 f"max={max(q_per_article)}, mean={np.mean(q_per_article):.1f}, "
                 f"median={np.median(q_per_article):.0f}")

    # Hard vs easy
    hard_count = sum(1 for q in questions if q["difficult"] == 1)
    easy_count = n_questions - hard_count
    lines.append(f"- **Easy questions:** {easy_count} ({easy_count/n_questions:.0%})")
    lines.append(f"- **Hard questions:** {hard_count} ({hard_count/n_questions:.0%})")

    # By source
    source_counts = Counter(a.get("source", "unknown") for a in articles)
    lines.append(f"\n### Articles by source")
    for source, count in source_counts.most_common():
        lines.append(f"- {source}: {count}")

    # By topic
    topic_counts = Counter(a.get("topic", "unknown") for a in articles)
    lines.append(f"\n### Articles by topic")
    for topic, count in topic_counts.most_common():
        lines.append(f"- {topic}: {count}")

    return "\n".join(lines)


def article_length_analysis(articles: list[dict]) -> str:
    """Analyse article lengths and n_ctx fit."""
    lines = ["\n## Article Length Analysis\n"]

    word_counts = [len(a["article"].split()) for a in articles]
    # Rough token estimate: words / 0.75
    token_estimates = [wc / 0.75 for wc in word_counts]

    lines.append(f"- **Word counts:** min={min(word_counts)}, max={max(word_counts)}, "
                 f"mean={np.mean(word_counts):.0f}, median={np.median(word_counts):.0f}")
    lines.append(f"- **Token estimates (words/0.75):** min={min(token_estimates):.0f}, "
                 f"max={max(token_estimates):.0f}, mean={np.mean(token_estimates):.0f}, "
                 f"median={np.median(token_estimates):.0f}")

    # n_ctx fit check (8192 context, ~200 tokens for prompt overhead)
    available_tokens = 8192 - 200
    too_long = sum(1 for t in token_estimates if t > available_tokens)
    lines.append(f"- **Articles exceeding n_ctx=8192 (with 200 token overhead):** "
                 f"{too_long}/{len(articles)} ({too_long/len(articles):.0%})")

    if too_long > 0:
        long_ones = [(a.get("article_id", "?"), len(a["article"].split()))
                     for a in articles if len(a["article"].split()) / 0.75 > available_tokens]
        long_ones.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"  - Longest: {long_ones[0][0]} ({long_ones[0][1]} words, "
                     f"~{long_ones[0][1]/0.75:.0f} tokens)")

    return "\n".join(lines)


def question_analysis(questions: list[dict]) -> str:
    """Analyse question and option characteristics."""
    lines = ["\n## Question Analysis\n"]

    # Question length
    q_word_counts = [len(q["question"].split()) for q in questions]
    lines.append(f"- **Question length (words):** min={min(q_word_counts)}, "
                 f"max={max(q_word_counts)}, mean={np.mean(q_word_counts):.1f}, "
                 f"median={np.median(q_word_counts):.0f}")

    # Option length
    all_option_lengths = []
    per_question_option_var = []
    for q in questions:
        opt_lens = [len(opt.split()) for opt in q["options"]]
        all_option_lengths.extend(opt_lens)
        per_question_option_var.append(np.std(opt_lens))

    lines.append(f"- **Option length (words):** min={min(all_option_lengths)}, "
                 f"max={max(all_option_lengths)}, mean={np.mean(all_option_lengths):.1f}, "
                 f"median={np.median(all_option_lengths):.0f}")
    lines.append(f"- **Option length std within question:** mean={np.mean(per_question_option_var):.1f}, "
                 f"median={np.median(per_question_option_var):.0f}")
    lines.append(f"  (Low variance = options are similar length = less spurious length signal)")

    # Gold label distribution (1-indexed)
    gold_labels = [q["gold_label"] for q in questions]
    label_counts = Counter(gold_labels)
    lines.append(f"\n### Gold label distribution (1-indexed)")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        lines.append(f"- Option {label}: {count} ({count/len(questions):.1%})")

    return "\n".join(lines)


def sample_questions(questions: list[dict], n: int = 10) -> str:
    """Print diverse sample questions."""
    lines = ["\n## Sample Questions\n"]

    # Group by (topic, source, difficult)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for q in questions:
        key = (q["topic"], q["source"], q["difficult"])
        groups[key].append(q)

    # Pick diverse samples
    selected = []
    import random
    rng = random.Random(42)

    # Try to get mix of easy/hard from different topics
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    for key in group_keys:
        if len(selected) >= n:
            break
        pool = groups[key]
        selected.append(rng.choice(pool))

    # Fill remainder if needed
    if len(selected) < n:
        remaining = [q for q in questions if q not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    for i, q in enumerate(selected[:n], 1):
        diff_str = "HARD" if q["difficult"] else "easy"
        article_preview = q["article_text"][:200].replace("\n", " ")
        lines.append(f"### Sample {i} [{diff_str}] — {q['source']}/{q['topic']}")
        lines.append(f"**Article:** {q['title'][:80]}")
        lines.append(f"**Preview:** {article_preview}...")
        lines.append(f"**Question:** {q['question']}")
        for j, opt in enumerate(q["options"], 1):
            marker = " **<--**" if j == q["gold_label"] else ""
            lines.append(f"  {j}) {opt}{marker}")
        lines.append(f"**Gold:** Option {q['gold_label']}")
        lines.append("")

    return "\n".join(lines)


def cross_article_analysis(articles: list[dict]) -> str:
    """Analyse article overlap for insufficient context construction."""
    lines = ["\n## Cross-Article Analysis (for insufficient context swaps)\n"]

    # Topic × Source matrix
    topic_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in articles:
        topic_source[a.get("topic", "unknown")][a.get("source", "unknown")] += 1

    topics = sorted(topic_source.keys())
    sources = sorted(set(s for ts in topic_source.values() for s in ts))

    lines.append("### Topic × Source matrix (article counts)")
    lines.append("")
    header = f"| {'Topic':<20} | " + " | ".join(f"{s:<12}" for s in sources) + " | Total |"
    lines.append(header)
    lines.append("|" + "-" * 22 + "|" + "|".join("-" * 14 for _ in sources) + "|-------|")

    for topic in topics:
        row_vals = [topic_source[topic].get(s, 0) for s in sources]
        total = sum(row_vals)
        row = f"| {topic:<20} | " + " | ".join(f"{v:<12}" for v in row_vals) + f" | {total:<5} |"
        lines.append(row)

    # Summary
    lines.append(f"\n### Swap feasibility")
    for topic in topics:
        n_articles = sum(topic_source[topic].values())
        lines.append(f"- **{topic}:** {n_articles} articles "
                     f"({'enough for same-topic swaps' if n_articles >= 2 else 'CROSS-TOPIC SWAP NEEDED'})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_path = project_root / "experiments" / "quality_dataset_profile.md"

    print("=" * 60)
    print("  QuALITY Dataset Exploration")
    print("=" * 60)

    report_parts = ["# QuALITY Dataset Profile\n"]

    for split_name, filename in [("Dev", "quality_dev.jsonl"), ("Train", "quality_train.jsonl")]:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n  WARNING: {filepath} not found, skipping {split_name}")
            continue

        print(f"\nLoading {split_name}: {filepath.name}")
        articles = load_quality_jsonl(filepath)
        questions = flatten_questions(articles)

        # Print to terminal
        stats = basic_stats(articles, split_name)
        print(stats)
        report_parts.append(stats)

    # Detailed analysis on dev set (our main evaluation set)
    dev_path = data_dir / "quality_dev.jsonl"
    if dev_path.exists():
        print("\n--- Detailed Dev Set Analysis ---")
        dev_articles = load_quality_jsonl(dev_path)
        dev_questions = flatten_questions(dev_articles)

        art_analysis = article_length_analysis(dev_articles)
        print(art_analysis)
        report_parts.append(art_analysis)

        q_analysis = question_analysis(dev_questions)
        print(q_analysis)
        report_parts.append(q_analysis)

        samples = sample_questions(dev_questions)
        print(samples)
        report_parts.append(samples)

        cross = cross_article_analysis(dev_articles)
        print(cross)
        report_parts.append(cross)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_parts))
    print(f"\nProfile saved to: {output_path}")


if __name__ == "__main__":
    main()
