"""
Compute per-question uncertainty signals from QuALITY experiment result files.

Adapted from v2 MMLU compute_signals.py:
  - Removed MMLU-specific SUBJECT_TO_PARENT mapping
  - Added QuALITY metadata: article_id, source, topic, difficult, context_condition
  - All Tier I, II, III signals preserved unchanged
  - Same CSV output format, same column naming

Usage:
    python -m analysis.compute_signals
    python analysis/compute_signals.py --results-dir results/ --output analysis/signals.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_LETTERS = ["A", "B", "C", "D"]
NUM_CHOICES = 4


def _tidy_condition(condition: str) -> str:
    """Convert e.g. 'quality_pilot_direct_nothink_sufficient' → 'DirectNothinkSufficient'."""
    parts = condition.split("_")
    skip = {"quality", "pilot", "full", "exp1", "exp2"}
    cleaned = [p for p in parts if p.lower() not in skip]
    return "".join(p.capitalize() for p in cleaned)


# ---------------------------------------------------------------------------
# Helper: answer token detection (mirrors inference.py logic)
# ---------------------------------------------------------------------------

def is_answer_token(token_str: str) -> bool:
    """Check if a token string is an answer letter."""
    return token_str.strip() in ("A", "B", "C", "D")


def _token_to_letter(token_str: str) -> str | None:
    """Map a token like 'B' or ' B' to the canonical letter."""
    stripped = token_str.strip()
    return stripped if stripped in ("A", "B", "C", "D") else None


def _get_top_logprobs(logprobs_entry: dict) -> list[dict]:
    """Extract top_logprobs list from a logprobs entry."""
    if isinstance(logprobs_entry, list):
        return logprobs_entry
    if "top_logprobs" in logprobs_entry:
        return logprobs_entry["top_logprobs"]
    return []


def _find_answer_logprobs_entry(
    raw_logprobs: list[dict], prompt_mode: str
) -> dict | None:
    """Find the logprobs entry for the answer token position."""
    if not raw_logprobs:
        return None

    if prompt_mode == "direct":
        return raw_logprobs[0]

    # CoT: scan backwards for last answer token
    for entry in reversed(raw_logprobs):
        top_lp = _get_top_logprobs(entry)
        if not top_lp:
            continue
        top_token = top_lp[0].get("token", "")
        if _token_to_letter(top_token) is not None:
            return entry
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_result_file(path: Path) -> dict | None:
    """Load a result JSON, returning None on error."""
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except ImportError:
        pass
    except (ValueError, OSError) as e:
        print(f"  WARNING: Could not load {path.name}: {e}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Could not load {path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Tier I: Within-prompt signals (first query only)
# ---------------------------------------------------------------------------

def compute_tier1_signals(
    first_query: dict, prompt_mode: str
) -> dict:
    """Compute single-prompt signals from the first query."""
    signals: dict = {}
    probs = first_query.get("canonical_probs", [])

    if len(probs) != NUM_CHOICES:
        return _nan_tier1()

    sorted_desc = sorted(probs, reverse=True)

    signals["msp"] = sorted_desc[0]
    signals["single_entropy"] = float(scipy_entropy(probs, base=2))
    signals["second_gap"] = sorted_desc[0] - sorted_desc[1]

    h_nats = float(scipy_entropy(probs))
    signals["effective_option_count_single"] = math.exp(h_nats)

    # Distribution shape
    if sorted_desc[0] > 0.80:
        signals["distribution_shape"] = "peaked"
    elif sorted_desc[0] + sorted_desc[1] > 0.80 and sorted_desc[1] > 0.15:
        signals["distribution_shape"] = "bimodal"
    elif sorted_desc[0] < 0.35:
        signals["distribution_shape"] = "flat"
    else:
        signals["distribution_shape"] = "spread"

    # Full top-20 logprobs signals
    raw_logprobs = first_query.get("raw_logprobs", [])
    answer_entry = _find_answer_logprobs_entry(raw_logprobs, prompt_mode)

    if answer_entry is not None:
        top_lp = _get_top_logprobs(answer_entry)
        if top_lp:
            _compute_full_vocab_signals(top_lp, signals)
        else:
            _set_full_vocab_nan(signals)
    else:
        _set_full_vocab_nan(signals)

    # CoT response length
    if raw_logprobs:
        signals["cot_response_length"] = len(raw_logprobs)
    else:
        signals["cot_response_length"] = float("nan")

    return signals


def _compute_full_vocab_signals(top_logprobs: list[dict], signals: dict) -> None:
    """Compute answer_coverage, hesitation_mass, top_token_is_answer, missing_letter_count."""
    token_probs = []
    for entry in top_logprobs:
        lp = entry.get("logprob", -100)
        token_probs.append((entry.get("token", ""), math.exp(lp)))

    total_mass = sum(p for _, p in token_probs)
    answer_mass = sum(p for tok, p in token_probs if is_answer_token(tok))

    signals["answer_coverage"] = answer_mass / total_mass if total_mass > 0 else float("nan")
    signals["hesitation_mass"] = 1.0 - signals["answer_coverage"]

    if token_probs:
        signals["top_token_is_answer"] = 1 if is_answer_token(token_probs[0][0]) else 0
    else:
        signals["top_token_is_answer"] = float("nan")

    present_letters = set()
    for tok, _ in token_probs:
        letter = _token_to_letter(tok)
        if letter is not None:
            present_letters.add(letter)
    signals["missing_letter_count"] = NUM_CHOICES - len(present_letters)


def _set_full_vocab_nan(signals: dict) -> None:
    signals["answer_coverage"] = float("nan")
    signals["hesitation_mass"] = float("nan")
    signals["top_token_is_answer"] = float("nan")
    signals["missing_letter_count"] = float("nan")


def _nan_tier1() -> dict:
    return {
        "msp": float("nan"),
        "single_entropy": float("nan"),
        "second_gap": float("nan"),
        "effective_option_count_single": float("nan"),
        "distribution_shape": None,
        "answer_coverage": float("nan"),
        "hesitation_mass": float("nan"),
        "top_token_is_answer": float("nan"),
        "missing_letter_count": float("nan"),
        "cot_response_length": float("nan"),
    }


# ---------------------------------------------------------------------------
# Alternative aggregation methods
# ---------------------------------------------------------------------------

def compute_alternative_aggregations(
    query_probs: list[list[float]], mean_probs: list[float]
) -> dict:
    """Compute final answers under multiple aggregation strategies."""
    n = len(query_probs)
    probs_arr = np.array(query_probs)
    mean_argmax = int(np.argmax(mean_probs))

    # Majority vote
    per_query_votes = [int(np.argmax(p)) for p in query_probs]
    vote_counts = Counter(per_query_votes)
    max_count = max(vote_counts.values())
    top_votes = [ans for ans, cnt in vote_counts.items() if cnt == max_count]
    majority = top_votes[0] if len(top_votes) == 1 else mean_argmax

    # Weighted vote
    weighted_scores = np.zeros(NUM_CHOICES)
    for p in query_probs:
        winner = int(np.argmax(p))
        weighted_scores[winner] += max(p)
    weighted = int(np.argmax(weighted_scores))

    # Geometric mean
    log_probs = np.log(probs_arr + 1e-30)
    geo_mean = np.exp(log_probs.mean(axis=0))
    geo_mean = geo_mean / geo_mean.sum()
    geo = int(np.argmax(geo_mean))

    # Median
    median_probs = np.median(probs_arr, axis=0)
    med = int(np.argmax(median_probs))

    return {
        "answer_majority_vote": majority,
        "answer_weighted_vote": weighted,
        "answer_geometric_mean": geo,
        "answer_median": med,
    }


# ---------------------------------------------------------------------------
# Tier II: Aggregated signals (across all queries)
# ---------------------------------------------------------------------------

def compute_tier2_signals(
    query_log: list[dict],
    mean_probs: list[float],
    final_answer: int,
    prompt_mode: str,
) -> dict:
    """Compute across-query aggregated signals."""
    n = len(query_log)
    if n <= 1:
        return _nan_tier2()

    # Collect per-query data
    query_probs = []
    per_query_coverage = []
    per_query_missing = []
    per_query_letters_present: list[set[str]] = []

    for ql in query_log:
        probs = ql.get("canonical_probs", [])
        if len(probs) == NUM_CHOICES:
            query_probs.append(probs)
        else:
            query_probs.append([0.25] * NUM_CHOICES)

        raw_logprobs = ql.get("raw_logprobs", [])
        answer_entry = _find_answer_logprobs_entry(raw_logprobs, prompt_mode)
        if answer_entry is not None:
            top_lp = _get_top_logprobs(answer_entry)
            if top_lp:
                token_probs = [(e.get("token", ""), math.exp(e.get("logprob", -100)))
                               for e in top_lp]
                total_mass = sum(p for _, p in token_probs)
                answer_mass = sum(p for tok, p in token_probs if is_answer_token(tok))
                per_query_coverage.append(
                    answer_mass / total_mass if total_mass > 0 else float("nan")
                )
                present = set()
                for tok, _ in token_probs:
                    letter = _token_to_letter(tok)
                    if letter is not None:
                        present.add(letter)
                per_query_missing.append(NUM_CHOICES - len(present))
                per_query_letters_present.append(present)
            else:
                per_query_coverage.append(float("nan"))
                per_query_missing.append(float("nan"))
                per_query_letters_present.append(set(ANSWER_LETTERS))
        else:
            per_query_coverage.append(float("nan"))
            per_query_missing.append(float("nan"))
            per_query_letters_present.append(set(ANSWER_LETTERS))

    probs_arr = np.array(query_probs)
    mean_p = np.array(mean_probs)

    signals: dict = {}

    # Vote-based
    per_query_argmax = [int(np.argmax(p)) for p in query_probs]
    signals["agreement"] = sum(1 for a in per_query_argmax if a == final_answer) / n

    vote_counts = Counter(per_query_argmax)
    vote_dist = np.array([vote_counts.get(i, 0) for i in range(NUM_CHOICES)], dtype=float)
    vote_dist = vote_dist / vote_dist.sum()
    signals["vote_entropy"] = float(scipy_entropy(vote_dist, base=2))

    # Distribution-based
    signals["mean_confidence"] = float(np.max(mean_p))
    signals["total_uncertainty"] = float(scipy_entropy(mean_p, base=2))

    per_query_h = [float(scipy_entropy(p, base=2)) for p in query_probs]
    signals["aleatoric"] = float(np.mean(per_query_h))
    signals["epistemic"] = max(0.0, signals["total_uncertainty"] - signals["aleatoric"])

    per_query_conf = [float(np.max(p)) for p in query_probs]
    signals["confidence_variance"] = float(np.std(per_query_conf))

    sorted_mean = sorted(mean_probs, reverse=True)
    signals["second_gap_agg"] = sorted_mean[0] - sorted_mean[1]

    # Rank stability
    if n >= 2:
        rankings = np.argsort(-np.array(query_probs), axis=1)
        idx_i, idx_j = zip(*combinations(range(n), 2))
        idx_i, idx_j = np.array(idx_i), np.array(idx_j)
        Ri = rankings[idx_i]
        Rj = rankings[idx_j]
        K = rankings.shape[1]
        n_item_pairs = K * (K - 1) // 2
        concordant = np.zeros(len(idx_i))
        for a in range(K):
            for b in range(a + 1, K):
                concordant += ((Ri[:, a] - Ri[:, b]) * (Rj[:, a] - Rj[:, b])) > 0
        taus = (2 * concordant - n_item_pairs) / n_item_pairs
        signals["rank_stability"] = float(np.mean(taus))
    else:
        signals["rank_stability"] = float("nan")

    # Mean pairwise JSD
    if n >= 2:
        P = np.array(query_probs) + 1e-30
        idx_i, idx_j = zip(*combinations(range(n), 2))
        idx_i, idx_j = np.array(idx_i), np.array(idx_j)
        Pi = P[idx_i]
        Pj = P[idx_j]
        M = 0.5 * (Pi + Pj)
        kl_im = np.sum(Pi * np.log2(Pi / M), axis=1)
        kl_jm = np.sum(Pj * np.log2(Pj / M), axis=1)
        jsds = 0.5 * (kl_im + kl_jm)
        signals["mean_pairwise_jsd"] = float(np.mean(jsds))
    else:
        signals["mean_pairwise_jsd"] = float("nan")

    # Agreement-confidence gap
    signals["agreement_confidence_gap"] = (
        signals["agreement"] - signals["mean_confidence"]
    )

    # Effective option count
    h_nats = float(scipy_entropy(mean_p))
    signals["effective_option_count"] = math.exp(h_nats)

    # Original question diagnostic
    first_query_probs = query_probs[0]
    original_argmax = int(np.argmax(first_query_probs))
    signals["original_matches_aggregate"] = 1 if original_argmax == final_answer else 0

    # Full-vocabulary aggregated
    valid_coverage = [c for c in per_query_coverage if not math.isnan(c)]
    if valid_coverage:
        signals["agg_answer_coverage"] = float(np.mean(valid_coverage))
        signals["agg_answer_coverage_var"] = float(np.var(valid_coverage))
    else:
        signals["agg_answer_coverage"] = float("nan")
        signals["agg_answer_coverage_var"] = float("nan")

    valid_missing = [m for m in per_query_missing if not (isinstance(m, float) and math.isnan(m))]
    if valid_missing:
        signals["missing_letters_mean"] = float(np.mean(valid_missing))
    else:
        signals["missing_letters_mean"] = float("nan")

    # Consistent/fragile eliminations
    if per_query_letters_present:
        consistent_elim = 0
        fragile_elim = 0
        for letter in ANSWER_LETTERS:
            present_count = sum(1 for s in per_query_letters_present if letter in s)
            if present_count == 0:
                consistent_elim += 1
            elif present_count < n:
                fragile_elim += 1
        signals["consistent_eliminations"] = consistent_elim
        signals["fragile_eliminations"] = fragile_elim
    else:
        signals["consistent_eliminations"] = float("nan")
        signals["fragile_eliminations"] = float("nan")

    return signals


def _nan_tier2() -> dict:
    keys = [
        "agreement", "vote_entropy", "mean_confidence", "total_uncertainty",
        "aleatoric", "epistemic", "confidence_variance", "second_gap_agg",
        "rank_stability", "mean_pairwise_jsd", "agreement_confidence_gap",
        "effective_option_count", "original_matches_aggregate",
        "agg_answer_coverage", "agg_answer_coverage_var", "missing_letters_mean",
        "consistent_eliminations", "fragile_eliminations",
    ]
    return {k: float("nan") for k in keys}


# ---------------------------------------------------------------------------
# Tier III: Position signals (shuffle conditions only)
# ---------------------------------------------------------------------------

def compute_tier3_signals(
    query_log: list[dict], correct_answer: int | None
) -> dict:
    """Compute position-sensitivity signals."""
    n = len(query_log)
    if n <= 1:
        return _nan_tier3()

    query_probs = []
    permutations = []
    for ql in query_log:
        probs = ql.get("canonical_probs", [])
        perm = ql.get("answer_permutation", [0, 1, 2, 3])
        if len(probs) == NUM_CHOICES and len(perm) == NUM_CHOICES:
            query_probs.append(probs)
            permutations.append(perm)

    if len(query_probs) < 2:
        return _nan_tier3()

    probs_arr = np.array(query_probs)
    signals: dict = {}

    # Position loyalty
    per_option_var = np.var(probs_arr, axis=0)
    signals["position_loyalty"] = float(np.mean(per_option_var))

    # Correct answer position variance
    if correct_answer is not None and 0 <= correct_answer < NUM_CHOICES:
        signals["correct_answer_position_var"] = float(
            np.var(probs_arr[:, correct_answer])
        )
    else:
        signals["correct_answer_position_var"] = float("nan")

    # Position preference entropy
    display_winner_counts = Counter()
    for probs, perm in zip(query_probs, permutations):
        display_probs = [probs[perm[d]] for d in range(NUM_CHOICES)]
        winner_pos = int(np.argmax(display_probs))
        display_winner_counts[winner_pos] += 1

    winner_dist = np.array(
        [display_winner_counts.get(d, 0) for d in range(NUM_CHOICES)], dtype=float
    )
    winner_dist = winner_dist / winner_dist.sum()
    signals["position_preference_entropy"] = float(scipy_entropy(winner_dist, base=2))

    return signals


def _nan_tier3() -> dict:
    return {
        "position_loyalty": float("nan"),
        "correct_answer_position_var": float("nan"),
        "position_preference_entropy": float("nan"),
    }


# ---------------------------------------------------------------------------
# Main processing: one result file → rows
# ---------------------------------------------------------------------------

def process_result_file(data: dict) -> list[dict]:
    """Process a single result file into a list of row dicts."""
    cfg = data.get("config", {})
    condition = cfg.get("run_name", "unknown")
    prompt_mode = cfg.get("prompt_mode", "direct")
    context_condition = cfg.get("context_condition", "sufficient")

    rows = []
    for qr in data.get("question_results", []):
        query_log = qr.get("query_log", [])
        if not query_log:
            continue

        qid = qr["question_id"]
        correct_answer = qr.get("correct_answer")
        mean_probs = qr.get("mean_probs", [])
        final_answer = qr.get("final_answer", 0)
        num_queries = qr.get("num_queries", len(query_log))

        # --- QuALITY metadata ---
        article_id = qr.get("article_id", "")
        difficult = qr.get("difficult", False)
        question_text = qr.get("question_text", "")
        options = qr.get("options", [])
        word_count = len(question_text.split())

        row: dict = {
            "question_id": qid,
            "question_text": question_text,
            "answers": " | ".join(options),
            "article_id": article_id,
            "context_condition": context_condition,
            "difficult": difficult,
            "correct_answer": correct_answer,
            "condition": condition,
            "condition_tidy": _tidy_condition(condition),
            "prompt_mode": prompt_mode,
            "final_answer": final_answer,
            "is_correct": (final_answer == correct_answer) if correct_answer is not None else None,
            "num_queries": num_queries,
            "question_word_count": word_count,
        }

        # --- Alternative aggregations ---
        query_probs = [
            ql.get("canonical_probs", [0.25] * NUM_CHOICES)
            for ql in query_log
            if len(ql.get("canonical_probs", [])) == NUM_CHOICES
        ]
        if not query_probs:
            query_probs = [[0.25] * NUM_CHOICES]

        if num_queries > 1:
            agg = compute_alternative_aggregations(query_probs, mean_probs)
        else:
            agg = {
                "answer_majority_vote": final_answer,
                "answer_weighted_vote": final_answer,
                "answer_geometric_mean": final_answer,
                "answer_median": final_answer,
            }

        for method in ["majority_vote", "weighted_vote", "geometric_mean", "median"]:
            ans = agg[f"answer_{method}"]
            row[f"answer_{method}"] = ans
            row[f"correct_{method}"] = (ans == correct_answer) if correct_answer is not None else None

        # --- Tier I ---
        tier1 = compute_tier1_signals(query_log[0], prompt_mode)
        row.update(tier1)

        # --- Tier II ---
        tier2 = compute_tier2_signals(query_log, mean_probs, final_answer, prompt_mode)
        row.update(tier2)

        # --- Aggregation agreement ---
        if num_queries > 1:
            methods_agreeing = sum(
                1 for m in ["majority_vote", "weighted_vote", "geometric_mean", "median"]
                if agg[f"answer_{m}"] == final_answer
            ) + 1
            row["aggregation_agreement"] = methods_agreeing
        else:
            row["aggregation_agreement"] = 5

        # --- Tier III (always on — we always shuffle) ---
        if num_queries > 1:
            tier3 = compute_tier3_signals(query_log, correct_answer)
        else:
            tier3 = _nan_tier3()
        row.update(tier3)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_data_summary(all_rows: list[dict], file_summaries: list[dict]) -> None:
    print("\n=== DATA LOADED ===")
    print(f"  Results files: {len(file_summaries)}")
    for fs in file_summaries:
        print(f"    - {fs['name']} ({fs['n_questions']} questions, "
              f"{fs['prompt_mode']}, {fs['context_condition']})")
    print(f"  Total rows in CSV: {len(all_rows):,}")


def print_accuracy_by_condition(df: pd.DataFrame) -> None:
    print("\n=== ACCURACY BY CONDITION ===")
    valid = df[df["is_correct"].notna()]
    grouped = valid.groupby("condition").agg(
        Accuracy=("is_correct", "mean"),
        N=("is_correct", "count"),
    ).sort_values("Accuracy", ascending=False)
    print(f"  {'Condition':<55} {'Accuracy':>8}  {'N':>6}")
    for cond, row in grouped.iterrows():
        print(f"  {cond:<55} {row['Accuracy']:>7.1%}  {row['N']:>6.0f}")

    # Breakdown by difficulty
    if "difficult" in df.columns:
        print("\n  By difficulty:")
        for diff_val in [False, True]:
            label = "Hard" if diff_val else "Easy"
            sub = valid[valid["difficult"] == diff_val]
            if len(sub) > 0:
                acc = sub["is_correct"].mean()
                print(f"    {label}: {acc:.1%} (N={len(sub)})")

    # Breakdown by context condition
    if "context_condition" in df.columns:
        print("\n  By context condition:")
        for cc in sorted(valid["context_condition"].unique()):
            sub = valid[valid["context_condition"] == cc]
            if len(sub) > 0:
                acc = sub["is_correct"].mean()
                print(f"    {cc}: {acc:.1%} (N={len(sub)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty signals from QuALITY experiment results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing result JSON files")
    parser.add_argument("--output", type=Path, default=Path("analysis/signals.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    results_dir = args.results_dir if args.results_dir.is_absolute() else project_root / args.results_dir
    output_path = args.output if args.output.is_absolute() else project_root / args.output

    # Find and process result files
    result_files = sorted(
        f for f in results_dir.glob("*.json")
        if not f.stem.endswith(".tmp")
    )
    if not result_files:
        print(f"ERROR: No result files found in {results_dir}")
        sys.exit(1)

    all_rows: list[dict] = []
    file_summaries: list[dict] = []

    for fp in result_files:
        t0 = time.time()
        print(f"\nProcessing {fp.name}...")
        data = load_result_file(fp)
        if data is None:
            continue

        cfg = data.get("config", {})
        n_questions = len(data.get("question_results", []))
        if n_questions == 0:
            print(f"  Skipping — no question results.")
            continue

        file_summaries.append({
            "name": cfg.get("run_name", fp.stem),
            "n_questions": n_questions,
            "prompt_mode": cfg.get("prompt_mode", "?"),
            "context_condition": cfg.get("context_condition", "?"),
        })

        rows = process_result_file(data)
        all_rows.extend(rows)

        elapsed = time.time() - t0
        print(f"  {len(rows):,} rows in {elapsed:.1f}s")
        del data

    if not all_rows:
        print("ERROR: No rows produced.")
        sys.exit(1)

    # Build DataFrame and save
    df = pd.DataFrame(all_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Terminal summary
    print_data_summary(all_rows, file_summaries)
    print_accuracy_by_condition(df)

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"\n=== OUTPUT ===")
    print(f"  Saved {len(df):,} rows x {len(df.columns)} columns to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
