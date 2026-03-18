"""
Compute per-question uncertainty signals from experiment result files.

Reads all result JSONs from results/, computes Tier I (single-query),
Tier II (aggregated), and Tier III (position) signals, and writes a CSV
with one row per question per condition.

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
from scipy.stats import entropy as scipy_entropy, kendalltau
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_LETTERS = ["A", "B", "C", "D"]
NUM_CHOICES = 4


# ---------------------------------------------------------------------------
# Helper: answer token detection (mirrors query.py logic)
# ---------------------------------------------------------------------------

def is_answer_token(token_str: str) -> bool:
    """Check if a token string is an answer letter (with or without leading space)."""
    return token_str.strip() in ("A", "B", "C", "D")


def _token_to_letter(token_str: str) -> str | None:
    """Map a token like 'B' or ' B' to the canonical letter. None if not a match."""
    stripped = token_str.strip()
    return stripped if stripped in ("A", "B", "C", "D") else None


def _get_top_logprobs(logprobs_entry: dict) -> list[dict]:
    """Extract top_logprobs list from a logprobs entry (handles Ollama format variants)."""
    if isinstance(logprobs_entry, list):
        return logprobs_entry
    if "top_logprobs" in logprobs_entry:
        return logprobs_entry["top_logprobs"]
    return []


def _find_answer_logprobs_entry(
    raw_logprobs: list[dict], prompt_mode: str
) -> dict | None:
    """Find the logprobs entry for the answer token position.

    For direct mode: raw_logprobs[0] (num_predict=1, only one entry).
    For CoT modes: scan from the end, find the last position where the
    top token is A/B/C/D. This is where the model stated its final answer.
    """
    if not raw_logprobs:
        return None

    if prompt_mode == "direct":
        return raw_logprobs[0]

    # CoT modes: scan backwards for the last answer token
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

def load_questions_db(questions_path: Path) -> dict[str, dict]:
    """Load data/questions.json, keyed by question_id."""
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    return {q["question_id"]: q for q in questions}


def load_result_file(path: Path) -> dict | None:
    """Load a result JSON, returning None on error. Uses orjson if available."""
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
    """Compute single-prompt signals from the first query of a question."""
    signals: dict = {}
    probs = first_query.get("canonical_probs", [])

    if len(probs) != NUM_CHOICES:
        # Malformed — return all NaN
        return _nan_tier1()

    # --- From normalised canonical_probs ---
    sorted_desc = sorted(probs, reverse=True)

    signals["msp"] = sorted_desc[0]
    signals["single_entropy"] = float(scipy_entropy(probs, base=2))
    signals["second_gap"] = sorted_desc[0] - sorted_desc[1]

    # Effective option count = exp(H) where H is in nats
    h_nats = float(scipy_entropy(probs))  # base e by default
    signals["effective_option_count_single"] = math.exp(h_nats)

    # Distribution shape classification (order: peaked → bimodal → flat → spread)
    if sorted_desc[0] > 0.80:
        signals["distribution_shape"] = "peaked"
    elif sorted_desc[0] + sorted_desc[1] > 0.80 and sorted_desc[1] > 0.15:
        signals["distribution_shape"] = "bimodal"
    elif sorted_desc[0] < 0.35:
        signals["distribution_shape"] = "flat"
    else:
        signals["distribution_shape"] = "spread"

    # --- From full top-20 logprobs ---
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
    # Convert logprobs to probabilities for the top-20 tokens
    token_probs = []
    for entry in top_logprobs:
        lp = entry.get("logprob", -100)
        token_probs.append((entry.get("token", ""), math.exp(lp)))

    total_mass = sum(p for _, p in token_probs)
    answer_mass = sum(p for tok, p in token_probs if is_answer_token(tok))

    signals["answer_coverage"] = answer_mass / total_mass if total_mass > 0 else float("nan")
    signals["hesitation_mass"] = 1.0 - signals["answer_coverage"]

    # Top token is answer?
    if token_probs:
        signals["top_token_is_answer"] = 1 if is_answer_token(token_probs[0][0]) else 0
    else:
        signals["top_token_is_answer"] = float("nan")

    # Missing letters: which of A, B, C, D are NOT in the top-20 at all?
    present_letters = set()
    for tok, _ in token_probs:
        letter = _token_to_letter(tok)
        if letter is not None:
            present_letters.add(letter)
    signals["missing_letter_count"] = NUM_CHOICES - len(present_letters)


def _set_full_vocab_nan(signals: dict) -> None:
    """Set full-vocabulary signals to NaN when raw_logprobs unavailable."""
    signals["answer_coverage"] = float("nan")
    signals["hesitation_mass"] = float("nan")
    signals["top_token_is_answer"] = float("nan")
    signals["missing_letter_count"] = float("nan")


def _nan_tier1() -> dict:
    """Return a Tier I signal dict with all NaN values."""
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
    """Compute final answers under multiple aggregation strategies.

    Returns a dict with answer_{method}, correct_{method} (correctness
    filled in by the caller since we don't have the ground truth here).
    """
    n = len(query_probs)
    probs_arr = np.array(query_probs)
    mean_argmax = int(np.argmax(mean_probs))

    # Majority vote: mode of per-query argmax, ties broken by mean_probs argmax
    per_query_votes = [int(np.argmax(p)) for p in query_probs]
    vote_counts = Counter(per_query_votes)
    max_count = max(vote_counts.values())
    top_votes = [ans for ans, cnt in vote_counts.items() if cnt == max_count]
    if len(top_votes) == 1:
        majority = top_votes[0]
    else:
        # Tie: fall back to mean_probs argmax
        majority = mean_argmax

    # Weighted vote: each vote weighted by max(prob) of that query
    weighted_scores = np.zeros(NUM_CHOICES)
    for p in query_probs:
        winner = int(np.argmax(p))
        weighted_scores[winner] += max(p)
    weighted = int(np.argmax(weighted_scores))

    # Geometric mean: exp(mean(log(probs + floor)))
    log_probs = np.log(probs_arr + 1e-30)
    geo_mean = np.exp(log_probs.mean(axis=0))
    geo_mean = geo_mean / geo_mean.sum()  # normalise
    geo = int(np.argmax(geo_mean))

    # Median: element-wise median
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
    """Compute across-query aggregated signals. Returns NaN for single-query."""
    n = len(query_log)
    if n <= 1:
        return _nan_tier2()

    # Collect per-query probability vectors and per-query full-vocab signals
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

        # Full-vocab signals per query
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

    # --- Vote-based ---
    per_query_argmax = [int(np.argmax(p)) for p in query_probs]
    signals["agreement"] = sum(1 for a in per_query_argmax if a == final_answer) / n

    vote_counts = Counter(per_query_argmax)
    vote_dist = np.array([vote_counts.get(i, 0) for i in range(NUM_CHOICES)], dtype=float)
    vote_dist = vote_dist / vote_dist.sum()
    signals["vote_entropy"] = float(scipy_entropy(vote_dist, base=2))

    # --- Distribution-based ---
    signals["mean_confidence"] = float(np.max(mean_p))
    signals["total_uncertainty"] = float(scipy_entropy(mean_p, base=2))

    per_query_h = [float(scipy_entropy(p, base=2)) for p in query_probs]
    signals["aleatoric"] = float(np.mean(per_query_h))
    signals["epistemic"] = max(0.0, signals["total_uncertainty"] - signals["aleatoric"])

    per_query_conf = [float(np.max(p)) for p in query_probs]
    signals["confidence_variance"] = float(np.std(per_query_conf))

    sorted_mean = sorted(mean_probs, reverse=True)
    signals["second_gap_agg"] = sorted_mean[0] - sorted_mean[1]

    # Rank stability: mean pairwise Kendall's tau.
    # With K=4 options, tau = 1 - 2*discordant_pairs/C(4,2). Vectorized
    # by computing all pairwise concordance counts with numpy broadcasting.
    if n >= 2:
        rankings = np.argsort(-np.array(query_probs), axis=1)  # (n, K)
        idx_i, idx_j = zip(*combinations(range(n), 2))
        idx_i, idx_j = np.array(idx_i), np.array(idx_j)
        Ri = rankings[idx_i]  # (n_pairs, K)
        Rj = rankings[idx_j]
        # For each pair, count concordant/discordant pairs among K items
        K = rankings.shape[1]
        n_item_pairs = K * (K - 1) // 2  # C(K, 2)
        concordant = np.zeros(len(idx_i))
        for a in range(K):
            for b in range(a + 1, K):
                concordant += ((Ri[:, a] - Ri[:, b]) * (Rj[:, a] - Rj[:, b])) > 0
        taus = (2 * concordant - n_item_pairs) / n_item_pairs
        signals["rank_stability"] = float(np.mean(taus))
    else:
        signals["rank_stability"] = float("nan")

    # Mean pairwise JSD: average Jensen-Shannon divergence between all pairs
    # of individual probability vectors. Different from epistemic uncertainty
    # (which is JSD from the mean). This catches cases where distributions are
    # drifting around even when argmax agreement is perfect.
    # Vectorized: compute all pairwise KL divergences at once using numpy.
    if n >= 2:
        P = np.array(query_probs) + 1e-30  # (n, K) with floor for log safety
        # For each pair (i, j): JSD = 0.5 * (KL(Pi||M) + KL(Pj||M)), M = 0.5*(Pi+Pj)
        # Vectorize over all C(n,2) pairs using index arrays
        idx_i, idx_j = zip(*combinations(range(n), 2))
        idx_i, idx_j = np.array(idx_i), np.array(idx_j)
        Pi = P[idx_i]  # (n_pairs, K)
        Pj = P[idx_j]
        M = 0.5 * (Pi + Pj)
        # KL(P||Q) = sum(P * log2(P/Q))
        kl_im = np.sum(Pi * np.log2(Pi / M), axis=1)
        kl_jm = np.sum(Pj * np.log2(Pj / M), axis=1)
        jsds = 0.5 * (kl_im + kl_jm)
        signals["mean_pairwise_jsd"] = float(np.mean(jsds))
    else:
        signals["mean_pairwise_jsd"] = float("nan")

    # Agreement-confidence gap: agreement - mean_confidence.
    # High agreement + low confidence = the model consistently barely picks
    # the same answer. This is a red flag (the NMR case: agreement=1.00,
    # confidence=0.502). Large positive gap = "always agrees but isn't sure."
    signals["agreement_confidence_gap"] = (
        signals["agreement"] - signals["mean_confidence"]
    )

    # Effective option count
    h_nats = float(scipy_entropy(mean_p))  # base e
    signals["effective_option_count"] = math.exp(h_nats)

    # Original question diagnostic
    first_query_probs = query_probs[0]
    original_argmax = int(np.argmax(first_query_probs))
    signals["original_matches_aggregate"] = 1 if original_argmax == final_answer else 0

    # --- Full-vocabulary aggregated ---
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

    # Consistent eliminations: options missing from top-20 on EVERY query
    # Fragile eliminations: missing on at least one but not all
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
    """Return Tier II signals as NaN (for single-query conditions)."""
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
    """Compute position-sensitivity signals. Only for shuffle=True + multi-query."""
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

    probs_arr = np.array(query_probs)  # (n, 4)
    signals: dict = {}

    # Position loyalty: average variance of each canonical option's prob across queries
    per_option_var = np.var(probs_arr, axis=0)  # (4,)
    signals["position_loyalty"] = float(np.mean(per_option_var))

    # Correct answer position variance
    if correct_answer is not None and 0 <= correct_answer < NUM_CHOICES:
        signals["correct_answer_position_var"] = float(
            np.var(probs_arr[:, correct_answer])
        )
    else:
        signals["correct_answer_position_var"] = float("nan")

    # Position preference entropy: which DISPLAY position gets highest prob?
    # For display pos d, canonical option = perm[d], so display prob = probs[perm[d]]
    display_winner_counts = Counter()
    for probs, perm in zip(query_probs, permutations):
        # Map canonical probs to display positions
        display_probs = [probs[perm[d]] for d in range(NUM_CHOICES)]
        winner_pos = int(np.argmax(display_probs))
        display_winner_counts[winner_pos] += 1

    n_queries = len(query_probs)
    winner_dist = np.array(
        [display_winner_counts.get(d, 0) for d in range(NUM_CHOICES)], dtype=float
    )
    winner_dist = winner_dist / winner_dist.sum()
    signals["position_preference_entropy"] = float(scipy_entropy(winner_dist, base=2))

    return signals


def _nan_tier3() -> dict:
    """Return Tier III signals as NaN."""
    return {
        "position_loyalty": float("nan"),
        "correct_answer_position_var": float("nan"),
        "position_preference_entropy": float("nan"),
    }


# ---------------------------------------------------------------------------
# Main processing: one result file → rows
# ---------------------------------------------------------------------------

def process_result_file(
    data: dict, questions_db: dict[str, dict]
) -> list[dict]:
    """Process a single result file into a list of row dicts."""
    cfg = data.get("config", {})
    condition = cfg.get("run_name", "unknown")
    prompt_mode = cfg.get("prompt_mode", "direct")
    shuffle = bool(cfg.get("shuffle_choices", False))
    para = bool(cfg.get("use_paraphrases", False))

    rows = []
    for qr in data.get("question_results", []):
        query_log = qr.get("query_log", [])
        if not query_log:
            continue

        qid = qr["question_id"]
        q_info = questions_db.get(qid, {})
        correct_answer = q_info.get("correct_answer")
        mean_probs = qr.get("mean_probs", [])
        final_answer = qr.get("final_answer", 0)
        num_queries = qr.get("num_queries", len(query_log))

        # --- Metadata columns ---
        row: dict = {
            "question_id": qid,
            "subject": q_info.get("subject", "unknown"),
            "correct_answer": correct_answer,
            "condition": condition,
            "prompt_mode": prompt_mode,
            "shuffle": shuffle,
            "para": para,
            "final_answer": final_answer,
            "is_correct": (final_answer == correct_answer) if correct_answer is not None else None,
            "num_queries": num_queries,
        }

        # --- Alternative aggregation methods ---
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
            # Single query: all methods agree
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

        # --- Tier I: single-prompt signals (first query) ---
        tier1 = compute_tier1_signals(query_log[0], prompt_mode)
        row.update(tier1)

        # --- Tier II: aggregated signals ---
        tier2 = compute_tier2_signals(query_log, mean_probs, final_answer, prompt_mode)
        row.update(tier2)

        # --- Aggregation agreement ---
        if num_queries > 1:
            methods_agreeing = sum(
                1 for m in ["majority_vote", "weighted_vote", "geometric_mean", "median"]
                if agg[f"answer_{m}"] == final_answer
            ) + 1  # +1 for mean_probs itself
            row["aggregation_agreement"] = methods_agreeing
        else:
            row["aggregation_agreement"] = 5  # all identical for single query

        # --- Tier III: position signals ---
        if shuffle and num_queries > 1:
            tier3 = compute_tier3_signals(query_log, correct_answer)
        else:
            tier3 = _nan_tier3()
        row.update(tier3)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Terminal output: summary sections
# ---------------------------------------------------------------------------

def print_data_summary(all_rows: list[dict], file_summaries: list[dict]) -> None:
    """Section 1: Data loaded."""
    print("\n=== DATA LOADED ===")
    print(f"  Results files: {len(file_summaries)}")
    for fs in file_summaries:
        print(f"    - {fs['name']} ({fs['n_questions']} questions, {fs['prompt_mode']}, "
              f"{'shuffle' if fs['shuffle'] else 'no shuffle'}, "
              f"{'para' if fs['para'] else 'no para'})")
    print(f"  Total rows in CSV: {len(all_rows):,}")


def print_accuracy_by_condition(df: pd.DataFrame) -> None:
    """Section 2: Accuracy per condition."""
    print("\n=== ACCURACY BY CONDITION ===")
    valid = df[df["is_correct"].notna()]
    grouped = valid.groupby("condition").agg(
        Accuracy=("is_correct", "mean"),
        N=("is_correct", "count"),
    ).sort_values("Accuracy", ascending=False)
    print(f"  {'Condition':<45} {'Accuracy':>8}  {'N':>6}")
    for cond, row in grouped.iterrows():
        print(f"  {cond:<45} {row['Accuracy']:>7.1%}  {row['N']:>6.0f}")


def print_aggregation_comparison(df: pd.DataFrame) -> None:
    """Section 3: Aggregation method comparison."""
    print("\n=== AGGREGATION METHOD COMPARISON ===")
    multi = df[df["num_queries"] > 1].copy()
    if multi.empty:
        print("  No multi-query conditions found.")
        return

    methods = ["is_correct", "correct_majority_vote", "correct_weighted_vote",
               "correct_geometric_mean", "correct_median"]
    labels = ["MeanProbs", "MajVote", "WeightedVote", "GeoMean", "Median"]

    header = f"  {'Condition':<45}" + "".join(f"{l:>13}" for l in labels)
    print(header)

    conditions = sorted(multi["condition"].unique())
    method_means = {l: [] for l in labels}
    for cond in conditions:
        cond_df = multi[multi["condition"] == cond]
        valid = cond_df[cond_df["is_correct"].notna()]
        vals = []
        for m, l in zip(methods, labels):
            acc = valid[m].mean() if m in valid.columns and valid[m].notna().any() else float("nan")
            vals.append(acc)
            if not math.isnan(acc):
                method_means[l].append(acc)
        line = f"  {cond:<45}" + "".join(f"{v:>12.1%} " if not math.isnan(v) else f"{'—':>13}" for v in vals)
        print(line)

    # Best method
    avg_by_method = {l: np.mean(v) if v else 0 for l, v in method_means.items()}
    best = max(avg_by_method, key=avg_by_method.get)
    print(f"\n  Best method overall: {best} (highest mean accuracy across conditions)")

    # Disagreement analysis
    valid_multi = multi[multi["is_correct"].notna()].copy()
    if not valid_multi.empty:
        valid_multi["all_agree"] = valid_multi["aggregation_agreement"] == 5
        n_disagree = (~valid_multi["all_agree"]).sum()
        n_total = len(valid_multi)
        print(f"  Questions where methods DISAGREE: {n_disagree:,} / {n_total:,} "
              f"({n_disagree / n_total:.1%})")
        if n_disagree > 0:
            agree_acc = valid_multi[valid_multi["all_agree"]]["is_correct"].mean()
            disagree_acc = valid_multi[~valid_multi["all_agree"]]["is_correct"].mean()
            print(f"    -> Accuracy when all methods agree:  {agree_acc:.1%}")
            print(f"    -> Accuracy when methods disagree:   {disagree_acc:.1%}")


def _compute_auroc_leaderboard(
    cond_df: pd.DataFrame,
) -> list[tuple[str, float, str]]:
    """Compute AUROC for all numeric signals on a single condition's data.

    Returns a sorted list of (signal_name, auroc, direction) tuples.
    """
    y = cond_df["is_correct"].astype(int)

    # All numeric signal columns to evaluate
    all_signals = [
        "msp", "mean_confidence", "agreement", "answer_coverage",
        "second_gap", "second_gap_agg", "rank_stability",
        "effective_option_count_single", "effective_option_count",
        "agg_answer_coverage", "original_matches_aggregate",
        "top_token_is_answer", "aggregation_agreement",
        "single_entropy", "total_uncertainty", "aleatoric", "epistemic",
        "confidence_variance", "vote_entropy", "hesitation_mass",
        "mean_pairwise_jsd", "agreement_confidence_gap",
        "missing_letter_count", "missing_letters_mean",
        "position_loyalty", "correct_answer_position_var",
        "agg_answer_coverage_var", "consistent_eliminations",
        "fragile_eliminations", "cot_response_length",
        "position_preference_entropy",
    ]

    results = []
    for signal in all_signals:
        if signal not in cond_df.columns:
            continue
        vals = cond_df[signal]
        valid_mask = vals.notna() & np.isfinite(vals.astype(float))
        if valid_mask.sum() < 20:
            continue
        y_sub = y[valid_mask]
        v_sub = vals[valid_mask].astype(float)
        if y_sub.nunique() < 2:
            continue
        try:
            raw_auroc = roc_auc_score(y_sub, v_sub)
        except Exception:
            continue

        # Normalise: always report AUROC > 0.5, with direction indicating
        # whether the signal predicts correctness when high or low.
        # This avoids hardcoding assumptions about ambiguous signals
        # (e.g. missing_letter_count could go either way).
        if raw_auroc >= 0.5:
            auroc = raw_auroc
            direction = "higher = more likely correct"
        else:
            auroc = 1.0 - raw_auroc
            direction = "lower = more likely correct"

        results.append((signal, auroc, direction))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _pick_best_condition(
    df: pd.DataFrame, target_prompt_mode: str | None = None,
) -> str | None:
    """Pick the fullest multi-query condition, preferring shuffle+para.

    If target_prompt_mode is given, only consider conditions with that prompt mode.
    """
    multi = df[(df["num_queries"] > 1) & (df["is_correct"].notna())]
    if target_prompt_mode:
        multi = multi[multi["prompt_mode"] == target_prompt_mode]
    if multi.empty:
        return None

    candidates = multi.groupby("condition").size().reset_index(name="n")
    # Prefer shuffle+para
    for _, r in candidates.sort_values("n", ascending=False).iterrows():
        cond_df = multi[multi["condition"] == r["condition"]]
        cfg_row = cond_df.iloc[0]
        if cfg_row.get("shuffle") and cfg_row.get("para"):
            return r["condition"]
    # Fall back to largest
    return candidates.sort_values("n", ascending=False).iloc[0]["condition"]


def _print_auroc_table(label: str, cond_name: str, results: list) -> None:
    """Print one AUROC leaderboard."""
    print(f"\n=== SIGNAL AUROC — {label}: {cond_name} ===")
    print(f"  {'Rank':<6}{'Signal':<30}{'AUROC':>7}   Direction")
    for i, (sig, auroc, direction) in enumerate(results, 1):
        print(f"  {i:<6}{sig:<30}{auroc:>7.3f}   {direction}")


def print_signal_auroc(df: pd.DataFrame) -> tuple | None:
    """Section 4: Signal AUROC leaderboard — one for DIRECT, one for COT."""
    multi = df[(df["num_queries"] > 1) & (df["is_correct"].notna())].copy()
    if multi.empty:
        print("\n=== SIGNAL AUROC ===\n  No multi-query conditions with correctness data.")
        return None

    direct_results = None
    direct_cond = None
    cot_results = None
    cot_cond = None

    # Direct condition leaderboard
    direct_cond = _pick_best_condition(df, target_prompt_mode="direct")
    if direct_cond:
        cond_df = multi[multi["condition"] == direct_cond].copy()
        direct_results = _compute_auroc_leaderboard(cond_df)
        _print_auroc_table("DIRECT", direct_cond, direct_results)

    # CoT condition leaderboard (try cot first, then cot_structured)
    cot_cond = _pick_best_condition(df, target_prompt_mode="cot")
    if cot_cond is None:
        cot_cond = _pick_best_condition(df, target_prompt_mode="cot_structured")
    if cot_cond:
        cond_df = multi[multi["condition"] == cot_cond].copy()
        cot_results = _compute_auroc_leaderboard(cond_df)
        _print_auroc_table("COT", cot_cond, cot_results)
        if direct_results:
            print("\n  Note: Tier I signals (msp, single_entropy, answer_coverage) are expected")
            print("  to have lower AUROC in CoT due to the logprob absorption effect —")
            print("  the reasoning chain commits to an answer before the answer token.")

    if not direct_results and not cot_results:
        print("\n=== SIGNAL AUROC ===\n  No suitable conditions found.")
        return None

    # Return the direct results for key comparisons (preferred); fall back to CoT
    primary = direct_results or cot_results
    primary_cond = direct_cond or cot_cond
    return primary, primary_cond


def print_key_comparisons(auroc_results: list | None, cond_name: str | None) -> None:
    """Section 5: Key comparisons."""
    print(f"\n=== KEY COMPARISONS ===")
    if not auroc_results:
        print("  No AUROC results available.")
        return

    lookup = {sig: (auroc, d) for sig, auroc, d in auroc_results}
    msp_auroc = lookup.get("msp", (None, ""))[0]

    comparisons = [
        ("Single-prompt MSP baseline", "msp"),
        ("Aggregated mean_confidence", "mean_confidence"),
        ("Agreement rate", "agreement"),
        ("Epistemic uncertainty", "epistemic"),
        ("Novel: answer_coverage (single query)", "answer_coverage"),
        ("Novel: missing_letters_mean", "missing_letters_mean"),
    ]

    for label, signal in comparisons:
        val = lookup.get(signal)
        if val is None:
            continue
        auroc, _ = val
        delta = f"  (+{auroc - msp_auroc:.3f} vs MSP)" if msp_auroc is not None and signal != "msp" else ""
        print(f"  {label + ':':<45} AUROC = {auroc:.3f}{delta}")


def print_2d_space(df: pd.DataFrame) -> None:
    """Section 6: 2D uncertainty space (median split)."""
    print("\n=== 2D UNCERTAINTY SPACE (median split) ===")
    multi = df[(df["num_queries"] > 1) & (df["is_correct"].notna())].copy()
    if multi.empty or "mean_confidence" not in multi.columns:
        print("  No multi-query data available.")
        return

    valid = multi[multi["mean_confidence"].notna() & multi["agreement"].notna()].copy()
    if len(valid) < 10:
        print("  Not enough data for median split.")
        return

    conf_med = valid["mean_confidence"].median()
    agree_med = valid["agreement"].median()

    quadrants = [
        ("High confidence + High agreement", (valid["mean_confidence"] >= conf_med) & (valid["agreement"] >= agree_med)),
        ("High confidence + Low agreement",  (valid["mean_confidence"] >= conf_med) & (valid["agreement"] < agree_med)),
        ("Low confidence + High agreement",  (valid["mean_confidence"] < conf_med) & (valid["agreement"] >= agree_med)),
        ("Low confidence + Low agreement",   (valid["mean_confidence"] < conf_med) & (valid["agreement"] < agree_med)),
    ]

    print(f"  {'Quadrant':<40} {'N':>6}   {'Accuracy':>8}")
    for label, mask in quadrants:
        subset = valid[mask]
        n = len(subset)
        acc = subset["is_correct"].mean() if n > 0 else float("nan")
        acc_str = f"{acc:.1%}" if not math.isnan(acc) else "—"
        print(f"  {label:<40} {n:>6}   {acc_str:>8}")


def print_novel_spotlight(df: pd.DataFrame) -> None:
    """Section 7: Novel signal spotlight."""
    print("\n=== NOVEL SIGNALS SPOTLIGHT ===")

    # Use the fullest multi-query condition
    multi = df[(df["num_queries"] > 1) & (df["is_correct"].notna())].copy()
    if multi.empty:
        print("  No multi-query data available.")
        return

    correct = multi[multi["is_correct"] == True]
    incorrect = multi[multi["is_correct"] == False]

    def _mean_safe(series):
        v = series.dropna()
        return float(v.mean()) if len(v) > 0 else float("nan")

    # Answer coverage (single query)
    ac_c = _mean_safe(correct["answer_coverage"])
    ac_i = _mean_safe(incorrect["answer_coverage"])
    print(f"  Answer coverage (single query):")
    print(f"    Mean (correct):   {ac_c:.2f}    Mean (incorrect): {ac_i:.2f}    "
          f"Diff: {ac_c - ac_i:.2f}")

    # Missing letters mean
    ml_c = _mean_safe(correct["missing_letters_mean"])
    ml_i = _mean_safe(incorrect["missing_letters_mean"])
    print(f"  Missing letters (mean across paraphrases):")
    print(f"    Mean (correct):   {ml_c:.1f}     Mean (incorrect): {ml_i:.1f}     "
          f"Diff: {ml_c - ml_i:.1f}")

    # Consistent eliminations
    ce_c = _mean_safe(correct["consistent_eliminations"])
    ce_i = _mean_safe(incorrect["consistent_eliminations"])
    print(f"  Consistent eliminations:")
    print(f"    Mean (correct):   {ce_c:.1f}     Mean (incorrect): {ce_i:.1f}     "
          f"Diff: {ce_c - ce_i:.1f}")

    # Top token not answer
    has_top = multi[multi["top_token_is_answer"].notna()]
    not_answer = has_top[has_top["top_token_is_answer"] == 0]
    overall_acc = multi["is_correct"].mean()
    n_not_ans = len(not_answer)
    pct_not_ans = n_not_ans / len(has_top) if len(has_top) > 0 else 0
    na_acc = not_answer["is_correct"].mean() if n_not_ans > 0 else float("nan")
    print(f"  Questions where top token is NOT an answer letter:")
    print(f"    N = {n_not_ans:,} ({pct_not_ans:.1%})    "
          f"Accuracy on these: {na_acc:.1%} (vs {overall_acc:.1%} overall)")

    # Aggregation disagreement
    disagree = multi[multi["aggregation_agreement"] < 5]
    n_dis = len(disagree)
    pct_dis = n_dis / len(multi) if len(multi) > 0 else 0
    dis_acc = disagree["is_correct"].mean() if n_dis > 0 else float("nan")
    print(f"  Aggregation methods disagree:")
    print(f"    N = {n_dis:,} ({pct_dis:.1%})    "
          f"Accuracy on these: {dis_acc:.1%} (vs {overall_acc:.1%} overall)")

    # Original != aggregate
    has_orig = multi[multi["original_matches_aggregate"].notna()]
    flipped = has_orig[has_orig["original_matches_aggregate"] == 0]
    n_flip = len(flipped)
    pct_flip = n_flip / len(has_orig) if len(has_orig) > 0 else 0
    flip_acc = flipped["is_correct"].mean() if n_flip > 0 else float("nan")
    print(f"  Original question answer != aggregate answer:")
    print(f"    N = {n_flip:,} ({pct_flip:.1%})    "
          f"Accuracy when flipped: {flip_acc:.1%} (vs {overall_acc:.1%} overall)")

    # Distribution shape breakdown
    has_shape = multi[multi["distribution_shape"].notna()]
    if not has_shape.empty:
        print(f"\n  Distribution shape breakdown (single query):")
        print(f"    {'Shape':<12} {'N':>6}  {'% of total':>10}   {'Accuracy':>8}")
        total_n = len(has_shape)
        for shape in ["peaked", "bimodal", "spread", "flat"]:
            subset = has_shape[has_shape["distribution_shape"] == shape]
            n = len(subset)
            pct = n / total_n if total_n > 0 else 0
            acc = subset["is_correct"].mean() if n > 0 else float("nan")
            acc_str = f"{acc:.1%}" if not math.isnan(acc) else "—"
            print(f"    {shape:<12} {n:>6}  {pct:>9.1%}   {acc_str:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute uncertainty signals from experiment results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing result JSON files")
    parser.add_argument("--output", type=Path, default=Path("analysis/signals.csv"),
                        help="Output CSV path")
    parser.add_argument("--questions", type=Path, default=Path("data/questions.json"),
                        help="Path to questions.json")
    args = parser.parse_args()

    # Resolve paths relative to project root if running as module
    project_root = Path(__file__).resolve().parent.parent
    results_dir = args.results_dir if args.results_dir.is_absolute() else project_root / args.results_dir
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    questions_path = args.questions if args.questions.is_absolute() else project_root / args.questions

    print("Loading question database...")
    questions_db = load_questions_db(questions_path)
    print(f"  {len(questions_db):,} questions loaded.")

    # Find and process result files (skip .tmp and old partial/superseded files)
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
            "shuffle": bool(cfg.get("shuffle_choices")),
            "para": bool(cfg.get("use_paraphrases")),
        })

        rows = process_result_file(data, questions_db)
        all_rows.extend(rows)

        elapsed = time.time() - t0
        print(f"  {len(rows):,} rows in {elapsed:.1f}s")

        # Release memory before loading next file
        del data

    if not all_rows:
        print("ERROR: No rows produced. Check result files.")
        sys.exit(1)

    # Build DataFrame and save CSV
    df = pd.DataFrame(all_rows)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # --- Terminal summary ---
    print_data_summary(all_rows, file_summaries)
    print_accuracy_by_condition(df)
    print_aggregation_comparison(df)

    auroc_result = print_signal_auroc(df)
    if auroc_result is not None:
        auroc_results, best_cond = auroc_result
        print_key_comparisons(auroc_results, best_cond)
    else:
        print_key_comparisons(None, None)

    print_2d_space(df)
    print_novel_spotlight(df)

    # Section 8: output summary
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"\n=== OUTPUT ===")
    print(f"  Saved {len(df):,} rows × {len(df.columns)} columns to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
