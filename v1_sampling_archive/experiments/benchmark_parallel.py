"""
Benchmark parallel worker counts for thinking mode.

Runs the same 6 questions with 1, 2, 3, and 4 parallel workers,
measuring wall-clock time, retries, and errors for each.

Usage:
    python experiments/benchmark_parallel.py
"""

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bayesian_uq.config import ParaphraseRecord
from bayesian_uq.dirichlet import exceedance_probability, init_prior, posterior_entropy, update_posterior
from bayesian_uq.query import OllamaClient, generate_permutation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"

# 6 fixed question IDs for the benchmark
TEST_QIDS = [
    "mmlu_redux_college_mathematics_0095",
    "mmlu_redux_astronomy_0028",
    "mmlu_redux_abstract_algebra_0002",
    "mmlu_redux_college_computer_science_0033",
    "mmlu_redux_high_school_psychology_0064",
    "mmlu_redux_jurisprudence_0040",
]

MODEL = "qwen3:8b-q4_K_M"
MAX_QUERIES = 20  # enough for convergence, not too many
THRESHOLD = 0.95
WORKER_COUNTS = [1, 2, 3, 4]


def load_test_questions():
    """Load just the 6 test questions."""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        all_qs = json.load(f)
    q_map = {q["question_id"]: q for q in all_qs}
    return [q_map[qid] for qid in TEST_QIDS]


def run_single_question(question, client, seed_offset):
    """Run one question through the pipeline, return stats."""
    rng = random.Random(42 + seed_offset)
    rng_mc = np.random.default_rng(42 + seed_offset)
    num_choices = len(question["choices"])
    alpha = init_prior(num_choices)

    retries = 0
    errors = 0
    queries_done = 0
    stopped_early = False

    for q_num in range(MAX_QUERIES):
        perm = generate_permutation(num_choices, rng, shuffle=False)
        try:
            raw, canonical, thinking = client.send_query(
                question_text=question["question_text"],
                choices=question["choices"],
                answer_permutation=perm,
            )
        except Exception as e:
            errors += 1
            continue

        queries_done += 1
        alpha = update_posterior(alpha, canonical)
        exc = exceedance_probability(alpha, 10000, rng_mc)

        if exc >= THRESHOLD:
            stopped_early = True
            break

    return {
        "qid": question["question_id"],
        "queries": queries_done,
        "errors": errors,
        "stopped_early": stopped_early,
    }


def benchmark_workers(questions, num_workers):
    """Run all questions with a given number of parallel workers."""
    print_lock = threading.Lock()
    results = []
    total_retries_approx = 0

    start = time.monotonic()

    def _run(idx):
        client = OllamaClient(model=MODEL, think=True, timeout=120)
        return run_single_question(questions[idx], client, seed_offset=idx)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run, i): i for i in range(len(questions))}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with print_lock:
                status = "EARLY" if result["stopped_early"] else "FULL"
                err_note = f" ({result['errors']} errors)" if result["errors"] else ""
                print(f"    {result['qid']}: {result['queries']}q {status}{err_note}")

    elapsed = time.monotonic() - start
    total_queries = sum(r["queries"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    return {
        "workers": num_workers,
        "elapsed": elapsed,
        "total_queries": total_queries,
        "total_errors": total_errors,
        "per_query": elapsed / total_queries if total_queries else 0,
        "results": results,
    }


def main():
    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = load_test_questions()
    print(f"Benchmark: {len(questions)} questions, think=True, model={MODEL}")
    print(f"Testing worker counts: {WORKER_COUNTS}")
    print(f"Max queries per question: {MAX_QUERIES}")
    print()

    # Verify Ollama is running
    client = OllamaClient(model=MODEL, think=True)
    if not client.check_connection():
        print("ERROR: Ollama not reachable. Start it with: ollama serve")
        sys.exit(1)

    # Warm up — ensure model is loaded in VRAM
    print("Warming up model...")
    client.send_query(
        question_text="What is 1+1?",
        choices=["1", "2", "3", "4"],
        answer_permutation=[0, 1, 2, 3],
    )
    print("Model warm. Starting benchmark.\n")

    all_results = []
    for n_workers in WORKER_COUNTS:
        print(f"--- {n_workers} worker(s) ---")
        result = benchmark_workers(questions, n_workers)
        all_results.append(result)
        print(
            f"  => {result['elapsed']:.1f}s total | "
            f"{result['total_queries']} queries | "
            f"{result['per_query']:.1f}s/query | "
            f"{result['total_errors']} errors"
        )
        print()

        # Brief pause between runs to let GPU cool / settle
        if n_workers < WORKER_COUNTS[-1]:
            time.sleep(5)

    # Summary table
    print("=" * 65)
    print(f"{'Workers':>8} {'Elapsed':>10} {'Queries':>8} {'s/query':>8} {'Errors':>7} {'Speedup':>8}")
    print("-" * 65)
    baseline = all_results[0]["elapsed"]
    for r in all_results:
        speedup = baseline / r["elapsed"] if r["elapsed"] > 0 else 0
        print(
            f"{r['workers']:>8} "
            f"{r['elapsed']:>9.1f}s "
            f"{r['total_queries']:>8} "
            f"{r['per_query']:>7.1f}s "
            f"{r['total_errors']:>7} "
            f"{speedup:>7.2f}x"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
