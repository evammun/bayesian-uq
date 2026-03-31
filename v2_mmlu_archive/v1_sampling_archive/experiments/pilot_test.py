"""
Pilot test: Bayesian UQ framework on 4 hardcoded questions (2 valid + 2 broken-premise).

For each question, runs 6 queries (original + 5 paraphrases) with shuffled answer order,
updates a Dirichlet posterior after each response, and computes exceedance probability
and entropy. Runs the full battery twice: once with thinking enabled, once without.

Usage:
    python experiments/pilot_test.py
"""

import json
import random
import requests
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b-q4_K_M"
NUM_MC_SAMPLES = 10_000          # Monte Carlo samples for exceedance probability
SEED = 42                        # For reproducibility of shuffles / MC draws

# ---------------------------------------------------------------------------
# Question bank: each entry has a label, the stem, and canonical answer choices
# (A/B/C/D in order). Ground-truth correct answer is noted for reference only —
# the framework doesn't use it.
# ---------------------------------------------------------------------------
QUESTIONS = [
    {
        "id": "Q1_valid",
        "label": "Mitochondria (valid)",
        "stem": "What is the primary function of the mitochondria in a cell?",
        "choices": {
            "A": "Protein synthesis",
            "B": "Energy production",
            "C": "Cell division",
            "D": "DNA replication",
        },
        "correct": "B",
        "paraphrases": [
            "What role do mitochondria primarily play inside a cell?",
            "Which function are mitochondria best known for in cellular biology?",
            "In a typical eukaryotic cell, what is the main job of the mitochondria?",
            "What is the chief responsibility of the mitochondria within cells?",
            "What do mitochondria mainly do for a cell?",
        ],
    },
    {
        "id": "Q1_broken",
        "label": "Mitochondria — Monocercomonoides (broken)",
        "stem": "What is the primary function of the mitochondria in a Monocercomonoides cell?",
        "choices": {
            "A": "Protein synthesis",
            "B": "Energy production",
            "C": "Cell division",
            "D": "DNA replication",
        },
        "correct": None,  # broken premise — Monocercomonoides has no mitochondria
        "paraphrases": [
            "What role do mitochondria primarily play inside a Monocercomonoides cell?",
            "Which function are mitochondria best known for in Monocercomonoides?",
            "In Monocercomonoides, what is the main job of the mitochondria?",
            "What is the chief responsibility of mitochondria within Monocercomonoides cells?",
            "What do mitochondria mainly do for a Monocercomonoides cell?",
        ],
    },
    {
        "id": "Q2_valid",
        "label": "Bowling strike (valid)",
        "stem": "In bowling, how many pins must you knock down to get a strike?",
        "choices": {
            "A": "zero",
            "B": "8",
            "C": "10",
            "D": "12",
        },
        "correct": "C",
        "paraphrases": [
            "How many pins need to fall for a strike in bowling?",
            "To score a strike in bowling, how many pins do you have to knock over?",
            "What number of pins must be knocked down for a bowling strike?",
            "In ten-pin bowling, how many pins must you topple to earn a strike?",
            "A bowling strike requires knocking down how many pins?",
        ],
    },
    {
        "id": "Q2_broken",
        "label": "Bowling strike — second throw (broken)",
        "stem": "In bowling, how many pins must you knock down on your second throw to get a strike?",
        "choices": {
            "A": "zero",
            "B": "8",
            "C": "10",
            "D": "12",
        },
        "correct": None,  # broken premise — a strike is first-throw only
        "paraphrases": [
            "How many pins need to fall on your second throw for a strike in bowling?",
            "To score a strike in bowling, how many pins must you knock over on the second ball?",
            "What number of pins must be knocked down on the second throw for a bowling strike?",
            "In ten-pin bowling, how many pins must you topple on your second throw to earn a strike?",
            "A bowling strike on the second throw requires knocking down how many pins?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Ollama API helper
# ---------------------------------------------------------------------------
def query_ollama(
    question_text: str,
    choices: dict[str, str],
    think: bool,
) -> str:
    """
    Send a multiple-choice question to the Ollama chat API with structured
    output (JSON schema constraining the answer to one of A/B/C/D).

    Returns the raw answer letter from the model (one of A, B, C, D).
    """
    # Build the prompt with the given (possibly shuffled) answer choices
    choice_lines = "\n".join(f"  {letter}) {text}" for letter, text in choices.items())
    user_message = (
        f"{question_text}\n\n"
        f"{choice_lines}\n\n"
        "Respond with the letter of the correct answer."
    )

    # JSON schema that constrains the model to return exactly one answer letter
    response_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": list(choices.keys()),
            }
        },
        "required": ["answer"],
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering multiple-choice questions. "
                    "Return ONLY the letter of the correct answer in the required JSON format."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "format": response_schema,
        "think": think,
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Extract the answer from the model's response content
    content = data["message"]["content"]
    try:
        parsed = json.loads(content)
        answer = parsed["answer"]
    except (json.JSONDecodeError, KeyError):
        # Fallback: try to find a single letter A-D in the raw text
        for ch in ["A", "B", "C", "D"]:
            if ch in content:
                answer = ch
                break
        else:
            answer = "?"  # Could not parse — will be visible in output
            print(f"    [WARNING] Could not parse response: {content!r}")

    return answer


# ---------------------------------------------------------------------------
# Answer-order shuffling
# ---------------------------------------------------------------------------
def make_shuffled_choices(
    canonical_choices: dict[str, str], rng: random.Random
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Shuffle the answer texts across the A/B/C/D slots.

    Returns:
        shuffled_choices:  {letter: text} in shuffled order (what the model sees)
        shuffled_to_canon: {shuffled_letter: canonical_letter} mapping back
    """
    letters = list(canonical_choices.keys())         # ['A', 'B', 'C', 'D']
    texts = list(canonical_choices.values())          # original texts in A-D order
    shuffled_indices = list(range(len(texts)))
    rng.shuffle(shuffled_indices)

    # Build new choice dict and reverse mapping
    shuffled_choices = {}
    shuffled_to_canon = {}
    for new_pos, old_idx in enumerate(shuffled_indices):
        new_letter = letters[new_pos]
        old_letter = letters[old_idx]
        shuffled_choices[new_letter] = texts[old_idx]
        shuffled_to_canon[new_letter] = old_letter

    return shuffled_choices, shuffled_to_canon


# ---------------------------------------------------------------------------
# Dirichlet posterior utilities
# ---------------------------------------------------------------------------
def compute_exceedance_probability(alpha: np.ndarray, rng: np.random.Generator) -> float:
    """
    Monte Carlo estimate of the exceedance probability for the leading answer.

    Uses the Gamma-sampling trick: draw independent Gamma(alpha_k, 1) samples,
    normalise to get Dirichlet draws, then count how often the leading component
    has the highest value.
    """
    # Gamma draws: shape (NUM_MC_SAMPLES, K)
    gamma_samples = np.column_stack(
        [rng.gamma(a, 1.0, size=NUM_MC_SAMPLES) for a in alpha]
    )
    # Normalise each row to get Dirichlet samples
    dirichlet_samples = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)

    # Leading answer = the one with the highest pseudo-count
    leading_idx = np.argmax(alpha)
    # Fraction of samples where the leading answer has the highest probability
    exceedance = np.mean(dirichlet_samples[:, leading_idx] == dirichlet_samples.max(axis=1))
    return float(exceedance)


def dirichlet_entropy(alpha: np.ndarray) -> float:
    """
    Compute the differential entropy of a Dirichlet distribution.

    H(Dir(alpha)) = ln B(alpha) - (K-1) psi(alpha_0)
                    + sum_k (alpha_k - 1) psi(alpha_k)
    where alpha_0 = sum(alpha), psi = digamma, and B is the multivariate Beta.
    """
    from scipy.special import digamma, gammaln

    alpha_0 = alpha.sum()
    K = len(alpha)
    # ln B(alpha) = sum ln Gamma(alpha_k) - ln Gamma(alpha_0)
    ln_B = gammaln(alpha).sum() - gammaln(alpha_0)
    entropy = (
        ln_B
        - (K - 1) * digamma(alpha_0)
        + ((alpha - 1) * digamma(alpha)).sum()
    )
    return float(entropy)


# ---------------------------------------------------------------------------
# Run one question through the full pipeline
# ---------------------------------------------------------------------------
def run_question(
    question: dict,
    think: bool,
    rng_shuffle: random.Random,
    rng_mc: np.random.Generator,
) -> dict:
    """
    Run original + 5 paraphrases for a single question variant.

    Returns a dict with raw responses, final alpha, exceedance, and entropy.
    """
    canonical_choices = question["choices"]
    letters = list(canonical_choices.keys())  # ['A', 'B', 'C', 'D']

    # All query texts: original stem first, then paraphrases
    query_texts = [question["stem"]] + question["paraphrases"]

    # Dirichlet prior: uniform [1, 1, 1, 1]
    alpha = np.ones(len(letters), dtype=float)

    raw_responses = []          # list of (query_text, canonical_answer)
    alpha_history = [alpha.copy()]
    exceedance_history = []

    for query_text in query_texts:
        # Shuffle answer order for this query
        shuffled_choices, shuffled_to_canon = make_shuffled_choices(
            canonical_choices, rng_shuffle
        )

        # Query the model
        shuffled_answer = query_ollama(query_text, shuffled_choices, think=think)

        # Map shuffled answer back to canonical letter
        canonical_answer = shuffled_to_canon.get(shuffled_answer, "?")
        raw_responses.append((query_text, canonical_answer))

        # Update Dirichlet posterior: increment pseudo-count for chosen answer
        if canonical_answer in letters:
            idx = letters.index(canonical_answer)
            alpha[idx] += 1.0

        alpha_history.append(alpha.copy())

        # Compute exceedance probability after this update
        exc = compute_exceedance_probability(alpha, rng_mc)
        exceedance_history.append(exc)

    # Final entropy
    entropy = dirichlet_entropy(alpha)

    return {
        "id": question["id"],
        "label": question["label"],
        "correct": question["correct"],
        "raw_responses": raw_responses,
        "final_alpha": alpha,
        "alpha_history": alpha_history,
        "exceedance_history": exceedance_history,
        "final_exceedance": exceedance_history[-1],
        "entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Pretty-print results for one question
# ---------------------------------------------------------------------------
def print_results(result: dict) -> None:
    """Print a summary block for one question variant."""
    letters = ["A", "B", "C", "D"]
    print(f"\n{'='*70}")
    print(f"  {result['label']}")
    if result["correct"]:
        print(f"  Ground truth: {result['correct']}")
    else:
        print(f"  Ground truth: N/A (broken premise)")
    print(f"{'='*70}")

    print("\n  Responses (query -> canonical answer):")
    for i, (query, ans) in enumerate(result["raw_responses"]):
        tag = "ORIG" if i == 0 else f"P{i}  "
        # Truncate long query text for display
        short_query = query if len(query) <= 65 else query[:62] + "..."
        print(f"    [{tag}] {short_query}")
        print(f"           -> Answer: {ans}   |  Exceedance: {result['exceedance_history'][i]:.3f}")

    print(f"\n  Final Dirichlet alpha: ", end="")
    for letter, a in zip(letters, result["final_alpha"]):
        print(f"{letter}={a:.0f}  ", end="")
    print()

    leading_idx = np.argmax(result["final_alpha"])
    print(f"  Leading answer:       {letters[leading_idx]}")
    print(f"  Exceedance prob:      {result['final_exceedance']:.4f}")
    print(f"  Posterior entropy:    {result['entropy']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  PILOT TEST — Bayesian UQ with Dirichlet Posterior")
    print(f"  Model: {MODEL}")
    print(f"  MC samples: {NUM_MC_SAMPLES}")
    print("=" * 70)

    # Check Ollama is reachable before starting
    try:
        health = requests.get("http://localhost:11434/api/tags", timeout=5)
        health.raise_for_status()
        print("  Ollama connection: OK")
    except requests.RequestException as e:
        print(f"  [ERROR] Cannot reach Ollama at {OLLAMA_URL}: {e}")
        print("  Make sure Ollama is running (ollama serve) and the model is pulled.")
        return

    # Run the full battery for each thinking mode
    for think_mode in [True, False]:
        mode_label = "THINKING ENABLED" if think_mode else "THINKING DISABLED"
        print(f"\n\n{'#'*70}")
        print(f"##  {mode_label}")
        print(f"{'#'*70}")

        # Fresh RNGs for each mode so results are comparable
        rng_shuffle = random.Random(SEED)
        rng_mc = np.random.default_rng(SEED)

        results = []
        for q in QUESTIONS:
            print(f"\n  Processing: {q['label']} ...")
            result = run_question(q, think=think_mode, rng_shuffle=rng_shuffle, rng_mc=rng_mc)
            results.append(result)
            print_results(result)

        # Summary comparison table
        print(f"\n\n{'-'*70}")
        print(f"  SUMMARY -- {mode_label}")
        print(f"{'-'*70}")
        print(f"  {'Question':<45} {'Lead':>4} {'Exc':>8} {'Entropy':>8}")
        print(f"  {'-'*45} {'-'*4} {'-'*8} {'-'*8}")
        for r in results:
            leading_idx = np.argmax(r["final_alpha"])
            lead_letter = ["A", "B", "C", "D"][leading_idx]
            print(
                f"  {r['label']:<45} {lead_letter:>4} "
                f"{r['final_exceedance']:>8.4f} {r['entropy']:>8.4f}"
            )

    print(f"\n\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
