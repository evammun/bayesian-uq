"""
Benchmark script comparing Ollama vs. llama-cpp-python.

Runs the same N questions through both backends and compares:
  - Wall time per query
  - Logprob values (should be very similar)
  - Predicted answers (should match)

Usage:
    python benchmark.py --model-path /path/to/qwen3.gguf --n 10 --prompt-mode direct

"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Import from prototype
from llama_client import (
    LlamaCppClient,
    extract_answer_logprobs as extract_logprobs_llamacpp,
    generate_permutation,
    ANSWER_LETTERS,
)

# Try to import from the project (will fail gracefully if not available)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from bayesian_uq.query import (
        OllamaClient,
        extract_answer_logprobs as extract_logprobs_ollama,
    )
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] Could not import OllamaClient. Ollama comparison disabled.")


def load_questions(num_questions: int = 10) -> list[dict]:
    """Load valid questions from the project database."""
    project_root = Path(__file__).parent.parent.parent
    questions_path = project_root / "data" / "questions.json"

    if not questions_path.exists():
        raise FileNotFoundError(f"Could not find {questions_path}")

    with open(questions_path) as f:
        all_questions = json.load(f)

    # Filter to valid questions
    valid_questions = [
        q for q in all_questions
        if (q.get("variant") in (None, "valid"))
        and q.get("correct_answer") is not None
    ]

    selected = valid_questions[:num_questions]
    print(f"[INFO] Loaded {len(selected)} questions")
    return selected


def run_benchmark(
    model_path: str,
    num_questions: int = 10,
    prompt_mode: str = "direct",
) -> None:
    """Run benchmark comparing Ollama and llama-cpp-python.

    Args:
        model_path: Path to the GGUF model file.
        num_questions: Number of questions to test.
        prompt_mode: "direct", "cot", or "cot_structured".
    """
    questions = load_questions(num_questions)

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: {prompt_mode.upper()} MODE")
    print(f"{'=' * 80}")
    print(f"Questions: {len(questions)}")
    print(f"Model path: {model_path}\n")

    # Initialize llama-cpp-python client
    print("[INFO] Initializing llama-cpp-python client...")
    try:
        llamacpp_client = LlamaCppClient(
            model_path=model_path,
            prompt_mode=prompt_mode,
            temperature=0.7,
            n_ctx=2048,
            n_gpu_layers=-1,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize llama-cpp-python client: {e}")
        return

    # Initialize Ollama client if available
    ollama_client = None
    if OLLAMA_AVAILABLE:
        print("[INFO] Initializing Ollama client...")
        try:
            ollama_client = OllamaClient(
                model="qwen3:8b-q4_K_M",
                prompt_mode=prompt_mode,
                temperature=0.7,
            )
            if not ollama_client.check_connection():
                print("[WARNING] Ollama not reachable. Skipping Ollama benchmark.")
                ollama_client = None
        except Exception as e:
            print(f"[WARNING] Failed to initialize Ollama client: {e}")
            ollama_client = None

    # Results storage
    results = {
        "llamacpp": {"times": [], "answers": [], "probs": []},
        "ollama": {"times": [], "answers": [], "probs": []},
    }

    # Run benchmark
    print(f"\n[INFO] Running {len(questions)} questions...\n")
    for q_idx, question in enumerate(questions):
        q_id = question["question_id"]
        q_text = question["question_text"]
        choices = question["choices"]
        correct_idx = question["correct_answer"]

        print(f"[Q{q_idx + 1:2d}] {q_id[:50]:<50} ", end="", flush=True)

        # Generate a fixed permutation for consistency
        answer_permutation = generate_permutation(shuffle=False)

        # llama-cpp-python query
        try:
            start = time.monotonic()
            response_text, all_logprobs, _, _ = llamacpp_client.send_query(
                question_text=q_text,
                choices=choices,
                answer_permutation=answer_permutation,
            )
            llamacpp_time = time.monotonic() - start

            (
                _,
                canon_lps,
                disp_ans,
                canon_ans,
                _,
            ) = extract_logprobs_llamacpp(
                all_logprobs, answer_permutation, prompt_mode=prompt_mode
            )

            results["llamacpp"]["times"].append(llamacpp_time)
            results["llamacpp"]["answers"].append(canon_ans)
            results["llamacpp"]["probs"].append(canon_lps)
            llamacpp_ans_str = f"{ANSWER_LETTERS[canon_ans]}"

        except Exception as e:
            llamacpp_time = None
            llamacpp_ans_str = "FAIL"
            print(f"llamacpp: {llamacpp_ans_str} ", end="", flush=True)

        # Ollama query (if available)
        if ollama_client:
            try:
                start = time.monotonic()
                response_text, all_logprobs, _, _ = ollama_client.send_query(
                    question_text=q_text,
                    choices=choices,
                    answer_permutation=answer_permutation,
                )
                ollama_time = time.monotonic() - start

                (
                    _,
                    canon_lps,
                    disp_ans,
                    canon_ans,
                    _,
                ) = extract_logprobs_ollama(
                    all_logprobs, answer_permutation, prompt_mode=prompt_mode
                )

                results["ollama"]["times"].append(ollama_time)
                results["ollama"]["answers"].append(canon_ans)
                results["ollama"]["probs"].append(canon_lps)
                ollama_ans_str = f"{ANSWER_LETTERS[canon_ans]}"

                # Calculate time ratio
                if llamacpp_time:
                    speedup = ollama_time / llamacpp_time
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"

                print(f"| Ollama: {ollama_ans_str} ({speedup_str})", flush=True)

            except Exception as e:
                print(f"| Ollama: FAIL", flush=True)
        else:
            print(f"| Ollama: SKIP", flush=True)

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"RESULTS")
    print(f"{'=' * 80}")

    # llama-cpp-python summary
    print(f"\nllama-cpp-python:")
    if results["llamacpp"]["times"]:
        avg_time = sum(results["llamacpp"]["times"]) / len(results["llamacpp"]["times"])
        total_time = sum(results["llamacpp"]["times"])
        print(f"  Queries: {len(results['llamacpp']['times'])}")
        print(f"  Avg time/query: {avg_time:.3f}s")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Min/Max: {min(results['llamacpp']['times']):.3f}s / {max(results['llamacpp']['times']):.3f}s")
    else:
        print(f"  No successful queries")

    # Ollama summary
    if ollama_client:
        print(f"\nOllama:")
        if results["ollama"]["times"]:
            avg_time = sum(results["ollama"]["times"]) / len(results["ollama"]["times"])
            total_time = sum(results["ollama"]["times"])
            print(f"  Queries: {len(results['ollama']['times'])}")
            print(f"  Avg time/query: {avg_time:.3f}s")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Min/Max: {min(results['ollama']['times']):.3f}s / {max(results['ollama']['times']):.3f}s")

            # Comparison
            if results["llamacpp"]["times"] and results["ollama"]["times"]:
                llamacpp_avg = sum(results["llamacpp"]["times"]) / len(results["llamacpp"]["times"])
                ollama_avg = sum(results["ollama"]["times"]) / len(results["ollama"]["times"])
                speedup = ollama_avg / llamacpp_avg
                improvement = (1 - llamacpp_avg / ollama_avg) * 100
                print(f"\nSpeedup (Ollama / llama-cpp-python): {speedup:.2f}x")
                print(f"Improvement: {improvement:.1f}% faster with llama-cpp-python")
        else:
            print(f"  No successful queries")

    # Answer agreement
    print(f"\nAnswer Agreement:")
    if results["llamacpp"]["answers"] and results["ollama"]["answers"]:
        min_len = min(len(results["llamacpp"]["answers"]), len(results["ollama"]["answers"]))
        agreements = sum(
            1 for i in range(min_len)
            if results["llamacpp"]["answers"][i] == results["ollama"]["answers"][i]
        )
        print(f"  Matching answers: {agreements}/{min_len} ({100*agreements/min_len:.1f}%)")
    elif results["llamacpp"]["answers"]:
        print(f"  Only llama-cpp-python produced answers")
    elif results["ollama"]["answers"]:
        print(f"  Only Ollama produced answers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark llama-cpp-python vs. Ollama"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of questions to benchmark (default: 10)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["direct", "cot", "cot_structured"],
        default="direct",
        help="Prompt mode (default: direct)",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            model_path=args.model_path,
            num_questions=args.n,
            prompt_mode=args.prompt_mode,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
