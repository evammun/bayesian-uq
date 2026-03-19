"""
Pilot test for llama-cpp-python prototype.

Loads 4 questions from the project database and runs them through the
LlamaCppClient in one or both modes (direct and/or CoT).

Usage:
    python test_pilot.py --model-path /path/to/qwen3.gguf --prompt-mode direct
    python test_pilot.py --model-path /path/to/qwen3.gguf --prompt-mode cot
    python test_pilot.py --model-path /path/to/qwen3.gguf  # runs both modes

"""

import argparse
import json
import sys
import time
from pathlib import Path

# Import from prototype
from llama_client import (
    LlamaCppClient,
    extract_answer_logprobs,
    generate_permutation,
    ANSWER_LETTERS,
)


def load_questions(num_questions: int = 4) -> list[dict]:
    """Load valid questions from the project database.

    Filters to questions where variant is 'valid' (or None) and correct_answer is not None.
    Takes the first num_questions questions.

    Returns:
        List of question dicts with keys: question_id, question_text, choices,
        correct_answer, subject.
    """
    # Find the questions.json file relative to this script
    project_root = Path(__file__).parent.parent.parent
    questions_path = project_root / "data" / "questions.json"

    print(f"[INFO] Loading questions from {questions_path}")

    if not questions_path.exists():
        raise FileNotFoundError(f"Could not find {questions_path}")

    with open(questions_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter to valid questions
    valid_questions = [
        q for q in all_questions
        if (q.get("variant") in (None, "valid"))
        and q.get("correct_answer") is not None
    ]

    selected = valid_questions[:num_questions]
    print(f"[INFO] Loaded {len(selected)} valid questions")

    return selected


def format_choices(choices: list[str]) -> str:
    """Format choices as A) ... B) ... C) ... D) ..."""
    return "\n".join(
        f"  {letter}) {text}"
        for letter, text in zip(ANSWER_LETTERS, choices)
    )


def run_test(
    model_path: str,
    prompt_mode: str = "both",
    num_questions: int = 4,
) -> None:
    """Run the pilot test.

    Args:
        model_path: Path to the GGUF model file.
        prompt_mode: "direct", "cot", or "both".
        num_questions: Number of questions to test (default 4).
    """
    # Load questions
    questions = load_questions(num_questions)
    print(f"\n[INFO] Testing with {len(questions)} questions")

    modes = (
        ["direct", "cot"] if prompt_mode == "both"
        else [prompt_mode]
    )

    for mode in modes:
        print(f"\n{'=' * 70}")
        print(f"MODE: {mode}")
        print(f"{'=' * 70}")

        # Initialize client
        try:
            client = LlamaCppClient(
                model_path=model_path,
                prompt_mode=mode,
                temperature=0.7,
                n_ctx=2048,
                n_gpu_layers=-1,
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize client: {e}")
            return

        mode_times = []
        mode_correct = 0

        # Run each question
        for q_idx, question in enumerate(questions):
            q_id = question["question_id"]
            q_text = question["question_text"]
            choices = question["choices"]
            correct_idx = question["correct_answer"]
            subject = question.get("subject", "unknown")

            print(f"\n[Q{q_idx + 1}] {q_id}")
            print(f"Subject: {subject}")
            print(f"Question: {q_text[:100]}...")
            print(f"Choices:\n{format_choices(choices)}")
            print(f"Correct answer: {ANSWER_LETTERS[correct_idx]}")

            # Generate a permutation (no shuffling for simplicity in pilot)
            answer_permutation = generate_permutation(shuffle=False)

            # Query the model
            query_start = time.monotonic()
            try:
                response_text, all_logprobs, thinking, _ = client.send_query(
                    question_text=q_text,
                    choices=choices,
                    answer_permutation=answer_permutation,
                )
            except Exception as e:
                print(f"  [ERROR] Query failed: {e}")
                continue

            query_time = time.monotonic() - query_start
            mode_times.append(query_time)

            # Extract answer logprobs
            try:
                (
                    disp_letter_lps,
                    canon_lps,
                    disp_ans,
                    canon_ans,
                    _,
                ) = extract_answer_logprobs(
                    all_logprobs,
                    answer_permutation,
                    prompt_mode=mode,
                )

                # Convert logprobs to probabilities
                canonical_probs = {}
                logprob_sum = {}
                for idx, lp in canon_lps.items():
                    logprob_sum[idx] = lp

                # Simple softmax (from logprobs)
                max_lp = max(logprob_sum.values()) if logprob_sum else 0
                exp_sum = sum(
                    __import__("math").exp(lp - max_lp)
                    for lp in logprob_sum.values()
                )
                for idx, lp in logprob_sum.items():
                    canonical_probs[idx] = __import__("math").exp(lp - max_lp) / exp_sum

                # Format output
                print(f"  Time: {query_time:.2f}s")
                print(f"  Model answer: {disp_ans} (canonical: {canon_ans})")
                print(f"  Correct: {canon_ans == correct_idx}")

                # Show probabilities
                prob_str = " | ".join(
                    f"{ANSWER_LETTERS[i]}: {canonical_probs.get(i, 0):.3f}"
                    for i in range(len(choices))
                )
                print(f"  Probs: [{prob_str}]")

                # Response preview
                resp_preview = response_text[:150].replace("\n", " ")
                print(f"  Response: {resp_preview}...")

                # Track correctness
                if canon_ans == correct_idx:
                    mode_correct += 1

            except ValueError as e:
                print(f"  [ERROR] Failed to extract logprobs: {e}")
                print(f"  Response preview: {response_text[:150]}")

        # Summary for this mode
        print(f"\n{'-' * 70}")
        print(f"SUMMARY ({mode}):")
        print(f"  Questions tested: {len(questions)}")
        print(f"  Correct: {mode_correct}/{len(questions)} ({100*mode_correct/len(questions):.1f}%)")
        if mode_times:
            avg_time = sum(mode_times) / len(mode_times)
            total_time = sum(mode_times)
            print(f"  Avg time per query: {avg_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pilot test for llama-cpp-python prototype"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["direct", "cot", "cot_structured", "both"],
        default="both",
        help="Prompt mode to test (default: both)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=4,
        help="Number of questions to test (default: 4)",
    )

    args = parser.parse_args()

    try:
        run_test(
            model_path=args.model_path,
            prompt_mode=args.prompt_mode,
            num_questions=args.num_questions,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
