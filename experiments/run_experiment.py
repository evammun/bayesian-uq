"""
CLI entry point for running Bayesian UQ experiments.

Usage:
    python experiments/run_experiment.py --config experiments/configs/pilot_nothink.yaml

Loads the experiment config, question database, and paraphrase bank,
then runs the Bayesian UQ pipeline and saves results to results/.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path so we can import bayesian_uq
# (not needed if the package is installed with `uv pip install -e .`)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from bayesian_uq.pipeline import (
    filter_questions,
    load_config,
    load_paraphrases,
    load_questions,
    run_experiment,
    stratified_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Bayesian UQ experiment against a local Ollama model",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to experiment config YAML (e.g. experiments/configs/pilot_nothink.yaml)",
    )
    parser.add_argument(
        "--questions", type=Path, default=None,
        help="Path to questions.json (default: data/questions.json)",
    )
    parser.add_argument(
        "--paraphrases", type=Path, default=None,
        help="Path to paraphrases.json (default: data/paraphrases.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    questions_path = args.questions or project_root / "data" / "questions.json"
    paraphrases_path = args.paraphrases or project_root / "data" / "paraphrases.json"
    output_dir = args.output_dir or project_root / "results"

    # Load everything
    config = load_config(args.config)
    questions = load_questions(questions_path)
    paraphrases = load_paraphrases(paraphrases_path)

    # Filter to the questions this experiment cares about
    filtered = filter_questions(questions, config.question_set)

    # Apply max_questions cap if set — uses stratified sampling across subjects
    if config.max_questions is not None:
        total_filtered = len(filtered)
        filtered = stratified_sample(filtered, config.max_questions, seed=config.seed)
        # Count subjects in the sample for the log message
        subjects_in_sample = len({q.subject for q in filtered})
        print(f"Loaded {len(questions)} questions total, "
              f"{total_filtered} match question_set='{config.question_set}'")
        print(f"Stratified sample: {len(filtered)} questions across "
              f"{subjects_in_sample} subjects (seed={config.seed})")
    else:
        print(f"Loaded {len(questions)} questions total, "
              f"{len(filtered)} match question_set='{config.question_set}'")
    print()

    if not filtered:
        print("ERROR: No questions matched the filter. Check your question_set value.")
        sys.exit(1)

    # Run the experiment (use config seed, CLI --seed overrides if provided)
    seed = args.seed if args.seed != 42 else config.seed
    run_experiment(
        config=config,
        questions=filtered,
        paraphrases=paraphrases,
        output_dir=output_dir,
        seed=seed,
    )


if __name__ == "__main__":
    main()
