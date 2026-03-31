"""
CLI entry point for running QuALITY experiments.

Usage:
    python experiments/run_experiment.py --config experiments/configs/quality_pilot_direct_nothink_sufficient.yaml
    python experiments/run_experiment.py --config experiments/configs/quality_pilot_direct_nothink_sufficient.yaml --resume results/partial.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so src/ is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from pre_action_uq.config import ExperimentResult, QuestionResult
from pre_action_uq.pipeline import load_config, run_experiment


def load_resume_data(resume_path: Path) -> tuple[set[str], list[QuestionResult]]:
    """Load completed question IDs and results from a partial run.

    Handles truncated JSON from interrupted writes: reads as much valid
    data as possible rather than crashing.
    """
    try:
        with open(resume_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # File was truncated mid-write. Try to salvage by finding the
        # last complete question_result entry.
        print(f"WARNING: {resume_path.name} has invalid JSON (truncated?). Attempting partial recovery...")
        with open(resume_path, encoding="utf-8") as f:
            raw = f.read()
        # Find all complete question_id entries as a count estimate
        import re
        question_ids = re.findall(r'"question_id":\s*"([^"]+)"', raw)
        # Try progressively shorter slices until JSON parses
        data = None
        for trim in [1000, 5000, 20000, 100000, 500000]:
            try:
                # Try to close the JSON structure
                truncated = raw[:len(raw) - trim]
                # Find last complete question result (ends with })
                last_brace = truncated.rfind("}")
                if last_brace > 0:
                    attempt = truncated[:last_brace + 1] + "\n  ]\n}"
                    data = json.loads(attempt)
                    break
            except (json.JSONDecodeError, ValueError):
                continue
        if data is None:
            print(f"  Could not recover any data. Starting fresh.")
            return set(), []
        print(f"  Recovered {len(data.get('question_results', []))} of ~{len(question_ids)} questions")

    completed_ids: set[str] = set()
    carried_over: list[QuestionResult] = []

    for qr_dict in data.get("question_results", []):
        qid = qr_dict.get("question_id", "")
        completed_ids.add(qid)
        carried_over.append(QuestionResult.model_validate(qr_dict))

    print(f"Loaded {len(completed_ids)} completed questions from {resume_path.name}")
    return completed_ids, carried_over


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a QuALITY experiment from a YAML config."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to partial result JSON to resume from",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)

    # Resume support
    completed_ids = None
    carried_over = None
    if args.resume:
        resume_path = args.resume
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        completed_ids, carried_over = load_resume_data(resume_path)

    # Output directory
    output_dir = project_root / "results"

    # Run experiment
    run_experiment(
        config=config,
        output_dir=output_dir,
        completed_ids=completed_ids,
        carried_over_results=carried_over,
        resume_file=resume_path if args.resume else None,
    )


if __name__ == "__main__":
    main()
