"""
Build the master question database (data/questions.json) from MMLU Redux 2.0.

Reads the MMLU Redux CSV, keeps only verified-correct questions (error_type == "ok"),
converts each into a QuestionRecord, and writes the result as JSON.

Usage:
    python data/build_questions_json.py
"""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

# Add src/ so we can import the Pydantic models for validation
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from bayesian_uq.config import QuestionRecord


def main() -> None:
    csv_path = project_root / "data" / "mmlu" / "mmlu_redux.csv"
    output_path = project_root / "data" / "questions.json"

    # ------------------------------------------------------------------
    # Read the CSV
    # ------------------------------------------------------------------
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"  Total rows in CSV: {len(df)}")

    # ------------------------------------------------------------------
    # Filter to verified-correct questions only
    # ------------------------------------------------------------------
    df_ok = df[df["error_type"] == "ok"].copy()
    print(f"  Rows with error_type='ok': {len(df_ok)}")
    print(f"  Dropped: {len(df) - len(df_ok)} rows with errors")

    # ------------------------------------------------------------------
    # Convert each row into a QuestionRecord
    # ------------------------------------------------------------------
    records: list[dict] = []

    # Track per-subject index for zero-padded question IDs
    subject_counters: dict[str, int] = {}

    for _, row in df_ok.iterrows():
        subject = row["subject"]

        # Increment per-subject counter
        idx = subject_counters.get(subject, 0)
        subject_counters[subject] = idx + 1

        # Build the question_id: "mmlu_redux_{subject}_{0000}"
        question_id = f"mmlu_redux_{subject}_{idx:04d}"

        # Parse the choices column — it's a stringified Python list like
        # "['Yes, with p=2.', 'Yes, with p=3.', 'Yes, with p=5.', 'No.']"
        choices = ast.literal_eval(row["choices"])

        # Validate and build the record
        record = QuestionRecord(
            question_id=question_id,
            question_text=row["question"],
            choices=choices,
            correct_answer=int(row["answer"]),
            subject=subject,
            pair_id=None,
            variant="valid",
            break_type=None,
            break_description=None,
            source="mmlu_redux",
        )
        records.append(record.model_dump())

    # ------------------------------------------------------------------
    # Write to JSON
    # ------------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(records)} questions to {output_path}")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    print(f"\nSubjects ({len(subject_counters)}):")
    for subject, count in sorted(subject_counters.items(), key=lambda x: -x[1]):
        print(f"  {subject:40s} {count:>4d}")

    print(f"\n{'='*60}")
    print(f"  Total questions: {len(records)}")
    print(f"  Subjects:        {len(subject_counters)}")
    print(f"{'='*60}")

    # Show a sample question
    sample = records[0]
    print(f"\nSample question:")
    print(f"  ID:       {sample['question_id']}")
    print(f"  Subject:  {sample['subject']}")
    print(f"  Text:     {sample['question_text'][:80]}...")
    print(f"  Choices:  {sample['choices']}")
    print(f"  Answer:   {sample['correct_answer']}")


if __name__ == "__main__":
    main()
