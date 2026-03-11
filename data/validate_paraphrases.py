"""
Validate the integrity of data/paraphrases.json against data/questions.json.

Runs 5 checks: coverage, count, original text match, quality sanity checks,
and answer choice leakage. Prints a summary report at the end.

Usage:
    python data/validate_paraphrases.py
"""

import json
import random
import re
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = DATA_DIR / "questions.json"
PARAPHRASES_PATH = DATA_DIR / "paraphrases.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load questions and paraphrases, returning dicts keyed by question_id."""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions_list = json.load(f)
    questions = {q["question_id"]: q for q in questions_list}

    with open(PARAPHRASES_PATH, encoding="utf-8") as f:
        paraphrases = json.load(f)

    return questions, paraphrases


# ---------------------------------------------------------------------------
# Check 1: Coverage
# ---------------------------------------------------------------------------

def check_coverage(questions, paraphrases):
    """Check which questions are missing from paraphrases.json."""
    missing = [qid for qid in questions if qid not in paraphrases]
    return missing


# ---------------------------------------------------------------------------
# Check 2: Paraphrase count
# ---------------------------------------------------------------------------

def check_counts(questions, paraphrases):
    """Check that each question has exactly 10 paraphrases."""
    wrong_count = []
    for qid in questions:
        if qid not in paraphrases:
            continue  # already caught by coverage check
        val = paraphrases[qid]
        if isinstance(val, dict):
            paras = val.get("paraphrases", [])
        elif isinstance(val, list):
            paras = val
        else:
            paras = []
        if len(paras) != 10:
            wrong_count.append((qid, len(paras)))
    return wrong_count


# ---------------------------------------------------------------------------
# Check 3: Original text match
# ---------------------------------------------------------------------------

def check_original_text(questions, paraphrases):
    """Verify the 'original' field matches question_text from questions.json."""
    mismatches = []
    for qid, q in questions.items():
        if qid not in paraphrases:
            continue
        val = paraphrases[qid]
        if isinstance(val, dict):
            original = val.get("original", "")
        else:
            continue  # old format has no 'original' field
        if original != q["question_text"]:
            mismatches.append((
                qid,
                q["question_text"][:80],
                original[:80],
            ))
    return mismatches


# ---------------------------------------------------------------------------
# Check 4: Quality sanity checks
# ---------------------------------------------------------------------------

def _get_paras(val):
    """Extract paraphrase strings from either format."""
    if isinstance(val, dict):
        return val.get("paraphrases", [])
    elif isinstance(val, list):
        # Old format: list of {text: ...} dicts
        return [p.get("text", "") if isinstance(p, dict) else str(p) for p in val]
    return []


def _get_original(val, questions, qid):
    """Get the original question text."""
    if isinstance(val, dict) and "original" in val:
        return val["original"]
    return questions.get(qid, {}).get("question_text", "")


# Regex to match answer choice markers like "A)", "A.", "(A)", etc.
ANSWER_CHOICE_RE = re.compile(r'\b[A-D]\)\s|\b[A-D]\.\s|\([A-D]\)\s')


def check_quality(questions, paraphrases):
    """Run quality sanity checks on paraphrase content."""
    identical_to_original = []
    empty_short = []
    contain_answer_letters = []
    duplicates = []

    for qid in questions:
        if qid not in paraphrases:
            continue
        val = paraphrases[qid]
        paras = _get_paras(val)
        original = _get_original(val, questions, qid)

        seen = set()
        for i, para in enumerate(paras):
            # Check: identical to original
            if para.strip() == original.strip():
                identical_to_original.append((qid, i))

            # Check: empty or too short
            if not para or len(para.strip()) < 10:
                empty_short.append((qid, i, len(para.strip())))

            # Check: contains answer choice markers
            if ANSWER_CHOICE_RE.search(para):
                contain_answer_letters.append((qid, i, para[:100]))

            # Check: duplicate within same question
            normalized = para.strip().lower()
            if normalized in seen:
                duplicates.append((qid, i))
            seen.add(normalized)

    return identical_to_original, empty_short, contain_answer_letters, duplicates


# ---------------------------------------------------------------------------
# Check 5: Answer choices leaked into paraphrases
# ---------------------------------------------------------------------------

def check_answer_leakage(questions, paraphrases, sample_size=20):
    """Check a sample of questions for answer choice text in paraphrases."""
    # Get question IDs that have paraphrases
    valid_qids = [qid for qid in questions if qid in paraphrases]
    rng = random.Random(42)
    sample = rng.sample(valid_qids, min(sample_size, len(valid_qids)))

    leaked = []
    for qid in sample:
        q = questions[qid]
        choices = q.get("choices", [])
        val = paraphrases[qid]
        paras = _get_paras(val)

        for i, para in enumerate(paras):
            for j, choice in enumerate(choices):
                # Only flag if the choice text is reasonably long (>15 chars)
                # to avoid false positives on short common phrases
                if len(choice) > 15 and choice in para:
                    leaked.append((qid, i, j, choice[:60]))

    return leaked, sample


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    questions, paraphrases = load_data()
    print(f"  questions.json:    {len(questions)} questions")
    print(f"  paraphrases.json:  {len(paraphrases)} entries")
    print()

    # --- Check 1: Coverage ---
    print("Check 1: Coverage...")
    missing = check_coverage(questions, paraphrases)
    if missing:
        print(f"  FAIL: {len(missing)} questions missing from paraphrases.json")
        for qid in missing[:10]:
            print(f"    - {qid}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    else:
        print("  PASS: All questions have paraphrase entries")
    print()

    # --- Check 2: Paraphrase count ---
    print("Check 2: Paraphrase count (should be 10 each)...")
    wrong_count = check_counts(questions, paraphrases)
    if wrong_count:
        print(f"  FAIL: {len(wrong_count)} questions have wrong count")
        for qid, n in wrong_count[:10]:
            print(f"    - {qid}: {n}/10")
        if len(wrong_count) > 10:
            print(f"    ... and {len(wrong_count) - 10} more")
    else:
        print("  PASS: All questions have exactly 10 paraphrases")
    print()

    # --- Check 3: Original text match ---
    print("Check 3: Original text matches question_text...")
    mismatches = check_original_text(questions, paraphrases)
    if mismatches:
        print(f"  FAIL: {len(mismatches)} mismatches found")
        for qid, qt, orig in mismatches[:5]:
            print(f"    - {qid}")
            print(f"      questions.json:    {qt}...")
            print(f"      paraphrases.json:  {orig}...")
    else:
        print("  PASS: All original fields match")
    print()

    # --- Check 4: Quality sanity checks ---
    print("Check 4: Quality sanity checks...")
    identical, empty_short, answer_letters, duplicates = check_quality(
        questions, paraphrases,
    )

    if identical:
        print(f"  WARN: {len(identical)} paraphrases identical to original")
        for qid, i in identical[:5]:
            print(f"    - {qid} paraphrase #{i}")
    else:
        print("  PASS: No paraphrases identical to original")

    if empty_short:
        print(f"  FAIL: {len(empty_short)} empty/short paraphrases (<10 chars)")
        for qid, i, length in empty_short[:5]:
            print(f"    - {qid} paraphrase #{i} ({length} chars)")
    else:
        print("  PASS: No empty/short paraphrases")

    if answer_letters:
        print(f"  WARN: {len(answer_letters)} paraphrases contain answer markers (A)/B)/etc.)")
        for qid, i, text in answer_letters[:5]:
            print(f"    - {qid} paraphrase #{i}: {text}...")
    else:
        print("  PASS: No paraphrases contain answer choice markers")

    if duplicates:
        print(f"  FAIL: {len(duplicates)} duplicate paraphrases within same question")
        for qid, i in duplicates[:5]:
            print(f"    - {qid} paraphrase #{i}")
    else:
        print("  PASS: No duplicate paraphrases")
    print()

    # --- Check 5: Answer choice leakage ---
    print("Check 5: Answer choices not leaked (sample of 20)...")
    leaked, sample = check_answer_leakage(questions, paraphrases)
    if leaked:
        print(f"  WARN: {len(leaked)} paraphrases contain answer choice text")
        for qid, i, j, choice in leaked[:5]:
            print(f"    - {qid} para #{i} contains choice {j}: \"{choice}\"")
    else:
        print(f"  PASS: No answer choice leakage in {len(sample)} sampled questions")
    print()

    # --- Summary ---
    total_q = len(questions)
    total_p = len(paraphrases)
    n_missing = len(missing)
    n_wrong = len(wrong_count)
    n_mismatch = len(mismatches)
    n_identical = len(identical)
    n_empty = len(empty_short)
    n_answer = len(answer_letters)
    n_dupes = len(duplicates)
    n_leaked = len(leaked)

    failures = n_missing + n_wrong + n_mismatch + n_empty + n_dupes
    warnings = n_identical + n_answer + n_leaked

    print("=" * 50)
    print("  VALIDATION REPORT")
    print("=" * 50)
    print(f"  Total questions in questions.json:  {total_q}")
    print(f"  Total in paraphrases.json:          {total_p}")
    print(f"  Missing:                            {n_missing}")
    print(f"  Wrong count (not 10):               {n_wrong}")
    print(f"  Original text mismatches:           {n_mismatch}")
    print(f"  Identical to original:              {n_identical}")
    print(f"  Empty/short paraphrases:            {n_empty}")
    print(f"  Contain answer markers:             {n_answer}")
    print(f"  Duplicate paraphrases:              {n_dupes}")
    print(f"  Answer choices leaked:              {n_leaked}")
    print()

    if failures > 0:
        print(f"  STATUS: FAIL ({failures} failures, {warnings} warnings)")
    elif warnings > 0:
        print(f"  STATUS: PASS with {warnings} warnings")
    else:
        print("  STATUS: PASS")
    print("=" * 50)


if __name__ == "__main__":
    main()
