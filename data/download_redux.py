from datasets import load_dataset
import csv
from pathlib import Path

# Download pre-merged MMLU-Redux 2.0 (all 57 subjects in one dataset)
ds = load_dataset("zwhe99/mmlu-redux-2.0", split="test")

output_path = Path(__file__).parent / "mmlu" / "mmlu_redux.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(list(ds.column_names))
    for row in ds:
        writer.writerow([row[col] for col in ds.column_names])

print(f"Saved {len(ds)} questions to {output_path}")
print(f"Columns: {ds.column_names}")
print(f"\nFirst row:\n{ds[0]}")