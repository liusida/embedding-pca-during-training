import argparse
import json
import os
from typing import Dict, List


def _load_results(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple summary of PCA JSONL results.")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a JSONL results file, e.g. results/pythia-160m/pca.jsonl",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, args.path)

    rows = _load_results(path)
    if not rows:
        print("No records found.")
        return

    print(f"Loaded {len(rows)} records from {args.path}")
    first = rows[0]
    model_id = first.get("model_id")
    steps = [r.get("step") for r in rows if isinstance(r.get("step"), int)]
    steps = sorted(set(steps))

    print(f"Model: {model_id}")
    print(f"Unique steps: {len(steps)} (min={min(steps)}, max={max(steps)})")


if __name__ == "__main__":
    main()

