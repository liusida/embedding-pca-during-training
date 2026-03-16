import argparse
import json
import os
from typing import Dict, List, Tuple

from src.config.loader import load_model_config
from src.hf_utils.checkpoints import list_revisions


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


def _summarize_rows(rows: List[Dict]) -> Tuple[str, str]:
    if not rows:
        return "<empty>", "No records found."

    first = rows[0]
    model_id = first.get("model_id") or first.get("model") or "<unknown>"

    steps = [r.get("step") for r in rows if isinstance(r.get("step"), int)]
    steps = sorted(set(steps))

    if steps:
        detail = f"Unique steps: {len(steps)} (min={min(steps)}, max={max(steps)})"
    else:
        revisions = sorted({str(r.get("revision")) for r in rows})
        detail = f"No numeric steps; unique revisions: {len(revisions)}"

    return model_id, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple summary of PCA JSONL results.")
    parser.add_argument(
        "--path",
        help="Path to a JSONL results file, e.g. results/pythia-160m/pca.jsonl. "
        "If omitted, summarize all results/*/pca.jsonl files.",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.path:
        # Single file mode
        path = os.path.join(root, args.path)
        rows = _load_results(path)
        if not rows:
            print(f"No records found in {args.path}")
            return

        model_id, detail = _summarize_rows(rows)

        # Try to compute total expected revisions for this model
        try:
            cfg = load_model_config(root, model_id)
            revs = list_revisions(
                repo_id=cfg.repo_id,
                parse_cfg=cfg.family_config.branch_parse,
                filter_cfg=cfg.family_config.checkpoint_filter,
            )
            total = len(revs)
            progress = f"{len(rows)}/{total}" if total > 0 else str(len(rows))
        except Exception:
            progress = str(len(rows))

        print(f"{args.path}: {progress} records | Model: {model_id} | {detail}")
        return

    # Multi-file mode: scan results/*/pca.jsonl
    results_root = os.path.join(root, "results")
    if not os.path.isdir(results_root):
        print("No results directory found.")
        return

    summaries: List[Tuple[str, int, str, str]] = []
    for model_dir in sorted(os.listdir(results_root)):
        p = os.path.join(results_root, model_dir, "pca.jsonl")
        if not os.path.exists(p):
            continue
        rows = _load_results(p)
        if not rows:
            continue
        model_id, detail = _summarize_rows(rows)
        rel = os.path.relpath(p, root)

        # Compute total expected revisions
        try:
            cfg = load_model_config(root, model_id)
            revs = list_revisions(
                repo_id=cfg.repo_id,
                parse_cfg=cfg.family_config.branch_parse,
                filter_cfg=cfg.family_config.checkpoint_filter,
            )
            total = len(revs)
        except Exception:
            total = 0

        summaries.append((model_id, len(rows), total, rel, detail))

    if not summaries:
        print("No result files found under results/*/pca.jsonl.")
        return

    for model_id, n_rows, total, rel, detail in summaries:
        if total > 0:
            progress = f"{n_rows}/{total}"
        else:
            progress = str(n_rows)
        print(f"{rel}: {progress} records | Model: {model_id} | {detail}")


if __name__ == "__main__":
    main()

