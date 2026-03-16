import json
import os
from typing import Dict, Iterator, Set


def get_results_path(root: str, output_dir: str, pca_file: str) -> str:
    out_dir = os.path.join(root, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, pca_file)


def load_done_revisions(path: str) -> Set[str]:
    """
    Read an existing JSONL results file and return the set of revision names
    that have already been processed.
    """
    done: Set[str] = set()
    if not os.path.exists(path):
        return done

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rev = rec.get("revision")
                if isinstance(rev, str):
                    done.add(rev)
            except json.JSONDecodeError:
                continue
    return done


def append_result(path: str, record: Dict) -> None:
    """
    Append a single JSON record to the results file, fsynced.
    """
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())

