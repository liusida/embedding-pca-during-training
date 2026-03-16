import argparse
import os
from typing import List

from src.config.loader import load_model_config
from src.pipeline.runner import run_model_pipeline


def _discover_models_for_family(root: str, family: str) -> List[str]:
    models_dir = os.path.join(root, "configs", "models")
    model_ids: List[str] = []
    for fname in os.listdir(models_dir):
        if not fname.endswith(".yaml"):
            continue
        model_id = fname[:-5]
        cfg = load_model_config(root, model_id)
        if cfg.family == family:
            model_ids.append(model_id)
    model_ids.sort()
    return model_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA pipeline for a model family.")
    parser.add_argument("--family", required=True, help="Family name, e.g. pythia or olmo")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models = _discover_models_for_family(root, args.family)
    if not models:
        print(f"No models configured for family '{args.family}'.")
        return

    print(f"Running family '{args.family}' for models: {models}")
    for m in models:
        run_model_pipeline(root=root, model_id=m)


if __name__ == "__main__":
    main()

