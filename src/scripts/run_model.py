import argparse
import os

from src.pipeline.runner import run_model_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA pipeline for a single model.")
    parser.add_argument("--model", required=True, help="Model id, e.g. pythia-160m")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_model_pipeline(root=root, model_id=args.model)


if __name__ == "__main__":
    main()

