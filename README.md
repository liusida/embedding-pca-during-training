## Embedding PCA During Training

This repository runs PCA on embedding-layer weights for many intermediate training checkpoints of the Pythia and OLMo model families.

### Layout

- `configs/`: Declarative configuration for model families and individual models.
- `src/`: Reusable library code for downloading checkpoints, loading tensors, and computing PCA.
- `scripts/`: Thin CLIs that orchestrate experiments for one model or an entire family.
- `models_cache/`: Local cache of Hugging Face snapshots (safe to delete).
- `results/`: JSONL results of PCA runs, per model.

### Quick start

1. Install dependencies (inside your virtualenv):

```bash
pip install -e .
```

2. Run PCA for a single model, for example Pythia-160M:

```bash
python -m src.scripts.run_model --model pythia-160m
```

3. Run PCA for all configured Pythia models:

```bash
python -m src.scripts.run_family --family pythia
```

