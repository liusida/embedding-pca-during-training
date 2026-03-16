import os
from typing import Optional

import numpy as np
from safetensors import safe_open

from src.config.loader import EmbeddingConfig, ModelConfig


def _matches_key(key: str, emb_cfg: EmbeddingConfig) -> bool:
    for pattern in emb_cfg.key_patterns:
        if all(sub in key for sub in pattern.contains_all):
            return True
    return False


def load_embedding_tensor(model_cfg: ModelConfig, snapshot_path: str) -> np.ndarray:
    """
    Scan safetensor files under snapshot_path and return the first tensor
    whose name matches the embedding key patterns from the config.
    """
    emb_cfg = model_cfg.family_config.embedding

    for root, _dirs, files in os.walk(snapshot_path):
        for fname in files:
            if not fname.endswith(".safetensors"):
                continue

            full_path = os.path.join(root, fname)
            with safe_open(full_path, framework="numpy") as sf:
                for key in sf.keys():
                    if _matches_key(key, emb_cfg):
                        return sf.get_tensor(key)

    raise RuntimeError(f"Embedding tensor not found under {snapshot_path}")

