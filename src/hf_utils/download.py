import os
from typing import Optional

from huggingface_hub import snapshot_download

from src.config.loader import ModelConfig


def download_checkpoint(model_cfg: ModelConfig, revision: str) -> str:
    """
    Download a specific revision to the model's cache_dir.

    Returns the local path where the snapshot was stored.
    """
    cache_dir = model_cfg.download.cache_dir
    local_dir = os.path.join(cache_dir, revision)
    os.makedirs(cache_dir, exist_ok=True)

    local_path = snapshot_download(
        repo_id=model_cfg.repo_id,
        revision=revision,
        local_dir=local_dir,
    )
    return local_path

