import os
import shutil

from huggingface_hub import snapshot_download

from src.config.loader import ModelConfig


def download_checkpoint(model_cfg: ModelConfig, revision: str) -> str:
    """
    Download a specific revision to the model's cache_dir.

    To avoid issues with partially downloaded checkpoints from previous
    interrupted runs, we clear the target local_dir before downloading.
    Resuming is handled via the JSONL results, not the raw cache.
    """
    cache_dir = model_cfg.download.cache_dir
    local_dir = os.path.join(cache_dir, revision)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, ignore_errors=True)

    local_path = snapshot_download(
        repo_id=model_cfg.repo_id,
        revision=revision,
        local_dir=local_dir,
    )
    return local_path

