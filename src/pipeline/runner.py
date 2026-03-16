import os
import time
from typing import List

import numpy as np

from src.config.loader import ModelConfig, load_model_config
from src.hf_utils.checkpoints import RevisionInfo, list_revisions
from src.hf_utils.download import download_checkpoint
from src.io_utils.results_writer import append_result, get_results_path, load_done_revisions
from src.pca.compute import run_pca
from src.tensors.embedding_loader import load_embedding_tensor


def _filter_new_revisions(revs: List[RevisionInfo], done: set[str]) -> List[RevisionInfo]:
    return [r for r in revs if r.name not in done]


def run_model_pipeline(root: str, model_id: str) -> None:
    """
    High-level pipeline for a single model:
    - load config
    - list & filter revisions
    - resume from existing results
    - per revision: download -> load embedding -> PCA -> write -> cleanup
    """
    cfg: ModelConfig = load_model_config(root, model_id)

    results_path = get_results_path(root, cfg.results.output_dir, cfg.results.pca_file)
    done = load_done_revisions(results_path)
    print(f"[{model_id}] already processed: {len(done)} revisions")

    all_revs = list_revisions(
        repo_id=cfg.repo_id,
        parse_cfg=cfg.family_config.branch_parse,
        filter_cfg=cfg.family_config.checkpoint_filter,
    )
    remaining_revs = _filter_new_revisions(all_revs, done)
    print(f"[{model_id}] remaining revisions: {len(remaining_revs)}")

    total = len(remaining_revs)
    if total == 0:
        print(f"[{model_id}] nothing to do.")
        return

    times: list[float] = []
    all_start = time.time()

    for idx, rev in enumerate(remaining_revs, start=1):
        start = time.time()
        print(f"\n[{model_id}] [{idx}/{total}] Processing revision: {rev.name}")

        local_path = download_checkpoint(cfg, rev.name)
        emb = load_embedding_tensor(cfg, local_path)
        print(f"[{model_id}] embedding shape: {emb.shape}")

        pca_stats = run_pca(emb, cfg.pca)

        record = {
            "model_id": cfg.model_id,
            "family": cfg.family,
            "repo_id": cfg.repo_id,
            "revision": rev.name,
            "stage": rev.stage,
            "step": rev.step,
            "shape": list(emb.shape),
            "explained_variance_ratio": pca_stats["explained_variance_ratio"].tolist(),
            "pca_config": {
                "n_components": cfg.pca.n_components,
                "whiten": cfg.pca.whiten,
            },
        }

        append_result(results_path, record)

        # Free memory and delete checkpoint directory
        del emb
        import shutil

        shutil.rmtree(local_path, ignore_errors=True)

        dt = time.time() - start
        times.append(dt)

        elapsed = time.time() - all_start
        avg = sum(times) / len(times)
        remaining = total - idx
        eta_left = avg * remaining

        print(f"[{model_id}] time for this revision: {dt:.1f}s")
        print(f"[{model_id}] elapsed: {elapsed / 60:.1f} min")
        print(f"[{model_id}] estimated left: {eta_left / 60:.1f} min")

