from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA

from src.config.loader import PcaConfig


def run_pca(emb: np.ndarray, cfg: PcaConfig) -> Dict[str, np.ndarray]:
    """
    Center the embedding matrix and run PCA according to cfg.

    Returns a dictionary containing explained_variance_ratio and
    possibly additional statistics in the future.
    """
    # Center rows
    X = emb - emb.mean(axis=0, keepdims=True)

    pca = PCA(
        n_components=cfg.n_components,
        whiten=cfg.whiten,
        random_state=cfg.random_state,
    )
    pca.fit(X)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }

