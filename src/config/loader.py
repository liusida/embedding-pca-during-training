import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BranchRegexPattern:
    pattern: str


@dataclass
class BranchParseConfig:
    regex_patterns: List[BranchRegexPattern]


@dataclass
class CheckpointFilterConfig:
    min_step: int = 0
    max_step: Optional[int] = None
    step_stride: int = 1


@dataclass
class EmbeddingKeyPattern:
    contains_all: List[str]


@dataclass
class EmbeddingConfig:
    key_patterns: List[EmbeddingKeyPattern]


@dataclass
class FamilyConfig:
    family: str
    branch_parse: BranchParseConfig
    checkpoint_filter: CheckpointFilterConfig
    embedding: EmbeddingConfig


@dataclass
class ResultsConfig:
    output_dir: str
    pca_file: str


@dataclass
class DownloadConfig:
    cache_dir: str


@dataclass
class PcaConfig:
    n_components: Optional[int]
    whiten: bool
    random_state: Optional[int]


@dataclass
class ModelConfig:
    model_id: str
    family: str
    repo_id: str
    results: ResultsConfig
    download: DownloadConfig
    pca: PcaConfig
    family_config: FamilyConfig


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_family_config(root: str, family: str) -> FamilyConfig:
    path = os.path.join(root, "configs", "families", f"{family}.yaml")
    data = _load_yaml(path)

    branch_parse = BranchParseConfig(
        regex_patterns=[
            BranchRegexPattern(pattern=p["pattern"])
            for p in data.get("branch_parse", {}).get("regex_patterns", [])
        ]
    )

    cp_filter_raw = data.get("checkpoint_filter", {}) or {}
    checkpoint_filter = CheckpointFilterConfig(
        min_step=int(cp_filter_raw.get("min_step", 0)),
        max_step=(
            int(cp_filter_raw["max_step"]) if cp_filter_raw.get("max_step") is not None else None
        ),
        step_stride=int(cp_filter_raw.get("step_stride", 1)),
    )

    embedding = EmbeddingConfig(
        key_patterns=[
            EmbeddingKeyPattern(contains_all=list(p.get("contains_all", [])))
            for p in data.get("embedding", {}).get("key_patterns", [])
        ]
    )

    return FamilyConfig(
        family=str(data["family"]),
        branch_parse=branch_parse,
        checkpoint_filter=checkpoint_filter,
        embedding=embedding,
    )


def load_model_config(root: str, model_id: str) -> ModelConfig:
    """
    Load and merge model-level and family-level configuration.
    """
    path = os.path.join(root, "configs", "models", f"{model_id}.yaml")
    data = _load_yaml(path)

    family = str(data["family"])
    family_config = _load_family_config(root, family)

    results_raw = data.get("results", {})
    download_raw = data.get("download", {})
    pca_raw = data.get("pca", {})

    results = ResultsConfig(
        output_dir=str(results_raw["output_dir"]),
        pca_file=str(results_raw["pca_file"]),
    )

    download = DownloadConfig(cache_dir=str(download_raw["cache_dir"]))

    pca_cfg = PcaConfig(
        n_components=pca_raw.get("n_components"),
        whiten=bool(pca_raw.get("whiten", False)),
        random_state=pca_raw.get("random_state"),
    )

    return ModelConfig(
        model_id=str(data["model_id"]),
        family=family,
        repo_id=str(data["repo_id"]),
        results=results,
        download=download,
        pca=pca_cfg,
        family_config=family_config,
    )

