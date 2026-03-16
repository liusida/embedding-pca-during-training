import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from huggingface_hub import HfApi

from src.config.loader import BranchParseConfig, CheckpointFilterConfig


@dataclass
class RevisionInfo:
    name: str
    stage: int
    step: int


def _parse_revision_name(name: str, cfg: BranchParseConfig) -> Tuple[int, int]:
    """
    Parse revision/branch name into (stage, step).
    Falls back to (inf, inf) if no pattern matches so it sorts last.
    """
    for pat in cfg.regex_patterns:
        m = re.search(pat.pattern, name)
        if not m:
            continue
        stage = int(m.groupdict().get("stage", 0))
        step = int(m.groupdict().get("step", 0))
        return stage, step
    return (10**9, 10**9)


def list_revisions(
    repo_id: str,
    parse_cfg: BranchParseConfig,
    filter_cfg: CheckpointFilterConfig,
    include_main: bool = False,
) -> List[RevisionInfo]:
    """
    List and sort training checkpoints as RevisionInfo objects.
    """
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    branches = [
        b for b in refs.branches if include_main or b.name != "main"
    ]

    parsed: List[RevisionInfo] = []
    for b in branches:
        stage, step = _parse_revision_name(b.name, parse_cfg)
        parsed.append(RevisionInfo(name=b.name, stage=stage, step=step))

    parsed.sort(key=lambda r: (r.stage, r.step))

    # Apply step-based filters
    min_step = filter_cfg.min_step
    max_step: Optional[int] = filter_cfg.max_step
    stride = max(filter_cfg.step_stride, 1)

    filtered: List[RevisionInfo] = []
    for idx, r in enumerate(parsed):
        if r.step < min_step:
            continue
        if max_step is not None and r.step > max_step:
            continue
        if idx % stride != 0:
            continue
        filtered.append(r)

    return filtered

