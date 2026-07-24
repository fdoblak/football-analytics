"""Load Stage 14 orchestration / review configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.orchestration.types import OrchestrationError


def default_pipeline_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "orchestration" / "single_player_pipeline.yaml"


def default_review_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "orchestration" / "review_hub.yaml"


def load_pipeline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_pipeline_config_path(project_root=project_root)
    if cfg_path.is_symlink():
        raise OrchestrationError(f"symlink rejected: {cfg_path}")
    if not cfg_path.is_file():
        raise OrchestrationError(f"pipeline config missing: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationError("pipeline config must be mapping")
    return data


def load_review_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_review_config_path(project_root=project_root)
    if cfg_path.is_symlink():
        raise OrchestrationError(f"symlink rejected: {cfg_path}")
    if not cfg_path.is_file():
        raise OrchestrationError(f"review config missing: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationError("review config must be mapping")
    return data


def pipeline_config_fingerprint(config: dict[str, Any]) -> str:
    return hash_canonical_json(config)


def review_config_fingerprint(config: dict[str, Any]) -> str:
    return hash_canonical_json(config)


__all__ = [
    "default_pipeline_config_path",
    "default_review_config_path",
    "load_pipeline_config",
    "load_review_config",
    "pipeline_config_fingerprint",
    "review_config_fingerprint",
]
