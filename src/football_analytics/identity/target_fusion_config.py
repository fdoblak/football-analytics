"""Strict loader for Stage 7E target identity fusion config."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "config_version",
        "producer",
        "producer_version",
        "method_id",
        "method_type",
        "identity_policy_path",
        "safety",
        "evidence_reliability",
        "candidate_rules",
        "conflict_rules",
        "review",
        "metric_eligibility",
        "audit",
        "output_policy",
        "safety_limits",
        "runtime_root",
        "deterministic_seed",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "notes",
    }
)


class TargetFusionConfigError(ValueError):
    """Target fusion config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetFusionConfigError(f"{label} must be a mapping")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TargetFusionConfigError(f"{label} must be a bool")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetFusionConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise TargetFusionConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise TargetFusionConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetFusionConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise TargetFusionConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise TargetFusionConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise TargetFusionConfigError(f"{label} must be <= {maximum}")
    return f


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TargetFusionConfigError(f"{label} must be a non-empty string")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise TargetFusionConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise TargetFusionConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise TargetFusionConfigError(f"config_version must be {CONFIG_VERSION}")

    safety = _require_mapping(raw["safety"], label="safety")
    for key, expected in (
        ("face_recognition", False),
        ("biometric_identity", False),
        ("cross_video_auto_link", False),
        ("physical_track_merge", False),
        ("auto_confirm", False),
        ("manual_only_confirmation", True),
        ("appearance_alone_cannot_confirm", True),
        ("jersey_alone_cannot_confirm", True),
        ("team_alone_cannot_confirm", True),
        ("two_auto_cues_max_provisional", True),
        ("evaluation_label_must_not_enter_decisions", True),
        ("synthetic_fixtures_not_accuracy_claims", True),
    ):
        if key not in safety:
            raise TargetFusionConfigError(f"safety missing {key}")
        if _require_bool(safety[key], label=f"safety.{key}") is not expected:
            raise TargetFusionConfigError(f"safety.{key} must be {expected}")

    cand = _require_mapping(raw["candidate_rules"], label="candidate_rules")
    if (
        _require_int(
            cand["min_independent_auto_supporting_for_provisional"],
            label="candidate_rules.min_independent_auto_supporting_for_provisional",
            minimum=2,
        )
        < 2
    ):
        raise TargetFusionConfigError("min provisional cues must be >= 2")
    if cand.get("confirmed_requires_scoped_manual_decision") is not True:
        raise TargetFusionConfigError("confirmed_requires_scoped_manual_decision must be true")

    conflict = _require_mapping(raw["conflict_rules"], label="conflict_rules")
    if conflict.get("silent_majority_resolution") is not False:
        raise TargetFusionConfigError("silent_majority_resolution must be false")

    review = _require_mapping(raw["review"], label="review")
    allowed = review.get("allowed_decisions")
    if not isinstance(allowed, list) or not allowed:
        raise TargetFusionConfigError("review.allowed_decisions required")
    for d in ("confirm", "reject", "keep_provisional", "revoke", "unknown"):
        if d not in allowed:
            raise TargetFusionConfigError(f"review.allowed_decisions missing {d}")
    if review.get("append_only_audit") is not True:
        raise TargetFusionConfigError("append_only_audit must be true")

    out_pol = _require_mapping(raw["output_policy"], label="output_policy")
    if out_pol.get("atomic_writes") is not True:
        raise TargetFusionConfigError("atomic_writes must be true")
    if out_pol.get("overwrite_allowed") is not False:
        raise TargetFusionConfigError("overwrite_allowed must be false")
    if _require_bool(raw["overwrite_allowed"], label="overwrite_allowed") is not False:
        raise TargetFusionConfigError("overwrite_allowed must be false")
    if _require_bool(raw["network_sources_allowed"], label="network_sources_allowed") is not False:
        raise TargetFusionConfigError("network_sources_allowed must be false")

    _require_int(raw["deterministic_seed"], label="deterministic_seed", minimum=0)
    _require_str(raw["runtime_root"], label="runtime_root")
    _require_str(raw["producer"], label="producer")
    _require_str(raw["producer_version"], label="producer_version")
    _require_str(raw["method_id"], label="method_id")
    _require_float(
        _require_mapping(raw["safety_limits"], label="safety_limits").get("timeout_seconds", 120.0),
        label="safety_limits.timeout_seconds",
        minimum=1.0,
    )
    return dict(raw)


def default_target_fusion_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "identity" / "target_identity_fusion.yaml"


def load_target_fusion_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_target_fusion_config_path(project_root=project_root)
    if p.is_symlink():
        raise TargetFusionConfigError(f"symlink rejected: {p}")
    raw_bytes = p.read_bytes()
    if len(raw_bytes) > MAX_CONFIG_BYTES:
        raise TargetFusionConfigError("config exceeds max bytes")
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise TargetFusionConfigError("config root must be a mapping")
    validated = _validate_config(data)
    return _deep_freeze(validated)


def target_fusion_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "TargetFusionConfigError",
    "default_target_fusion_config_path",
    "load_target_fusion_config",
    "target_fusion_config_fingerprint",
]
