"""Strict tracking contract policy loader (Stage 6A)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.tracking.types import (
    LifecycleState,
    ObservationSource,
    TrackEntityType,
    TrackingContractError,
)

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "policy_version",
        "config_version",
        "id_namespace",
        "lifecycle",
        "allowed_transitions",
        "observation_sources",
        "entity_scopes",
        "gap_reason_codes",
        "review_rules",
        "bbox",
        "time",
        "safety_limits",
        "provenance_requirements",
        "notes",
    }
)


class PolicyError(TrackingContractError):
    """Tracking policy config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{label} must be a mapping")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{label} must be a bool")
    return value


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise PolicyError(f"{label} must be >= {minimum}")
    return value


def _require_str_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise PolicyError(f"{label} must be a list")
    if not value and not allow_empty:
        raise PolicyError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise PolicyError(f"{label} entries must be non-empty strings")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def default_tracking_policy_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "tracking" / "tracking_contract_policy.yaml"


def load_tracking_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_tracking_policy_path(project_root=project_root)
    if p.is_symlink():
        raise PolicyError(f"symlink rejected: {p}")
    raw_bytes = p.read_bytes()
    if len(raw_bytes) > MAX_CONFIG_BYTES:
        raise PolicyError("policy exceeds max bytes")
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise PolicyError("policy root must be a mapping")
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise PolicyError(f"policy missing keys: {sorted(missing)}")
    if int(data["config_version"]) != CONFIG_VERSION:
        raise PolicyError("unsupported config_version")

    life = _require_mapping(data["lifecycle"], label="lifecycle")
    _require_int(
        life["confirmation_observation_threshold"],
        label="confirmation threshold",
        minimum=1,
    )
    _require_int(life["max_lost_gap_us"], label="max_lost_gap_us", minimum=0)
    _require_bool(life["allow_birth_confirmed"], label="allow_birth_confirmed")
    _require_bool(life["reopen_terminated"], label="reopen_terminated")
    if life["reopen_terminated"] is not False:
        raise PolicyError("reopen_terminated must be false")
    _require_bool(life["merge_allowed"], label="merge_allowed")
    if life["merge_allowed"] is not False:
        raise PolicyError("merge_allowed must be false in Stage 6A")

    transitions = _require_mapping(data["allowed_transitions"], label="allowed_transitions")
    for src, dsts in transitions.items():
        allow_empty = str(src) == "terminated"
        _require_str_list(dsts, label=f"allowed_transitions[{src}]", allow_empty=allow_empty)
        if src != "null" and src not in {s.value for s in LifecycleState}:
            raise PolicyError(f"unknown transition source: {src}")
        for d in dsts:
            if d not in {s.value for s in LifecycleState}:
                raise PolicyError(f"unknown transition destination: {d}")
    if "terminated" in transitions and list(transitions["terminated"]):
        raise PolicyError("terminated must have empty transition list")

    obs = _require_mapping(data["observation_sources"], label="observation_sources")
    allowed = set(_require_str_list(obs["allowed"], label="observation_sources.allowed"))
    for s in ObservationSource:
        if s.value not in allowed:
            raise PolicyError(f"observation source missing: {s.value}")
    mapping = _require_mapping(obs["mapping_to_observation_state"], label="mapping")
    if mapping.get("detection_associated") != "observed":
        raise PolicyError("detection_associated must map to observed")
    phys = _require_mapping(obs["physical_metric_eligible"], label="physical_metric_eligible")
    if phys.get("predicted") is not False or phys.get("interpolated") is not False:
        raise PolicyError("predicted/interpolated must be physical_metric_eligible=false")

    scopes = _require_mapping(data["entity_scopes"], label="entity_scopes")
    allowed_ent = set(_require_str_list(scopes["allowed"], label="entity_scopes.allowed"))
    for e in TrackEntityType:
        if e.value not in allowed_ent:
            raise PolicyError(f"entity scope missing: {e.value}")
    if scopes.get("human_ball_merge_forbidden") is not True:
        raise PolicyError("human_ball_merge_forbidden must be true")

    time = _require_mapping(data["time"], label="time")
    if time.get("invent_from_fps_forbidden") is not True:
        raise PolicyError("invent_from_fps_forbidden must be true")
    if time.get("vfr_supported") is not True:
        raise PolicyError("vfr_supported must be true")

    ns = _require_mapping(data["id_namespace"], label="id_namespace")
    if ns.get("track_id_is_player_identity") is not False:
        raise PolicyError("track_id_is_player_identity must be false")
    if ns.get("reuse_forbidden") is not True:
        raise PolicyError("reuse_forbidden must be true")

    return MappingProxyType({k: _deep_freeze(v) for k, v in data.items()})


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(policy))


__all__ = [
    "CONFIG_VERSION",
    "PolicyError",
    "default_tracking_policy_path",
    "load_tracking_policy",
    "policy_fingerprint",
]
