"""Strict human-ball interaction policy loader (Stage 10A)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.interaction.types import PolicyError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "schema_version",
        "policy_version",
        "config_version",
        "stage",
        "contracts_only",
        "no_real_interaction_inference",
        "result_levels",
        "automatic_baseline",
        "eligibility",
        "lifecycle",
        "coverage",
        "forbidden_outputs",
        "review",
        "safety",
        "notes",
    }
)


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


def default_interaction_policy_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "interaction" / "human_ball_interaction_policy.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise PolicyError(f"symlink rejected: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise PolicyError(f"config too large: {path}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise PolicyError("config root must be mapping")
    return data


def load_interaction_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_interaction_policy_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise PolicyError(f"interaction policy missing keys: {sorted(missing)}")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise PolicyError("interaction policy schema_version mismatch")
    if data.get("contracts_only") is not True:
        raise PolicyError("interaction policy must be contracts_only")
    if data.get("no_real_interaction_inference") is not True:
        raise PolicyError("interaction policy must set no_real_interaction_inference")
    return _deep_freeze(data)


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(policy))


def assert_contract_only_policy(policy: Mapping[str, Any]) -> None:
    auto = policy.get("automatic_baseline") or {}
    if auto.get("nearest_player_is_not_possession") is not True:
        raise PolicyError("nearest_player_is_not_possession must be true")
    if auto.get("max_state") != "provisional":
        raise PolicyError("automatic baseline max_state must be provisional")
    forbidden = policy.get("forbidden_outputs") or {}
    for key in (
        "pass_count",
        "dribble_count",
        "duel_count",
        "box_touch_count",
        "real_possession_inference",
        "automatic_confirmed_possession",
        "opta_or_accuracy_claim",
    ):
        if forbidden.get(key) is not True:
            raise PolicyError(f"forbidden_outputs.{key} must be true")
    safety = policy.get("safety") or {}
    if safety.get("no_real_interaction_inference") is not True:
        raise PolicyError("safety.no_real_interaction_inference must be true")
    if safety.get("no_event_metric_computation") is not True:
        raise PolicyError("safety.no_event_metric_computation must be true")
    life = policy.get("lifecycle") or {}
    if life.get("hard_gap_no_possession_carry") is not True:
        raise PolicyError("hard_gap_no_possession_carry must be true")
    elig = policy.get("eligibility") or {}
    if elig.get("missing_ball_is_not_no_possession") is not True:
        raise PolicyError("missing_ball_is_not_no_possession must be true")


__all__ = [
    "CONFIG_VERSION",
    "default_interaction_policy_path",
    "load_interaction_policy",
    "policy_fingerprint",
    "assert_contract_only_policy",
]
