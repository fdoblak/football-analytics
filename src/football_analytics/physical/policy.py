"""Strict trajectory / physical-metrics policy loaders (Stage 9A)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.physical.types import PolicyError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TRAJECTORY_TOP = frozenset(
    {
        "schema_version",
        "policy_version",
        "config_version",
        "input_eligibility",
        "gap_types",
        "gap_policy",
        "segment_rules",
        "sample_layers",
        "filter_resample_placeholders",
        "coordinate_frame",
        "review",
        "safety",
        "notes",
    }
)

REQUIRED_METRICS_TOP = frozenset(
    {
        "schema_version",
        "policy_version",
        "config_version",
        "manual_target_confirmation_required",
        "predicted_interpolated_eligibility",
        "distance",
        "speed",
        "sprint",
        "heatmap",
        "activity_coverage",
        "zones",
        "coverage_thresholds",
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


def default_trajectory_policy_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "physical" / "trajectory_policy.yaml"


def default_metrics_policy_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "physical" / "physical_metrics_policy.yaml"


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


def load_trajectory_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_trajectory_policy_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TRAJECTORY_TOP - set(data)
    if missing:
        raise PolicyError(f"trajectory_policy missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise PolicyError("unsupported trajectory config_version")
    elig = data["input_eligibility"]
    if not isinstance(elig, dict):
        raise PolicyError("input_eligibility must be mapping")
    if elig.get("predicted_interpolated_eligible") is not False:
        raise PolicyError("predicted_interpolated_eligible must be false")
    if elig.get("require_confirmed_target") is not True:
        raise PolicyError("require_confirmed_target must be true")
    if data["gap_policy"].get("silent_fill_forbidden") is not True:
        raise PolicyError("silent_fill_forbidden must be true")
    if data["gap_policy"].get("distance_bridge_forbidden") is not True:
        raise PolicyError("distance_bridge_forbidden must be true")
    if str(data["coordinate_frame"].get("attack_direction_default")) != "unknown":
        raise PolicyError("attack_direction_default must be unknown")
    if data["safety"].get("no_real_metric_computation") is not True:
        raise PolicyError("no_real_metric_computation must be true")
    if data["filter_resample_placeholders"].get("enabled") is not False:
        raise PolicyError("filter_resample_placeholders.enabled must be false")
    return MappingProxyType(_deep_freeze(data))  # type: ignore[arg-type]


def load_metrics_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_metrics_policy_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_METRICS_TOP - set(data)
    if missing:
        raise PolicyError(f"physical_metrics_policy missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise PolicyError("unsupported metrics config_version")
    if data.get("manual_target_confirmation_required") is not True:
        raise PolicyError("manual_target_confirmation_required must be true")
    if data.get("predicted_interpolated_eligibility") is not False:
        raise PolicyError("predicted_interpolated_eligibility must be false")
    if data["activity_coverage"].get("composite_score_enabled") is not False:
        raise PolicyError("composite_score_enabled must be false")
    if data["activity_coverage"].get("low_coverage_is_not_low_activity") is not True:
        raise PolicyError("low_coverage_is_not_low_activity must be true")
    if data["zones"].get("progression_metrics_enabled") is not False:
        raise PolicyError("progression_metrics_enabled must be false")
    if data["zones"].get("attack_relative_names_forbidden") is not True:
        raise PolicyError("attack_relative_names_forbidden must be true")
    if data["safety"].get("no_real_metric_computation") is not True:
        raise PolicyError("no_real_metric_computation must be true")
    if data["speed"].get("forbid_frame_fps_time") is not True:
        raise PolicyError("forbid_frame_fps_time must be true")
    if data["sprint"].get("single_frame_spike_not_sprint") is not True:
        raise PolicyError("single_frame_spike_not_sprint must be true")
    if data["heatmap"].get("unseen_time_not_zero_activity") is not True:
        raise PolicyError("unseen_time_not_zero_activity must be true")
    return MappingProxyType(_deep_freeze(data))  # type: ignore[arg-type]


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(policy))


def assert_contract_only_policies(
    trajectory: Mapping[str, Any], metrics: Mapping[str, Any]
) -> None:
    _ = load_trajectory_policy  # noqa: F841 — loaded policies already validated
    if trajectory.get("safety", {}).get("no_real_metric_computation") is not True:
        raise PolicyError("trajectory safety.no_real_metric_computation must be true")
    if metrics.get("safety", {}).get("no_real_metric_computation") is not True:
        raise PolicyError("metrics safety.no_real_metric_computation must be true")


__all__ = [
    "CONFIG_VERSION",
    "default_trajectory_policy_path",
    "default_metrics_policy_path",
    "load_trajectory_policy",
    "load_metrics_policy",
    "policy_fingerprint",
    "assert_contract_only_policies",
]
