"""Strict calibration policy / coordinate-system loaders (Stage 8A)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.calibration.types import PolicyError
from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_POLICY_TOP = frozenset(
    {
        "policy_version",
        "config_version",
        "homography",
        "coverage",
        "segments",
        "projection",
        "attack_direction",
        "safety",
        "leakage_separation",
        "reference_contracts",
        "notes",
    }
)

REQUIRED_COORD_TOP = frozenset(
    {
        "config_version",
        "coordinate_system_version",
        "image",
        "pitch",
        "attack_direction",
        "coordinate_frames",
        "notes",
    }
)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{label} must be a mapping")
    return value


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


def default_policy_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "calibration" / "calibration_contract_policy.yaml"


def default_coordinate_system_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "calibration" / "pitch_coordinate_system.yaml"


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


def load_calibration_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_policy_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_POLICY_TOP - set(data)
    if missing:
        raise PolicyError(f"policy missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise PolicyError("unsupported policy config_version")
    hom = _require_mapping(data["homography"], label="homography")
    if int(hom.get("min_correspondences", 0)) < 4:
        raise PolicyError("min_correspondences must be >= 4")
    if hom.get("reject_mirrored") is not True:
        raise PolicyError("reject_mirrored must be true")
    segs = _require_mapping(data["segments"], label="segments")
    if segs.get("silent_gap_fill") is not False:
        raise PolicyError("silent_gap_fill must be false")
    if segs.get("interpolated_homography_physical_metric_eligible") is not False:
        raise PolicyError("interpolated_homography_physical_metric_eligible must be false")
    proj = _require_mapping(data["projection"], label="projection")
    if proj.get("predicted_observation_physical_metric_eligible") is not False:
        raise PolicyError("predicted_observation_physical_metric_eligible must be false")
    attack = _require_mapping(data["attack_direction"], label="attack_direction")
    if str(attack.get("default")) != "unknown":
        raise PolicyError("attack_direction.default must be unknown")
    if attack.get("invent_team_side") is not False:
        raise PolicyError("invent_team_side must be false")
    safety = _require_mapping(data["safety"], label="safety")
    for key in (
        "no_sv_kp_inference",
        "no_sv_lines_inference",
        "no_real_keypoint_detection",
        "no_physical_metric_computation",
    ):
        if safety.get(key) is not True:
            raise PolicyError(f"safety.{key} must be true")
    return _deep_freeze(data)


def load_coordinate_system(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_coordinate_system_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_COORD_TOP - set(data)
    if missing:
        raise PolicyError(f"coordinate system missing keys: {sorted(missing)}")
    image = _require_mapping(data["image"], label="image")
    if str(image.get("origin")) != "top_left" or str(image.get("y_axis")) != "down":
        raise PolicyError("image must be top-left origin with y-down")
    pitch = _require_mapping(data["pitch"], label="pitch")
    for key in ("default_length_m", "default_width_m"):
        if float(pitch.get(key, 0)) <= 0:
            raise PolicyError(f"pitch.{key} must be positive")
    attack = _require_mapping(data["attack_direction"], label="attack_direction")
    if str(attack.get("default")) != "unknown":
        raise PolicyError("attack_direction.default must be unknown")
    if attack.get("invent_team_side") is not False:
        raise PolicyError("invent_team_side must be false")
    return _deep_freeze(data)


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(policy))


def coordinate_system_fingerprint(cfg: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(cfg))


__all__ = [
    "CONFIG_VERSION",
    "default_policy_path",
    "default_coordinate_system_path",
    "load_calibration_policy",
    "load_coordinate_system",
    "policy_fingerprint",
    "coordinate_system_fingerprint",
]
