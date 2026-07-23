"""Strict loader for Stage 7C team assignment baseline config."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "config_version",
        "producer",
        "producer_version",
        "method_id",
        "method_type",
        "appearance_reid_config_path",
        "identity_policy_path",
        "eligibility",
        "clustering",
        "assignment",
        "review",
        "output_policy",
        "safety_limits",
        "runtime_root",
        "deterministic_seed",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "selection_matrix",
        "notes",
    }
)


class TeamAssignmentConfigError(ValueError):
    """Team assignment baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeamAssignmentConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TeamAssignmentConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise TeamAssignmentConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise TeamAssignmentConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeamAssignmentConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise TeamAssignmentConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise TeamAssignmentConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise TeamAssignmentConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TeamAssignmentConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeamAssignmentConfigError(f"{label} must be a non-empty string")
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
        raise TeamAssignmentConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise TeamAssignmentConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise TeamAssignmentConfigError(f"config_version must be {CONFIG_VERSION}")

    if _require_str(raw["method_type"], label="method_type") != "appearance_clustering":
        raise TeamAssignmentConfigError("method_type must be appearance_clustering")

    elig = dict(_require_mapping(raw["eligibility"], label="eligibility"))
    for key in (
        "human_only",
        "observed_only",
        "exclude_confirmed_goalkeeper_from_seed",
        "unknown_role_may_seed",
        "graphics_replay_forbidden",
    ):
        elig[key] = _require_bool(elig[key], label=f"eligibility.{key}")
    elig["min_coverage"] = _require_float(
        elig["min_coverage"], label="eligibility.min_coverage", minimum=0.0, maximum=1.0
    )
    elig["min_quality"] = _require_float(
        elig["min_quality"], label="eligibility.min_quality", minimum=0.0, maximum=1.0
    )
    elig["min_observed_samples"] = _require_int(
        elig["min_observed_samples"], label="eligibility.min_observed_samples", minimum=1
    )
    seed_roles = list(elig.get("seed_roles") or [])
    if not seed_roles or not all(isinstance(x, str) and x for x in seed_roles):
        raise TeamAssignmentConfigError("eligibility.seed_roles must be non-empty strings")
    elig["seed_roles"] = seed_roles
    excl = list(elig.get("exclude_roles_from_seed") or [])
    if not all(isinstance(x, str) and x for x in excl):
        raise TeamAssignmentConfigError("eligibility.exclude_roles_from_seed must be strings")
    elig["exclude_roles_from_seed"] = excl
    if elig["unknown_role_may_seed"]:
        raise TeamAssignmentConfigError("unknown_role_may_seed must be false in Stage 7C")

    cl = dict(_require_mapping(raw["clustering"], label="clustering"))
    cl["n_clusters"] = _require_int(cl["n_clusters"], label="clustering.n_clusters", minimum=2)
    if cl["n_clusters"] != 2:
        raise TeamAssignmentConfigError("Stage 7C baseline requires n_clusters=2")
    for key, mn in (
        ("min_seeds", 2),
        ("min_cluster_size", 1),
        ("max_lloyd_iters", 1),
        ("deterministic_seed", 0),
    ):
        cl[key] = _require_int(
            cl[key], label=f"clustering.{key}", minimum=mn if key != "deterministic_seed" else 0
        )
    for key in (
        "min_centroid_separation",
        "max_intra_cluster_spread",
        "outlier_distance",
        "assignment_max_distance",
        "ambiguity_margin",
        "similar_kit_separation_floor",
        "edge_texture_weight",
        "cross_shot_alignment_min_similarity",
    ):
        cl[key] = _require_float(cl[key], label=f"clustering.{key}", minimum=0.0)
    for key in ("color_dims_only", "cross_shot_alignment_enabled", "cross_video_auto_transfer"):
        cl[key] = _require_bool(cl[key], label=f"clustering.{key}")
    if cl["cross_video_auto_transfer"]:
        raise TeamAssignmentConfigError("cross_video_auto_transfer must be false")
    cl["init"] = _require_str(cl["init"], label="clustering.init")
    cl["label_ordering"] = _require_str(cl["label_ordering"], label="clustering.label_ordering")
    if cl["label_ordering"] != "centroid_fingerprint_asc":
        raise TeamAssignmentConfigError(
            "label_ordering must be centroid_fingerprint_asc for deterministic team_a/team_b"
        )
    cl["cluster_count_policy"] = _require_str(
        cl["cluster_count_policy"], label="clustering.cluster_count_policy"
    )

    asn = dict(_require_mapping(raw["assignment"], label="assignment"))
    for key in (
        "auto_real_team_naming",
        "auto_home_away",
        "auto_target_confirmation",
        "auto_goalkeeper_team_from_kit",
    ):
        asn[key] = _require_bool(asn[key], label=f"assignment.{key}")
        if asn[key]:
            raise TeamAssignmentConfigError(f"assignment.{key} must be false in Stage 7C")
    asn["player_team_role"] = _require_str(
        asn["player_team_role"], label="assignment.player_team_role"
    )
    if asn["player_team_role"] != "unknown":
        raise TeamAssignmentConfigError("player_team_role must be unknown (no home/away invent)")
    asn["referee_staff_team_role"] = _require_str(
        asn["referee_staff_team_role"], label="assignment.referee_staff_team_role"
    )
    if asn["referee_staff_team_role"] != "official":
        raise TeamAssignmentConfigError("referee_staff_team_role must be official")
    asn["referee_staff_team_id"] = _require_str(
        asn["referee_staff_team_id"], label="assignment.referee_staff_team_id"
    )
    if asn["referee_staff_team_id"] != "unknown":
        raise TeamAssignmentConfigError("referee_staff_team_id must be unknown")
    asn["goalkeeper_default_team_id"] = _require_str(
        asn["goalkeeper_default_team_id"], label="assignment.goalkeeper_default_team_id"
    )
    if asn["goalkeeper_default_team_id"] != "unknown":
        raise TeamAssignmentConfigError("goalkeeper_default_team_id must be unknown")
    asn["unknown_role_max_status"] = _require_str(
        asn["unknown_role_max_status"], label="assignment.unknown_role_max_status"
    )
    if asn["unknown_role_max_status"] != "candidate":
        raise TeamAssignmentConfigError("unknown_role_max_status must be candidate")
    asn["source"] = _require_str(asn["source"], label="assignment.source")
    if asn["source"] not in {"model", "rule", "manual"}:
        raise TeamAssignmentConfigError("assignment.source must be model|rule|manual")

    review = dict(_require_mapping(raw["review"], label="review"))
    for key in (
        "sample_ambiguous",
        "sample_similar_kit",
        "sample_outlier",
        "sample_role_conflict",
    ):
        review[key] = _require_bool(review[key], label=f"review.{key}")
    review["max_review_items"] = _require_int(
        review["max_review_items"], label="review.max_review_items", minimum=0
    )

    out_pol = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    out_pol["atomic_writes"] = _require_bool(out_pol["atomic_writes"], label="atomic_writes")
    out_pol["overwrite_allowed"] = _require_bool(
        out_pol["overwrite_allowed"], label="overwrite_allowed"
    )
    if out_pol["overwrite_allowed"]:
        raise TeamAssignmentConfigError("overwrite_allowed must be false")
    mode = out_pol["chmod_mode"]
    if isinstance(mode, str):
        mode = int(mode, 0)
    out_pol["chmod_mode"] = _require_int(mode, label="chmod_mode", minimum=0)
    out_pol["write_cluster_provenance"] = _require_bool(
        out_pol["write_cluster_provenance"], label="write_cluster_provenance"
    )

    safety = dict(_require_mapping(raw["safety_limits"], label="safety_limits"))
    safety["max_tracks_per_video"] = _require_int(
        safety["max_tracks_per_video"], label="max_tracks_per_video", minimum=1
    )
    safety["max_assignments_per_run"] = _require_int(
        safety["max_assignments_per_run"], label="max_assignments_per_run", minimum=1
    )
    safety["timeout_seconds"] = _require_float(
        safety["timeout_seconds"], label="timeout_seconds", minimum=1.0
    )

    matrix = list(raw["selection_matrix"])
    if not matrix:
        raise TeamAssignmentConfigError("selection_matrix must be non-empty")
    selected = [m for m in matrix if str(m.get("status", "")).lower() == "selected"]
    if len(selected) != 1:
        raise TeamAssignmentConfigError("exactly one selection_matrix entry must be selected")
    if (
        "anonymous" not in str(selected[0].get("candidate", "")).lower()
        and "2-cluster" not in str(selected[0].get("candidate", "")).lower()
        and "2cluster" not in str(selected[0].get("candidate", "")).lower()
    ):
        # Accept "anonymous 2-cluster..."
        cand = str(selected[0].get("candidate", "")).lower()
        if "cluster" not in cand or "anonymous" not in cand:
            raise TeamAssignmentConfigError(
                "selected method must be anonymous 2-cluster appearance"
            )

    if _require_bool(raw["overwrite_allowed"], label="overwrite_allowed"):
        raise TeamAssignmentConfigError("overwrite_allowed must be false")
    if _require_bool(raw["symlinks_allowed"], label="symlinks_allowed"):
        raise TeamAssignmentConfigError("symlinks_allowed must be false")
    if _require_bool(raw["network_sources_allowed"], label="network_sources_allowed"):
        raise TeamAssignmentConfigError("network_sources_allowed must be false")

    return {
        "config_version": CONFIG_VERSION,
        "producer": _require_str(raw["producer"], label="producer"),
        "producer_version": _require_str(raw["producer_version"], label="producer_version"),
        "method_id": _require_str(raw["method_id"], label="method_id"),
        "method_type": "appearance_clustering",
        "appearance_reid_config_path": _require_str(
            raw["appearance_reid_config_path"], label="appearance_reid_config_path"
        ),
        "identity_policy_path": _require_str(
            raw["identity_policy_path"], label="identity_policy_path"
        ),
        "eligibility": elig,
        "clustering": cl,
        "assignment": asn,
        "review": review,
        "output_policy": out_pol,
        "safety_limits": safety,
        "runtime_root": _require_str(raw["runtime_root"], label="runtime_root"),
        "deterministic_seed": _require_int(
            raw["deterministic_seed"], label="deterministic_seed", minimum=0
        ),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "selection_matrix": matrix,
        "notes": list(raw["notes"]),
    }


def load_team_assignment_config(path: Path | str) -> Mapping[str, Any]:
    p = Path(path)
    if p.is_symlink():
        raise TeamAssignmentConfigError(f"symlink rejected: {p}")
    if not p.is_file():
        raise TeamAssignmentConfigError(f"config not found: {p}")
    if p.stat().st_size > MAX_CONFIG_BYTES:
        raise TeamAssignmentConfigError("config too large")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TeamAssignmentConfigError("config root must be mapping")
    validated = _validate_config(raw)
    return _deep_freeze(validated)


def team_assignment_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_team_assignment_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "identity" / "team_assignment_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "TeamAssignmentConfigError",
    "load_team_assignment_config",
    "team_assignment_config_fingerprint",
    "default_team_assignment_config_path",
]
