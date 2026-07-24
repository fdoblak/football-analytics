"""Stage 9E synthetic fusion fixtures (reuses 9B–9D helpers)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from football_analytics.physical.motion_config import (
    load_motion_baseline_config,
    motion_baseline_config_fingerprint,
)
from football_analytics.physical.motion_fixtures import constant_speed_points, single_sprint_points
from football_analytics.physical.motion_service import compute_physical_motion
from football_analytics.physical.spatial_config import (
    load_spatial_baseline_config,
    spatial_baseline_config_fingerprint,
)
from football_analytics.physical.spatial_service import compute_spatial_metrics
from football_analytics.physical.trajectory_config import (
    load_trajectory_baseline_config,
    trajectory_baseline_config_fingerprint,
)
from football_analytics.physical.trajectory_fixtures import (
    continuous_movement_bundle,
    revoked_identity_bundle,
)
from football_analytics.physical.trajectory_service import prepare_target_trajectory


def run_consistent_chain(output_root: Path) -> dict[str, Any]:
    """9B → 9C → 9D on the same synthetic confirmed target and point set."""
    output_root.mkdir(parents=True, exist_ok=True)
    traj_cfg = load_trajectory_baseline_config()
    traj_fp = trajectory_baseline_config_fingerprint(traj_cfg)
    candidates = continuous_movement_bundle(traj_fp)
    traj = prepare_target_trajectory(
        candidates=candidates, output_dir=output_root / "traj", config=traj_cfg
    )
    motion_cfg = load_motion_baseline_config()
    motion_fp = motion_baseline_config_fingerprint(motion_cfg)
    spatial_cfg = load_spatial_baseline_config()
    spatial_fp = spatial_baseline_config_fingerprint(spatial_cfg)

    # Shared eligible filtered-like points for both motion and spatial
    shared = constant_speed_points(motion_fp, speed_mps=5.0, n=11)
    for p in shared:
        p["run_id"] = candidates[0]["run_id"]
        p["video_id"] = candidates[0]["video_id"]
        p["target_player_id"] = candidates[0]["target_player_id"]
        p["identity_assignment_id"] = candidates[0]["identity_assignment_id"]
        p["identity_quality"] = "confirmed"
        p["policy_fingerprint"] = spatial_fp

    motion = compute_physical_motion(
        primary_points=shared, output_dir=output_root / "motion", config=motion_cfg
    )
    spatial = compute_spatial_metrics(
        primary_points=shared, output_dir=output_root / "spatial", config=spatial_cfg
    )
    return {
        "trajectory": traj,
        "motion": motion,
        "spatial": spatial,
        "identity": {
            "run_id": candidates[0]["run_id"],
            "video_id": candidates[0]["video_id"],
            "target_player_id": candidates[0]["target_player_id"],
            "identity_assignment_id": candidates[0]["identity_assignment_id"],
            "identity_status": "confirmed",
            "assignment_revoked": False,
        },
        "fingerprints": {
            "trajectory_config": traj_fp,
            "motion_config": motion_fp,
            "spatial_config": spatial_fp,
        },
        "motion_points": shared,
        "spatial_points": shared,
    }


def mismatched_target_block(base: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out["target_player_id"] = "other_target_99"
    return out


def revoked_identity_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    return revoked_identity_bundle(policy_fingerprint)


def sprint_chain_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    return single_sprint_points(policy_fingerprint)


__all__ = [
    "run_consistent_chain",
    "mismatched_target_block",
    "revoked_identity_points",
    "sprint_chain_points",
]
