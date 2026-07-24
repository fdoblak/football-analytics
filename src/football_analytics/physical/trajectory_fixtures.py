"""Stage 9B synthetic trajectory fixtures (no video/model)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.physical.fixtures import sample_row


def base_ids() -> dict[str, str]:
    return {
        "run_id": generate_run_id(),
        "video_id": "video_synth_01",
        "target_player_id": "target_player_01",
        "identity_assignment_id": "asn_confirmed_01",
    }


def candidate_point(
    ids: Mapping[str, str],
    *,
    sample_id: str,
    frame_index: int,
    video_time_us: int,
    pitch_x_m: float,
    pitch_y_m: float,
    policy_fingerprint: str,
    track_id: int = 0,
    identity_quality: str = "confirmed",
    mapping_status: str = "mapped",
    metric_eligibility: str = "eligible",
    uncertainty_m: float = 0.4,
    segment_id: str = "cal_seg_01",
    shot_cut: bool = False,
    non_playable: bool = False,
    assignment_revoked: bool = False,
    calibration_invalid: bool = False,
    observation_source: str = "detection_associated",
) -> dict[str, Any]:
    row = sample_row(
        ids["run_id"],
        ids["video_id"],
        ids["target_player_id"],
        sample_id,
        identity_assignment_id=ids["identity_assignment_id"],
        track_id=track_id,
        frame_index=frame_index,
        video_time_us=video_time_us,
        pitch_x_m=pitch_x_m,
        pitch_y_m=pitch_y_m,
        policy_fingerprint=policy_fingerprint,
        identity_quality=identity_quality,
        mapping_status=mapping_status,
        metric_eligibility=metric_eligibility,
    )
    row["uncertainty_m"] = uncertainty_m
    row["segment_id"] = segment_id
    row["shot_cut"] = shot_cut
    row["non_playable"] = non_playable
    row["assignment_revoked"] = assignment_revoked
    row["calibration_invalid"] = calibration_invalid
    row["observation_source"] = observation_source
    row["physical_metric_eligibility"] = metric_eligibility
    return row


def continuous_movement_bundle(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    return [
        candidate_point(
            ids,
            sample_id=f"raw_{i:02d}",
            frame_index=i,
            video_time_us=i * 100_000,
            pitch_x_m=10.0 + i * 0.5,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(8)
    ]


def jump_spike_bundle(policy_fingerprint: str) -> list[dict[str, Any]]:
    pts = continuous_movement_bundle(policy_fingerprint)
    ids = {
        "run_id": pts[0]["run_id"],
        "video_id": pts[0]["video_id"],
        "target_player_id": pts[0]["target_player_id"],
        "identity_assignment_id": pts[0]["identity_assignment_id"],
    }
    # replace middle with jump
    pts[3] = candidate_point(
        ids,
        sample_id="raw_jump",
        frame_index=3,
        video_time_us=300_000,
        pitch_x_m=80.0,
        pitch_y_m=60.0,
        policy_fingerprint=policy_fingerprint,
    )
    return pts


def hard_gap_bundle(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    a = [
        candidate_point(
            ids,
            sample_id=f"raw_a{i}",
            frame_index=i,
            video_time_us=i * 100_000,
            pitch_x_m=10.0 + i,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(3)
    ]
    b = [
        candidate_point(
            ids,
            sample_id=f"raw_b{i}",
            frame_index=20 + i,
            video_time_us=2_500_000 + i * 100_000,
            pitch_x_m=30.0 + i,
            pitch_y_m=22.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(3)
    ]
    return a + b


def shot_boundary_bundle(policy_fingerprint: str) -> list[dict[str, Any]]:
    pts = continuous_movement_bundle(policy_fingerprint)[:4]
    ids = {
        "run_id": pts[0]["run_id"],
        "video_id": pts[0]["video_id"],
        "target_player_id": pts[0]["target_player_id"],
        "identity_assignment_id": pts[0]["identity_assignment_id"],
    }
    pts[2] = candidate_point(
        ids,
        sample_id="raw_shot",
        frame_index=2,
        video_time_us=200_000,
        pitch_x_m=11.0,
        pitch_y_m=20.0,
        policy_fingerprint=policy_fingerprint,
        shot_cut=True,
    )
    return pts


def revoked_identity_bundle(policy_fingerprint: str) -> list[dict[str, Any]]:
    pts = continuous_movement_bundle(policy_fingerprint)[:3]
    for p in pts:
        p["assignment_revoked"] = True
        p["identity_quality"] = "revoked"
        p["metric_eligibility"] = "not_eligible"
        p["physical_metric_eligibility"] = "not_eligible"
    return pts


def vfr_bundle(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    times = [0, 33_000, 80_000, 150_000, 240_000, 310_000]
    return [
        candidate_point(
            ids,
            sample_id=f"raw_vfr_{i}",
            frame_index=i,
            video_time_us=t,
            pitch_x_m=10.0 + i * 0.4,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i, t in enumerate(times)
    ]


__all__ = [
    "base_ids",
    "candidate_point",
    "continuous_movement_bundle",
    "jump_spike_bundle",
    "hard_gap_bundle",
    "shot_boundary_bundle",
    "revoked_identity_bundle",
    "vfr_bundle",
]
