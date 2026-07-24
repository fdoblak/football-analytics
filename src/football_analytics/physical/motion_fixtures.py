"""Stage 9C synthetic motion fixtures (deterministic; no video/model)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.trajectory_fixtures import base_ids, candidate_point


def _tag_filtered(
    points: Sequence[Mapping[str, Any]], *, segment_id: str = "traj_seg_01"
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, p in enumerate(points):
        row = dict(p)
        row["sample_source"] = "filtered"
        row["derived_from_sample_ids"] = [str(p.get("sample_id", f"raw_{i}"))]
        row["trajectory_segment_id"] = segment_id
        row["metric_eligibility"] = "eligible"
        row["eligibility_status"] = "eligible"
        row["mapping_status"] = "mapped"
        out.append(row)
    return out


def constant_speed_points(
    policy_fingerprint: str,
    *,
    speed_mps: float = 5.0,
    n: int = 8,
    dt_us: int = 100_000,
    x0: float = 10.0,
    y0: float = 20.0,
    segment_id: str = "traj_seg_01",
) -> list[dict[str, Any]]:
    """Straight-line constant speed; analytical distance = speed * (n-1)*dt_s."""
    ids = base_ids()
    dx = speed_mps * (dt_us / 1_000_000.0)
    pts = [
        candidate_point(
            ids,
            sample_id=f"raw_{i:02d}",
            frame_index=i,
            video_time_us=i * dt_us,
            pitch_x_m=x0 + i * dx,
            pitch_y_m=y0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(n)
    ]
    return _tag_filtered(pts, segment_id=segment_id)


def known_distance_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """10 m along x over 2 s at 5 m/s (11 samples @ 0.2 s)."""
    return constant_speed_points(
        policy_fingerprint, speed_mps=5.0, n=11, dt_us=200_000, segment_id="traj_seg_known"
    )


def vfr_constant_speed_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    # Variable frame times; constant 4 m/s along x.
    times = [0, 40_000, 90_000, 160_000, 250_000, 330_000]
    speed = 4.0
    pts = []
    x = 10.0
    prev_t = 0
    for i, t in enumerate(times):
        if i > 0:
            x += speed * ((t - prev_t) / 1_000_000.0)
        pts.append(
            candidate_point(
                ids,
                sample_id=f"raw_vfr_{i}",
                frame_index=i,
                video_time_us=t,
                pitch_x_m=x,
                pitch_y_m=20.0,
                policy_fingerprint=policy_fingerprint,
            )
        )
        prev_t = t
    return _tag_filtered(pts, segment_id="traj_seg_vfr")


def zero_delta_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    pts = [
        candidate_point(
            ids,
            sample_id="raw_z0",
            frame_index=0,
            video_time_us=100_000,
            pitch_x_m=10.0,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        ),
        candidate_point(
            ids,
            sample_id="raw_z1",
            frame_index=1,
            video_time_us=100_000,  # zero delta
            pitch_x_m=10.5,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        ),
    ]
    return _tag_filtered(pts, segment_id="traj_seg_zd")


def single_point_segment(policy_fingerprint: str) -> list[dict[str, Any]]:
    return constant_speed_points(policy_fingerprint, n=1, segment_id="traj_seg_single")


def hard_gap_two_segments(policy_fingerprint: str) -> dict[str, list[dict[str, Any]]]:
    """Two segments separated by hard gap — distance must not bridge."""
    a = constant_speed_points(policy_fingerprint, speed_mps=5.0, n=4, segment_id="traj_seg_a")
    ids = {
        "run_id": a[0]["run_id"],
        "video_id": a[0]["video_id"],
        "target_player_id": a[0]["target_player_id"],
        "identity_assignment_id": a[0]["identity_assignment_id"],
    }
    b_raw = [
        candidate_point(
            ids,
            sample_id=f"raw_b{i}",
            frame_index=20 + i,
            video_time_us=3_000_000 + i * 100_000,
            pitch_x_m=50.0 + i * 0.5,
            pitch_y_m=22.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(4)
    ]
    b = _tag_filtered(b_raw, segment_id="traj_seg_b")
    return {"traj_seg_a": a, "traj_seg_b": b}


def outlier_spike_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Mostly 5 m/s with one impossible spike interval."""
    pts = constant_speed_points(policy_fingerprint, speed_mps=5.0, n=6)
    # Jump far in one step → >12 m/s
    pts[3]["pitch_x_m"] = float(pts[2]["pitch_x_m"]) + 5.0  # 5m / 0.1s = 50 m/s
    return pts


def low_coverage_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Short measured window vs large analysis window expectation."""
    return constant_speed_points(policy_fingerprint, speed_mps=5.0, n=3, dt_us=100_000)


def below_sprint_threshold_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """6.5 m/s — below 7.0 entry threshold."""
    return constant_speed_points(
        policy_fingerprint, speed_mps=6.5, n=20, dt_us=100_000, segment_id="traj_seg_nosprint"
    )


def hysteresis_sprint_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Accelerate above entry, dip between exit and entry (stay in), then exit."""
    ids = base_ids()
    # Build speed profile by cumulative distance over fixed dt.
    # Speeds (m/s): 5,5,8,8,8,8,8,8,8,8,6.5,6.5,6.5,5,5  (15 intervals → 16 pts)
    speeds = [5.0, 5.0] + [8.0] * 8 + [6.5, 6.5, 6.5] + [5.0, 5.0]
    dt_us = 100_000
    pts = []
    x = 10.0
    t = 0
    pts.append(
        candidate_point(
            ids,
            sample_id="raw_h00",
            frame_index=0,
            video_time_us=0,
            pitch_x_m=x,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
    )
    for i, sp in enumerate(speeds, start=1):
        t += dt_us
        x += sp * (dt_us / 1_000_000.0)
        pts.append(
            candidate_point(
                ids,
                sample_id=f"raw_h{i:02d}",
                frame_index=i,
                video_time_us=t,
                pitch_x_m=x,
                pitch_y_m=20.0,
                policy_fingerprint=policy_fingerprint,
            )
        )
    return _tag_filtered(pts, segment_id="traj_seg_hyst")


def short_burst_below_min_duration(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Brief 8 m/s burst shorter than min_duration_us (1s)."""
    ids = base_ids()
    speeds = [5.0, 8.0, 8.0, 8.0, 5.0]  # 0.3s at 8 m/s
    dt_us = 100_000
    pts = []
    x = 10.0
    t = 0
    pts.append(
        candidate_point(
            ids,
            sample_id="raw_s00",
            frame_index=0,
            video_time_us=0,
            pitch_x_m=x,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
    )
    for i, sp in enumerate(speeds, start=1):
        t += dt_us
        x += sp * (dt_us / 1_000_000.0)
        pts.append(
            candidate_point(
                ids,
                sample_id=f"raw_s{i:02d}",
                frame_index=i,
                video_time_us=t,
                pitch_x_m=x,
                pitch_y_m=20.0,
                policy_fingerprint=policy_fingerprint,
            )
        )
    return _tag_filtered(pts, segment_id="traj_seg_short")


def single_sprint_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """One clear sprint: 8 m/s for 1.5 s (≥1s, ≥5m)."""
    return constant_speed_points(
        policy_fingerprint,
        speed_mps=8.0,
        n=16,
        dt_us=100_000,
        segment_id="traj_seg_sprint1",
    )


def multi_sprint_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Two sprints separated by slow stretch (same segment, no hard gap)."""
    ids = base_ids()
    # sprint1: 8 m/s × 1.2s, slow 3 m/s × 0.5s, sprint2: 8 m/s × 1.2s
    speeds = [8.0] * 12 + [3.0] * 5 + [8.0] * 12
    dt_us = 100_000
    pts = []
    x = 10.0
    t = 0
    pts.append(
        candidate_point(
            ids,
            sample_id="raw_m00",
            frame_index=0,
            video_time_us=0,
            pitch_x_m=x,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
    )
    for i, sp in enumerate(speeds, start=1):
        t += dt_us
        x += sp * (dt_us / 1_000_000.0)
        pts.append(
            candidate_point(
                ids,
                sample_id=f"raw_m{i:02d}",
                frame_index=i,
                video_time_us=t,
                pitch_x_m=x,
                pitch_y_m=20.0,
                policy_fingerprint=policy_fingerprint,
            )
        )
    return _tag_filtered(pts, segment_id="traj_seg_multi")


def gap_split_sprints(policy_fingerprint: str) -> dict[str, list[dict[str, Any]]]:
    """Two sprint-capable segments separated by hard gap — must not merge."""
    a = constant_speed_points(policy_fingerprint, speed_mps=8.0, n=16, segment_id="traj_seg_sa")
    ids = {
        "run_id": a[0]["run_id"],
        "video_id": a[0]["video_id"],
        "target_player_id": a[0]["target_player_id"],
        "identity_assignment_id": a[0]["identity_assignment_id"],
    }
    b_raw = [
        candidate_point(
            ids,
            sample_id=f"raw_sb{i}",
            frame_index=100 + i,
            video_time_us=5_000_000 + i * 100_000,
            pitch_x_m=40.0 + i * 0.8,
            pitch_y_m=30.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(16)
    ]
    b = _tag_filtered(b_raw, segment_id="traj_seg_sb")
    return {"traj_seg_sa": a, "traj_seg_sb": b}


def uncertain_not_evaluable_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    pts = constant_speed_points(policy_fingerprint, speed_mps=8.0, n=16)
    for p in pts:
        p["uncertainty_m"] = 5.0  # above policy max 2.5
    return pts


def shot_boundary_split(policy_fingerprint: str) -> dict[str, list[dict[str, Any]]]:
    """Simulate pre/post shot as separate segments (no bridge)."""
    a = constant_speed_points(policy_fingerprint, n=4, segment_id="traj_seg_pre_shot")
    ids = {
        "run_id": a[0]["run_id"],
        "video_id": a[0]["video_id"],
        "target_player_id": a[0]["target_player_id"],
        "identity_assignment_id": a[0]["identity_assignment_id"],
    }
    b_raw = [
        candidate_point(
            ids,
            sample_id=f"raw_post{i}",
            frame_index=10 + i,
            video_time_us=500_000 + i * 100_000,
            pitch_x_m=12.0 + i * 0.5,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
            shot_cut=True,
        )
        for i in range(4)
    ]
    b = _tag_filtered(b_raw, segment_id="traj_seg_post_shot")
    return {"traj_seg_pre_shot": a, "traj_seg_post_shot": b}


__all__ = [
    "constant_speed_points",
    "known_distance_points",
    "vfr_constant_speed_points",
    "zero_delta_points",
    "single_point_segment",
    "hard_gap_two_segments",
    "outlier_spike_points",
    "low_coverage_points",
    "below_sprint_threshold_points",
    "hysteresis_sprint_points",
    "short_burst_below_min_duration",
    "single_sprint_points",
    "multi_sprint_points",
    "gap_split_sprints",
    "uncertain_not_evaluable_points",
    "shot_boundary_split",
]
