"""Stage 9D synthetic spatial fixtures (deterministic; no video/model)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.trajectory_fixtures import base_ids, candidate_point


def _tag(
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


def stationary_zone_dwell(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Stay in goal_a_third / centre corridor for 2 s."""
    ids = base_ids()
    pts = [
        candidate_point(
            ids,
            sample_id=f"raw_{i:02d}",
            frame_index=i,
            video_time_us=i * 100_000,
            pitch_x_m=10.0 + i * 0.01,
            pitch_y_m=34.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(21)
    ]
    return _tag(pts, segment_id="traj_seg_dwell")


def pitch_crossing(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Walk length-wise at 5 m/s across thirds."""
    ids = base_ids()
    # 0 → 90 m in 18 s @ 5 m/s, dt=0.2s → 91 points
    n = 91
    dt = 200_000
    dx = 5.0 * 0.2
    pts = [
        candidate_point(
            ids,
            sample_id=f"raw_c{i:03d}",
            frame_index=i,
            video_time_us=i * dt,
            pitch_x_m=min(90.0, i * dx),
            pitch_y_m=34.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(n)
    ]
    return _tag(pts, segment_id="traj_seg_cross")


def vfr_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    times = [0, 40_000, 110_000, 200_000, 330_000, 500_000]
    pts = []
    x = 40.0
    prev = 0
    for i, t in enumerate(times):
        if i > 0:
            x += 3.0 * ((t - prev) / 1_000_000.0)
        pts.append(
            candidate_point(
                ids,
                sample_id=f"raw_v{i}",
                frame_index=i,
                video_time_us=t,
                pitch_x_m=x,
                pitch_y_m=20.0,
                policy_fingerprint=policy_fingerprint,
            )
        )
        prev = t
    return _tag(pts, segment_id="traj_seg_vfr")


def single_point(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    return _tag(
        [
            candidate_point(
                ids,
                sample_id="raw_one",
                frame_index=0,
                video_time_us=0,
                pitch_x_m=50.0,
                pitch_y_m=34.0,
                policy_fingerprint=policy_fingerprint,
            )
        ],
        segment_id="traj_seg_one",
    )


def hard_gap_segments(policy_fingerprint: str) -> list[dict[str, Any]]:
    a = stationary_zone_dwell(policy_fingerprint)[:6]
    for p in a:
        p["trajectory_segment_id"] = "traj_seg_a"
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
            frame_index=50 + i,
            video_time_us=5_000_000 + i * 100_000,
            pitch_x_m=80.0 + i * 0.01,
            pitch_y_m=34.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(6)
    ]
    b = _tag(b_raw, segment_id="traj_seg_b")
    return a + b


def zone_boundary_edge(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Points straddling third boundary at x = 35.0 (105/3)."""
    ids = base_ids()
    xs = [34.5, 34.9, 35.0, 35.1, 35.5]
    pts = [
        candidate_point(
            ids,
            sample_id=f"raw_e{i}",
            frame_index=i,
            video_time_us=i * 100_000,
            pitch_x_m=x,
            pitch_y_m=10.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i, x in enumerate(xs)
    ]
    return _tag(pts, segment_id="traj_seg_edge")


def out_of_pitch_point(policy_fingerprint: str) -> list[dict[str, Any]]:
    pts = stationary_zone_dwell(policy_fingerprint)[:5]
    pts[2]["pitch_x_m"] = 120.0  # outside
    return pts


def low_coverage_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    return stationary_zone_dwell(policy_fingerprint)[:3]


def speed_class_ladder(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Intervals at known class speeds: 0.2, 1.0, 3.0, 5.0, 8.0 m/s."""
    ids = base_ids()
    speeds = [0.2, 1.0, 3.0, 5.0, 8.0]
    # 5 intervals of 0.5s each → 6 points
    dt = 500_000
    pts = []
    x = 20.0
    t = 0
    pts.append(
        candidate_point(
            ids,
            sample_id="raw_l00",
            frame_index=0,
            video_time_us=0,
            pitch_x_m=x,
            pitch_y_m=34.0,
            policy_fingerprint=policy_fingerprint,
        )
    )
    for i, sp in enumerate(speeds, start=1):
        t += dt
        x += sp * (dt / 1_000_000.0)
        pts.append(
            candidate_point(
                ids,
                sample_id=f"raw_l{i:02d}",
                frame_index=i,
                video_time_us=t,
                pitch_x_m=x,
                pitch_y_m=34.0,
                policy_fingerprint=policy_fingerprint,
            )
        )
    return _tag(pts, segment_id="traj_seg_ladder")


def penalty_presence_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    """Physical presence inside goal_a_penalty — not a touch event."""
    ids = base_ids()
    pts = [
        candidate_point(
            ids,
            sample_id=f"raw_p{i}",
            frame_index=i,
            video_time_us=i * 100_000,
            pitch_x_m=5.0 + i * 0.1,
            pitch_y_m=34.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(11)
    ]
    return _tag(pts, segment_id="traj_seg_pen")


def uncertain_points(policy_fingerprint: str) -> list[dict[str, Any]]:
    pts = stationary_zone_dwell(policy_fingerprint)[:8]
    for p in pts:
        p["uncertainty_m"] = 5.0
    return pts


__all__ = [
    "stationary_zone_dwell",
    "pitch_crossing",
    "vfr_points",
    "single_point",
    "hard_gap_segments",
    "zone_boundary_edge",
    "out_of_pitch_point",
    "low_coverage_points",
    "speed_class_ladder",
    "penalty_presence_points",
    "uncertain_points",
]
