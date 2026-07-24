"""Stage 9B trajectory segment splitting and gap records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.types import CONTRACT_VERSION


def _boundary_reason(
    prev: Mapping[str, Any] | None,
    cur: Mapping[str, Any],
    *,
    long_gap_us: int,
) -> str | None:
    if prev is None:
        return None
    if str(prev.get("run_id")) != str(cur.get("run_id")) or str(prev.get("video_id")) != str(
        cur.get("video_id")
    ):
        return "unknown"
    if str(prev.get("identity_assignment_id")) != str(cur.get("identity_assignment_id")):
        return "identity_gap"
    if cur.get("assignment_revoked") is True:
        return "identity_gap"
    if int(prev.get("track_id", -1)) != int(cur.get("track_id", -2)):
        return "track_boundary"
    if cur.get("shot_cut") is True or str(cur.get("boundary_hint", "")) == "shot_boundary":
        return "shot_boundary"
    if cur.get("non_playable") is True:
        return "non_playable_gap"
    if str(prev.get("segment_id")) != str(cur.get("segment_id")):
        return "calibration_gap"
    if cur.get("calibration_invalid") is True:
        return "calibration_gap"
    if str(cur.get("physical_metric_eligibility", "eligible")) != "eligible":
        return "unknown"
    dt = int(cur["video_time_us"]) - int(prev["video_time_us"])
    if dt < 0:
        return "unknown"
    if dt >= long_gap_us:
        return "tracking_gap"
    return None


def split_trajectory_segments(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split filtered points into non-overlapping half-open segments + gap rows.

    Returns (segments, gaps, points_with_segment_ids).
    """
    long_gap_us = int(config["segment_split"]["long_gap_us"])
    ordered = sorted(
        (dict(p) for p in points), key=lambda p: (int(p["video_time_us"]), str(p["sample_id"]))
    )
    segments: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    if not ordered:
        return segments, gaps, []

    current: list[dict[str, Any]] = []
    start_reason = "none"
    seg_idx = 0
    tagged: list[dict[str, Any]] = []

    def flush(end_reason: str) -> None:
        nonlocal seg_idx, current, start_reason
        if not current:
            return
        first, last = current[0], current[-1]
        n = len(current)
        status = "continuous" if n >= 2 else "insufficient"
        metric_elig = "eligible" if n >= 2 else "not_eligible"
        sid = f"traj_seg_{seg_idx:03d}"
        for p in current:
            p["trajectory_segment_id"] = sid
            tagged.append(p)
        end_us = int(last["video_time_us"]) + 1  # half-open: include last sample instant
        segments.append(
            {
                "run_id": first["run_id"],
                "video_id": first["video_id"],
                "target_player_id": first["target_player_id"],
                "trajectory_segment_id": sid,
                "identity_assignment_id": first["identity_assignment_id"],
                "track_id": int(first["track_id"]),
                "start_time_us": int(first["video_time_us"]),
                "end_time_us": end_us,
                "raw_sample_count": n,
                "eligible_sample_count": n,
                "duration_us": max(0, end_us - int(first["video_time_us"])),
                "calibration_segment_ids": sorted(
                    {str(p.get("segment_id")) for p in current if p.get("segment_id")}
                ),
                "start_boundary_reason": start_reason,
                "end_boundary_reason": end_reason,
                "coverage_ratio": 1.0 if n >= 2 else 0.0,
                "max_sample_interval_us": max(
                    (
                        int(current[i]["video_time_us"]) - int(current[i - 1]["video_time_us"])
                        for i in range(1, n)
                    ),
                    default=0,
                ),
                "uncertainty_summary_m": max(
                    (
                        float(p["uncertainty_m"])
                        for p in current
                        if p.get("uncertainty_m") is not None
                    ),
                    default=None,
                ),
                "segment_status": status,
                "metric_eligibility": metric_elig,
                "manual_review_required": False,
                "pitch_coordinate_frame_id": str(
                    first.get("pitch_coordinate_frame_id", "canonical_pitch")
                ),
                "input_fingerprint": None,
                "output_fingerprint": None,
                "policy_fingerprint": policy_fingerprint,
                "reason_codes": (["SINGLE_POINT_SEGMENT"] if n < 2 else []),
                "quality_flags": [],
                "provenance_json": None,
                "contract_version": CONTRACT_VERSION,
            }
        )
        seg_idx += 1
        current = []
        start_reason = end_reason if end_reason != "none" else "none"

    prev: dict[str, Any] | None = None
    for r in ordered:
        reason = _boundary_reason(prev, r, long_gap_us=long_gap_us)
        if reason is not None and current:
            flush(reason)
            if prev is not None:
                gaps.append(
                    {
                        "run_id": r["run_id"],
                        "video_id": r["video_id"],
                        "target_player_id": r["target_player_id"],
                        "gap_id": f"gap_{len(gaps):03d}",
                        "gap_type": reason if reason != "unknown" else "unknown",
                        "start_time_us": int(prev["video_time_us"]),
                        "end_time_us": int(r["video_time_us"]),
                        "duration_us": max(0, int(r["video_time_us"]) - int(prev["video_time_us"])),
                        "preceding_segment_id": None,
                        "following_segment_id": None,
                        "track_id": int(r.get("track_id") or 0),
                        "identity_assignment_id": str(r.get("identity_assignment_id")),
                        "allows_distance_bridge": False,
                        "allows_interpolation_default": False,
                        "manual_review_required": False,
                        "reason_codes": [reason],
                        "quality_flags": [],
                        "policy_fingerprint": policy_fingerprint,
                        "provenance_json": None,
                        "contract_version": CONTRACT_VERSION,
                    }
                )
            start_reason = reason
        current.append(r)
        prev = r
    flush("none")

    for g in gaps:
        t0, t1 = int(g["start_time_us"]), int(g["end_time_us"])
        for s in segments:
            if int(s["end_time_us"]) - 1 == t0 or (
                int(s["start_time_us"]) <= t0 < int(s["end_time_us"])
            ):
                g["preceding_segment_id"] = s["trajectory_segment_id"]
            if int(s["start_time_us"]) == t1:
                g["following_segment_id"] = s["trajectory_segment_id"]

    return segments, gaps, tagged


__all__ = ["split_trajectory_segments"]
