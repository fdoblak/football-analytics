"""Stage 9B deterministic time-based resampling within continuous segments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.types import CONTRACT_VERSION


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def resample_segment(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy_fingerprint: str,
    segment_id: str,
) -> list[dict[str, Any]]:
    """Resample one continuous eligible segment. No extrapolation beyond endpoints."""
    rs = config["resample"]
    if len(points) < 2:
        return []
    grid_us = int(rs["grid_us"])
    max_gap = int(rs["max_interpolation_gap_us"])
    ordered = sorted(points, key=lambda p: int(p["video_time_us"]))
    t0 = int(ordered[0]["video_time_us"])
    t1 = int(ordered[-1]["video_time_us"])
    by_time = {int(p["video_time_us"]): dict(p) for p in ordered}
    out: list[dict[str, Any]] = []
    # Inclusive start, inclusive end on grid snaps within [t0, t1]
    t = t0
    idx = 0
    while t <= t1:
        if t in by_time and rs.get("preserve_exact_observed_timestamps", True):
            src = by_time[t]
            row = dict(src)
            row["sample_source"] = "resampled"
            row["sample_id"] = f"rsp_{segment_id}_{t}"
            row["derived_from_sample_ids"] = [str(src["sample_id"])]
            row["trajectory_segment_id"] = segment_id
            row["metric_eligibility"] = "not_eligible"  # derived default
            row["eligibility_status"] = "eligible"
            row["quality_flags"] = list(src.get("quality_flags") or []) + [
                "resampled_exact_observed"
            ]
            row["policy_fingerprint"] = policy_fingerprint
            row["provenance_json"] = json.dumps(
                {
                    "method": rs["method"],
                    "interpolation_ratio": 0.0,
                    "neighbors": [str(src["sample_id"])],
                    "derived": True,
                },
                sort_keys=True,
            )
            out.append(row)
        else:
            # find neighbors
            while idx + 1 < len(ordered) and int(ordered[idx + 1]["video_time_us"]) < t:
                idx += 1
            left = ordered[idx]
            right = ordered[min(idx + 1, len(ordered) - 1)]
            tl, tr = int(left["video_time_us"]), int(right["video_time_us"])
            if tr <= tl or t < tl or t > tr:
                t += grid_us
                continue
            if (tr - tl) > max_gap:
                t += grid_us
                continue
            ratio = (t - tl) / float(tr - tl)
            x = _lerp(float(left["pitch_x_m"]), float(right["pitch_x_m"]), ratio)
            y = _lerp(float(left["pitch_y_m"]), float(right["pitch_y_m"]), ratio)
            ul = left.get("uncertainty_m")
            ur = right.get("uncertainty_m")
            unc_vals = [float(v) for v in (ul, ur) if v is not None and math.isfinite(float(v))]
            unc = max(unc_vals) if unc_vals else None
            row = {
                "run_id": left["run_id"],
                "video_id": left["video_id"],
                "target_player_id": left["target_player_id"],
                "identity_assignment_id": left["identity_assignment_id"],
                "sample_id": f"rsp_{segment_id}_{t}",
                "track_id": int(left["track_id"]),
                "observation_id": None,
                "detection_id": None,
                "projection_id": None,
                "frame_index": int(left["frame_index"]),
                "video_time_us": t,
                "pitch_x_m": x,
                "pitch_y_m": y,
                "pitch_coordinate_frame_id": left.get(
                    "pitch_coordinate_frame_id", "canonical_pitch"
                ),
                "pitch_template_fingerprint": left.get("pitch_template_fingerprint", "a" * 64),
                "calibration_id": left.get("calibration_id"),
                "segment_id": left.get("segment_id"),
                "source_point_type": left.get("source_point_type", "bbox_bottom_centre"),
                "sample_source": "resampled",
                "derived_from_sample_ids": [str(left["sample_id"]), str(right["sample_id"])],
                "mapping_status": "mapped",
                "calibration_quality": left.get("calibration_quality", "good"),
                "identity_quality": left.get("identity_quality", "confirmed"),
                "uncertainty_m": unc,
                "uncertainty_x_m": unc,
                "uncertainty_y_m": unc,
                "eligibility_status": "eligible",
                "gap_boundary_reason": "none",
                "metric_eligibility": "not_eligible",
                "trajectory_segment_id": segment_id,
                "manual_review_required": False,
                "reason_codes": ["DERIVED_RESAMPLED"],
                "quality_flags": ["resampled_derived"],
                "evidence_fingerprint": left.get("evidence_fingerprint"),
                "projected_positions_fingerprint": left.get("projected_positions_fingerprint"),
                "identity_artifact_fingerprint": left.get("identity_artifact_fingerprint"),
                "calibration_artifact_fingerprint": left.get("calibration_artifact_fingerprint"),
                "policy_fingerprint": policy_fingerprint,
                "provenance_json": json.dumps(
                    {
                        "method": rs["method"],
                        "interpolation_ratio": ratio,
                        "neighbors": [str(left["sample_id"]), str(right["sample_id"])],
                        "derived": True,
                        "no_extrapolation": True,
                    },
                    sort_keys=True,
                ),
                "contract_version": CONTRACT_VERSION,
            }
            out.append(row)
        t += grid_us
    return out


def resample_all_segments(
    filtered_points: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy_fingerprint: str,
) -> list[dict[str, Any]]:
    by_seg: dict[str, list[dict[str, Any]]] = {}
    for p in filtered_points:
        sid = str(p.get("trajectory_segment_id") or "")
        if not sid:
            continue
        by_seg.setdefault(sid, []).append(dict(p))
    out: list[dict[str, Any]] = []
    for seg in segments:
        if str(seg.get("segment_status")) != "continuous":
            continue
        if int(seg.get("eligible_sample_count", 0)) < 2:
            continue
        sid = str(seg["trajectory_segment_id"])
        pts = by_seg.get(sid, [])
        out.extend(
            resample_segment(
                pts, config=config, policy_fingerprint=policy_fingerprint, segment_id=sid
            )
        )
    return out


__all__ = ["resample_segment", "resample_all_segments"]
