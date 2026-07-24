"""Stage 10B proximity computation (image/pitch distances; nearest ≠ possession)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.interaction.eligibility import pitch_distance_usable
from football_analytics.interaction.types import CONTRACT_VERSION


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def image_distance_px(
    *,
    human_x: float,
    human_y: float,
    ball_x: float,
    ball_y: float,
) -> float:
    return math.hypot(float(ball_x) - float(human_x), float(ball_y) - float(human_y))


def pitch_distance_m(
    *,
    human_x_m: float | None,
    human_y_m: float | None,
    ball_x_m: float | None,
    ball_y_m: float | None,
) -> float | None:
    if human_x_m is None or human_y_m is None or ball_x_m is None or ball_y_m is None:
        return None
    return math.hypot(float(ball_x_m) - float(human_x_m), float(ball_y_m) - float(human_y_m))


def normalize_image_distance(
    distance_px: float, *, frame_width: float, frame_height: float, mode: str
) -> float:
    if mode != "frame_diagonal":
        return distance_px
    diag = math.hypot(float(frame_width), float(frame_height))
    if diag <= 0:
        return distance_px
    return distance_px / diag


def build_proximity_row(
    point: Mapping[str, Any],
    *,
    proximity_id: str,
    policy_fingerprint: str,
    config: Mapping[str, Any],
    is_nearest_human: bool,
) -> dict[str, Any]:
    prox_cfg = config.get("proximity") or {}
    img = _finite(point.get("image_distance_px"))
    if (
        img is None
        and point.get("human_image_x") is not None
        and point.get("ball_image_x") is not None
    ):
        img = image_distance_px(
            human_x=float(point["human_image_x"]),
            human_y=float(point["human_image_y"]),
            ball_x=float(point["ball_image_x"]),
            ball_y=float(point["ball_image_y"]),
        )
    pitch = _finite(point.get("pitch_distance_m"))
    if pitch is None:
        pitch = pitch_distance_m(
            human_x_m=_finite(point.get("human_pitch_x_m")),
            human_y_m=_finite(point.get("human_pitch_y_m")),
            ball_x_m=_finite(point.get("ball_pitch_x_m")),
            ball_y_m=_finite(point.get("ball_pitch_y_m")),
        )

    cand = {
        "calibration_status": str(point.get("calibration_status", "unknown")),
        "ball_air_state": str(point.get("ball_air_state", "unknown")),
        "ball_observation_state": str(point.get("ball_observation_state", "missing")),
        "human_observation_state": str(point.get("human_observation_state", "missing")),
    }
    usable, pitch_reasons = pitch_distance_usable(cand)
    if not usable:
        pitch = None

    reasons: list[str] = list(point.get("reason_codes") or [])
    reasons.extend(pitch_reasons)

    evidence_level = "candidate"
    eligibility = "eligible"
    play = str(point.get("playability_status", "playable"))
    if play in {"replay", "non_playable"}:
        evidence_level = "not_evaluable"
        eligibility = "excluded"
        reasons.append("REPLAY_OR_CUT_TERMINATES")
    if str(point.get("ball_candidate_status", "primary")) == "ambiguous":
        evidence_level = "unknown"
        eligibility = "not_evaluable"
        reasons.append("AMBIGUOUS_PRIMARY_BALL")
    if str(point.get("ball_observation_state")) == "missing":
        evidence_level = "not_evaluable"
        eligibility = "not_evaluable"
        reasons.append("MISSING_BALL_NOT_NO_POSSESSION")
    if str(point.get("human_observation_state")) in {"predicted", "interpolated"}:
        evidence_level = "rejected"
        eligibility = "ineligible"
        reasons.append("PREDICTED_SOLE_EVIDENCE")
    if str(point.get("ball_observation_state")) in {"predicted", "interpolated"}:
        evidence_level = "rejected"
        eligibility = "ineligible"
        reasons.append("PREDICTED_SOLE_EVIDENCE")

    thr_px = float(prox_cfg.get("image_distance_threshold_px", 80.0))
    thr_m = float(prox_cfg.get("pitch_distance_threshold_m", 2.5))
    near = False
    if img is not None and img <= thr_px:
        near = True
    if usable and pitch is not None and pitch <= thr_m:
        near = True
    if not near and eligibility == "eligible":
        evidence_level = "rejected"
        eligibility = "ineligible"
        reasons.append("OUTSIDE_PROXIMITY_THRESHOLD")

    evidence_space = "none"
    if img is not None and usable and pitch is not None:
        evidence_space = "both"
    elif img is not None:
        evidence_space = "image"
    elif usable and pitch is not None:
        evidence_space = "pitch"

    return {
        "run_id": str(point["run_id"]),
        "video_id": str(point["video_id"]),
        "proximity_id": proximity_id,
        "human_track_id": int(point["human_track_id"]),
        "ball_track_id": point.get("ball_track_id"),
        "frame_index": int(point["frame_index"]),
        "video_time_us": int(point["video_time_us"]),
        "image_distance_px": img,
        "pitch_distance_m": pitch,
        "foot_reference_type": str(prox_cfg.get("foot_reference_type", "bbox_bottom_centre")),
        "ball_reference_type": str(prox_cfg.get("ball_reference_type", "detection_centre")),
        "human_observation_state": str(point.get("human_observation_state", "observed")),
        "ball_observation_state": str(point.get("ball_observation_state", "observed")),
        "evidence_space": evidence_space,
        "ball_air_state": str(point.get("ball_air_state", "unknown")),
        "ball_candidate_status": str(point.get("ball_candidate_status", "primary")),
        "calibration_status": str(point.get("calibration_status", "unknown")),
        "playability_status": play,
        "target_relationship": str(point.get("target_relationship", "confirmed_target")),
        "evidence_level": evidence_level,
        "is_nearest_human": bool(is_nearest_human),
        "nearest_implies_possession": False,
        "pitch_distance_usable": bool(usable and pitch is not None),
        "uncertainty": _finite(point.get("uncertainty")),
        "eligibility_status": eligibility,
        "human_observation_id": point.get("human_observation_id"),
        "ball_observation_id": point.get("ball_observation_id"),
        "human_projection_id": point.get("human_projection_id"),
        "ball_projection_id": point.get("ball_projection_id"),
        "identity_assignment_id": point.get("identity_assignment_id"),
        "analysis_window_id": point.get("analysis_window_id"),
        "manual_review_required": evidence_level in {"unknown", "contested"},
        "reason_codes": sorted(set(reasons)),
        "quality_flags": list(point.get("quality_flags") or []),
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def mark_nearest_and_build(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy_fingerprint: str,
) -> list[dict[str, Any]]:
    """Build proximity rows; mark nearest per (run,video,time) without implying possession."""
    by_time: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for p in points:
        key = (str(p["run_id"]), str(p["video_id"]), int(p["video_time_us"]))
        by_time.setdefault(key, []).append(p)

    rows: list[dict[str, Any]] = []
    seq = 0
    for key in sorted(by_time.keys(), key=lambda k: (k[0], k[1], k[2])):
        group = by_time[key]
        dists: list[tuple[float, Mapping[str, Any]]] = []
        for p in group:
            img = _finite(p.get("image_distance_px"))
            if img is None and p.get("human_image_x") is not None:
                img = image_distance_px(
                    human_x=float(p["human_image_x"]),
                    human_y=float(p["human_image_y"]),
                    ball_x=float(p["ball_image_x"]),
                    ball_y=float(p["ball_image_y"]),
                )
            dists.append((float(img if img is not None else 1e18), p))
        dists.sort(key=lambda t: t[0])
        nearest_track = int(dists[0][1]["human_track_id"]) if dists else None
        for _d, p in dists:
            seq += 1
            pid = f"prox_{seq:04d}"
            is_nearest = int(p["human_track_id"]) == nearest_track
            rows.append(
                build_proximity_row(
                    p,
                    proximity_id=pid,
                    policy_fingerprint=policy_fingerprint,
                    config=config,
                    is_nearest_human=is_nearest,
                )
            )
    return rows


__all__ = [
    "image_distance_px",
    "pitch_distance_m",
    "normalize_image_distance",
    "build_proximity_row",
    "mark_nearest_and_build",
]
