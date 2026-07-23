"""Synthetic-only tracking contract fixtures (Stage 6A; no video/model)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.tracking.types import (
    CONTRACT_VERSION,
    LifecycleState,
    ObservationSource,
    TrackEntityType,
)


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    spec = get_contract(name, 1)
    schema = compile_arrow_schema(spec)
    return pa.Table.from_pylist(rows, schema=schema)


def base_context(
    *,
    run_id: str | None = None,
    video_id: str = "clip_track_01",
    n_frames: int = 10,
    vfr: bool = False,
) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    videos = _cast(
        "videos",
        [
            {
                "run_id": rid,
                "video_id": video_id,
                "source_sha256": "b" * 64,
                "container": "mp4",
                "codec": "h264",
                "width_px": 1280,
                "height_px": 720,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "time_base_numerator": 1,
                "time_base_denominator": 25,
                "frame_count": n_frames,
                "duration_us": None if vfr else n_frames * 40000,
                "has_audio": False,
                "source_ref": f"logical_{video_id}",
            }
        ],
    )
    # VFR: irregular microsecond gaps (no fps invent).
    times = []
    t = 0
    for i in range(n_frames):
        times.append(t)
        t += 33000 if (vfr and i % 2 == 0) else 40000
    frames = _cast(
        "frames",
        [
            {
                "run_id": rid,
                "video_id": video_id,
                "frame_index": i,
                "pts": i,
                "video_time_us": times[i],
                "duration_us": (times[i + 1] - times[i]) if i + 1 < n_frames else 40000,
                "is_key_frame": i % 4 == 0,
                "decode_status": "ok",
            }
            for i in range(n_frames)
        ],
    )
    windows = _cast(
        "analysis_windows",
        [
            {
                "run_id": rid,
                "video_id": video_id,
                "analysis_window_id": "aw_play_001",
                "start_time_us": 0,
                "end_time_us": times[min(5, n_frames - 1)],
                "start_frame_index": 0,
                "end_frame_index_exclusive": min(6, n_frames),
                "shot_id": "shot_001",
                "camera_segment_ids": ["cam_001"],
                "view_family": "main_broadcast",
                "framing_scale": "wide",
                "replay_status": "live",
                "graphics_status": "none",
                "playability": "playable",
                "tracking_eligibility": "eligible",
                "calibration_eligibility": "eligible",
                "identity_eligibility": "conditionally_eligible",
                "ball_analysis_eligibility": "eligible",
                "live_event_eligibility": "unknown",
                "physical_metric_eligibility": "eligible",
                "decision_codes": ["PLAYABLE_WIDE_VIEW"],
                "manual_review_required": False,
                "coverage": 1.0,
                "confidence": 0.95,
                "timeline_mapping_quality": "exact_identity",
                "source_refs": ["shot_001"],
                "policy_version": "1",
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": rid,
                "video_id": video_id,
                "analysis_window_id": "aw_graphics_001",
                "start_time_us": times[min(6, n_frames - 1)] if n_frames > 6 else times[-1],
                "end_time_us": times[-1] + 40000,
                "start_frame_index": min(6, n_frames - 1),
                "end_frame_index_exclusive": n_frames,
                "shot_id": "shot_002",
                "camera_segment_ids": ["cam_002"],
                "view_family": "graphics",
                "framing_scale": "unknown",
                "replay_status": "live",
                "graphics_status": "full_screen",
                "playability": "non_playable",
                "tracking_eligibility": "ineligible",
                "calibration_eligibility": "ineligible",
                "identity_eligibility": "ineligible",
                "ball_analysis_eligibility": "ineligible",
                "live_event_eligibility": "ineligible",
                "physical_metric_eligibility": "ineligible",
                "decision_codes": ["GRAPHICS_NON_PLAYABLE"],
                "manual_review_required": False,
                "coverage": 1.0,
                "confidence": 0.9,
                "timeline_mapping_quality": "exact_identity",
                "source_refs": ["shot_002"],
                "policy_version": "1",
                "provenance_json": None,
                "contract_version": 1,
            },
        ],
    )
    return {
        "run_id": rid,
        "video_id": video_id,
        "videos": videos,
        "frames": frames,
        "analysis_windows": windows,
        "times": times,
    }


def _det_row(
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    *,
    class_id: int = 0,
    class_name: str = "person",
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 40.0, 80.0),
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "model_id": "det_dummy_v1",
        "is_interpolated": False,
        "quality_flags": [],
    }


def _attr_row(
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    *,
    entity_type: str = "human",
    role_label: str = "unknown",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "entity_type": entity_type,
        "role_label": role_label,
        "role_source": "unknown",
        "role_score": None,
        "occlusion": None,
        "truncation": None,
        "visibility": None,
        "review_status": "unreviewed",
        "attribute_source_ref": None,
        "provenance_json": None,
        "contract_version": 1,
    }


def _obs_row(
    run_id: str,
    video_id: str,
    frame_index: int,
    track_id: int,
    *,
    detection_id: int | None,
    observation_state: str,
    quality_flags: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 40.0, 80.0),
    class_id: int = 0,
    confidence: float | None = 0.9,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "track_id": track_id,
        "detection_id": detection_id,
        "class_id": class_id,
        "confidence": confidence,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "observation_state": observation_state,
        "model_id": "track_contract_synth_v1",
        "quality_flags": list(quality_flags or []),
    }


def _life_row(
    run_id: str,
    video_id: str,
    track_id: int,
    event_index: int,
    frame_index: int,
    video_time_us: int,
    lifecycle_state: str,
    previous_state: str | None,
    *,
    entity_type: str = "human",
    transition_reason: str = "synthetic",
    observation_source: str | None = None,
    manual_review_required: bool = False,
    quality_flags: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "track_id": track_id,
        "event_index": event_index,
        "frame_index": frame_index,
        "video_time_us": video_time_us,
        "lifecycle_state": lifecycle_state,
        "previous_state": previous_state,
        "entity_type": entity_type,
        "transition_reason": transition_reason,
        "observation_source": observation_source,
        "manual_review_required": manual_review_required,
        "quality_flags": list(quality_flags or []),
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def _summary_row(
    run_id: str,
    video_id: str,
    track_id: int,
    obs: Sequence[Mapping[str, Any]],
    *,
    termination_reason: str = "end_of_clip",
    quality_flags: Sequence[str] | None = None,
) -> dict[str, Any]:
    frames_idx = [int(o["frame_index"]) for o in obs]
    observed = sum(1 for o in obs if o["observation_state"] == "observed")
    predicted = sum(1 for o in obs if o["observation_state"] == "predicted")
    confs = [float(o["confidence"]) for o in obs if o.get("confidence") is not None]
    return {
        "run_id": run_id,
        "video_id": video_id,
        "track_id": track_id,
        "class_id": int(obs[0]["class_id"]),
        "first_frame_index": min(frames_idx),
        "last_frame_index": max(frames_idx),
        "observation_count": len(obs),
        "observed_count": observed,
        "predicted_count": predicted,
        "mean_confidence": (sum(confs) / len(confs)) if confs else None,
        "max_confidence": max(confs) if confs else None,
        "termination_reason": termination_reason,
        "quality_flags": list(quality_flags or []),
    }


def valid_birth_confirmed_bundle(*, run_id: str | None = None) -> dict[str, Any]:
    """Scenario 1: birth → tentative → confirmed with detection associations."""
    ctx = base_context(run_id=run_id, n_frames=8)
    rid, vid = ctx["run_id"], ctx["video_id"]
    times = ctx["times"]
    dets = [_det_row(rid, vid, i, i) for i in range(3)]
    attrs = [_attr_row(rid, vid, i, i, role_label="unknown") for i in range(3)]
    obs = [_obs_row(rid, vid, i, 0, detection_id=i, observation_state="observed") for i in range(3)]
    life = [
        _life_row(
            rid,
            vid,
            0,
            0,
            0,
            times[0],
            LifecycleState.TENTATIVE.value,
            None,
            observation_source=ObservationSource.DETECTION_ASSOCIATED.value,
            transition_reason="birth",
        ),
        _life_row(
            rid,
            vid,
            0,
            1,
            2,
            times[2],
            LifecycleState.CONFIRMED.value,
            LifecycleState.TENTATIVE.value,
            observation_source=ObservationSource.DETECTION_ASSOCIATED.value,
            transition_reason="confirmation_threshold",
        ),
    ]
    return {
        **ctx,
        "detections": _cast("detections", dets),
        "detection_attributes": _cast("detection_attributes", attrs),
        "track_observations": _cast("track_observations", obs),
        "track_summaries": _cast("track_summaries", [_summary_row(rid, vid, 0, obs)]),
        "track_lifecycle": _cast("track_lifecycle", life),
    }


def lost_recover_bundle(*, run_id: str | None = None) -> dict[str, Any]:
    """Scenario 2: confirmed → lost → recovered (within max_lost_gap_us)."""
    ctx = base_context(run_id=run_id, n_frames=8)
    rid, vid = ctx["run_id"], ctx["video_id"]
    times = ctx["times"]
    dets = [_det_row(rid, vid, i, i) for i in (0, 1, 4)]
    attrs = [_attr_row(rid, vid, i, i) for i in (0, 1, 4)]
    obs = [
        _obs_row(rid, vid, 0, 0, detection_id=0, observation_state="observed"),
        _obs_row(rid, vid, 1, 0, detection_id=1, observation_state="observed"),
        _obs_row(
            rid,
            vid,
            2,
            0,
            detection_id=None,
            observation_state="predicted",
            quality_flags=["physical_metric_ineligible"],
            confidence=None,
        ),
        _obs_row(rid, vid, 4, 0, detection_id=4, observation_state="observed"),
    ]
    life = [
        _life_row(rid, vid, 0, 0, 0, times[0], "tentative", None, transition_reason="birth"),
        _life_row(
            rid, vid, 0, 1, 1, times[1], "confirmed", "tentative", transition_reason="confirm"
        ),
        _life_row(rid, vid, 0, 2, 2, times[2], "lost", "confirmed", transition_reason="miss"),
        _life_row(rid, vid, 0, 3, 4, times[4], "confirmed", "lost", transition_reason="recover"),
    ]
    return {
        **ctx,
        "detections": _cast("detections", dets),
        "detection_attributes": _cast("detection_attributes", attrs),
        "track_observations": _cast("track_observations", obs),
        "track_summaries": _cast("track_summaries", [_summary_row(rid, vid, 0, obs)]),
        "track_lifecycle": _cast("track_lifecycle", life),
    }


def terminated_bundle(*, run_id: str | None = None) -> dict[str, Any]:
    """Scenario 3: confirmed → terminated."""
    ctx = base_context(run_id=run_id, n_frames=6)
    rid, vid = ctx["run_id"], ctx["video_id"]
    times = ctx["times"]
    dets = [_det_row(rid, vid, 0, 0), _det_row(rid, vid, 1, 1)]
    attrs = [_attr_row(rid, vid, 0, 0), _attr_row(rid, vid, 1, 1)]
    obs = [
        _obs_row(rid, vid, 0, 0, detection_id=0, observation_state="observed"),
        _obs_row(rid, vid, 1, 0, detection_id=1, observation_state="observed"),
    ]
    life = [
        _life_row(rid, vid, 0, 0, 0, times[0], "tentative", None, transition_reason="birth"),
        _life_row(
            rid, vid, 0, 1, 1, times[1], "confirmed", "tentative", transition_reason="confirm"
        ),
        _life_row(
            rid,
            vid,
            0,
            2,
            1,
            times[1],
            "terminated",
            "confirmed",
            transition_reason="end_of_window",
        ),
    ]
    return {
        **ctx,
        "detections": _cast("detections", dets),
        "detection_attributes": _cast("detection_attributes", attrs),
        "track_observations": _cast("track_observations", obs),
        "track_summaries": _cast(
            "track_summaries",
            [_summary_row(rid, vid, 0, obs, termination_reason="end_of_window")],
        ),
        "track_lifecycle": _cast("track_lifecycle", life),
    }


def mutate_lifecycle_reopen(bundle: dict[str, Any]) -> dict[str, Any]:
    """Scenario 4 helper: append illegal reopen after terminated."""
    rows = bundle["track_lifecycle"].to_pylist()
    last = rows[-1]
    rows.append(
        _life_row(
            last["run_id"],
            last["video_id"],
            int(last["track_id"]),
            int(last["event_index"]) + 1,
            int(last["frame_index"]),
            int(last["video_time_us"]),
            "confirmed",
            "terminated",
            transition_reason="illegal_reopen",
        )
    )
    out = dict(bundle)
    out["track_lifecycle"] = _cast("track_lifecycle", rows)
    return out


__all__ = [
    "TrackEntityType",
    "base_context",
    "valid_birth_confirmed_bundle",
    "lost_recover_bundle",
    "terminated_bundle",
    "mutate_lifecycle_reopen",
    "_cast",
    "_det_row",
    "_attr_row",
    "_obs_row",
    "_life_row",
    "_summary_row",
]
