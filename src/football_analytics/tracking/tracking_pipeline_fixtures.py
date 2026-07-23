"""Synthetic fixtures for Stage 6D tracking fusion (runtime / unit tests only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.tracking.fixtures import (
    _attr_row,
    _det_row,
    _life_row,
    _obs_row,
    _summary_row,
)

RUNTIME_ROOT = Path("/home/fdoblak/workspace/tracking_pipeline_checks")

SOURCE_SHA_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SOURCE_SHA_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TIMELINE_FP_A = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
TIMELINE_FP_B = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
DETECTION_FP_A = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
DETECTION_FP_B = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
WINDOW_FP_A = "1111111111111111111111111111111111111111111111111111111111111111"
WINDOW_FP_B = "2222222222222222222222222222222222222222222222222222222222222222"


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/tracking_pipeline_checks"):
        raise RuntimeError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    return resolved


def make_frame_rows(run_id: str, video_id: str, n: int, *, fps: int = 25) -> list[dict[str, Any]]:
    step = int(1_000_000 / fps)
    return [
        {
            "run_id": run_id,
            "video_id": video_id,
            "frame_index": i,
            "pts": i,
            "video_time_us": i * step,
            "duration_us": step,
            "is_key_frame": i == 0,
            "decode_status": "ok",
        }
        for i in range(n)
    ]


def make_analysis_window_row(
    run_id: str,
    video_id: str,
    *,
    n_frames: int = 4,
    shot_id: str = "shot_01",
    window_id: str = "aw_fuse_01",
    playability: str = "playable",
    tracking: str = "eligible",
    replay_status: str = "live",
    start_frame: int = 0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": window_id,
        "start_time_us": start_frame * 40000,
        "end_time_us": max(1, n_frames) * 40000,
        "start_frame_index": start_frame,
        "end_frame_index_exclusive": start_frame + n_frames,
        "shot_id": shot_id,
        "camera_segment_ids": ["cam_01"],
        "view_family": "main_broadcast",
        "framing_scale": "wide",
        "replay_status": replay_status,
        "graphics_status": "none",
        "playability": playability,
        "tracking_eligibility": tracking,
        "calibration_eligibility": "eligible",
        "identity_eligibility": "unknown",
        "ball_analysis_eligibility": "eligible",
        "live_event_eligibility": "eligible",
        "physical_metric_eligibility": "eligible",
        "decision_codes": ["TRACKING_ELIGIBLE"],
        "manual_review_required": False,
        "coverage": 1.0,
        "confidence": 1.0,
        "timeline_mapping_quality": "exact_identity",
        "source_refs": [shot_id],
        "policy_version": "1",
        "provenance_json": '{"stage":"6D"}',
        "contract_version": 1,
    }


def make_detection_receipt(
    *,
    run_id: str,
    video_id: str,
    source_sha: str = SOURCE_SHA_A,
    timeline_fp: str = TIMELINE_FP_A,
    detection_fp: str = DETECTION_FP_A,
    window_fp: str = WINDOW_FP_A,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_id": f"det_pipe_{run_id[:12]}",
        "run_id": run_id,
        "video_id": video_id,
        "pipeline_id": "detection_fusion_baseline",
        "pipeline_version": "1.0.0",
        "config_fingerprint": "a" * 64,
        "source_video_sha256": source_sha,
        "timeline_fingerprint": timeline_fp,
        "detection_bundle_fingerprint": detection_fp,
        "analysis_window_fingerprint": window_fp,
        "status": "succeeded",
        "quality_gate_status": "pass_with_findings",
        "ground_truth_evaluation_status": "NOT_EVALUATED_NO_REVIEWED_DETECTION_GROUND_TRUTH",
        "artifacts": {
            "source_video_sha256": source_sha,
            "timeline_fingerprint": timeline_fp,
            "detection_bundle_fingerprint": detection_fp,
            "analysis_window_fingerprint": window_fp,
        },
        "provenance": {"stage": "5E", "label": "synthetic"},
    }


def make_tracking_receipt(
    *,
    run_id: str,
    video_id: str,
    tracker_id: str,
    source_sha: str = SOURCE_SHA_A,
    timeline_fp: str = TIMELINE_FP_A,
    detection_fp: str = DETECTION_FP_A,
    window_fp: str = WINDOW_FP_A,
    config_fingerprint: str | None = None,
) -> dict[str, Any]:
    cfg = config_fingerprint or ("b" * 64)
    return {
        "schema_version": 1,
        "receipt_id": f"trk_{tracker_id[:12]}",
        "run_id": run_id,
        "video_id": video_id,
        "tracker_id": tracker_id,
        "config_fingerprint": cfg,
        "policy_fingerprint": "c" * 64,
        "source_video_sha256": source_sha,
        "timeline_fingerprint": timeline_fp,
        "detection_bundle_fingerprint": detection_fp,
        "analysis_window_fingerprint": window_fp,
        "status": "succeeded",
        "artifacts": {
            "source_video_sha256": source_sha,
            "timeline_fingerprint": timeline_fp,
            "detection_bundle_fingerprint": detection_fp,
            "analysis_window_fingerprint": window_fp,
        },
        "provenance": {"stage": "6B" if "human" in tracker_id else "6C", "label": "synthetic"},
    }


def build_minimal_tracking_fusion_inputs(
    run_id: str,
    video_id: str = "video_01",
    *,
    n_frames: int = 4,
    collide_track_ids: bool = True,
) -> dict[str, Any]:
    """Build synthetic detection + human/ball track tables for fusion tests (≤20 frames)."""
    n_frames = min(n_frames, 20)
    times = [i * 40000 for i in range(n_frames)]

    dets: list[dict[str, Any]] = []
    attrs: list[dict[str, Any]] = []
    h_obs: list[dict[str, Any]] = []
    b_obs: list[dict[str, Any]] = []
    h_life: list[dict[str, Any]] = []
    b_life: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []

    human_tid = 0
    ball_tid = 0 if collide_track_ids else 10

    for fi in range(n_frames):
        # Human detection id 0
        dets.append(
            _det_row(
                run_id, video_id, fi, 0, class_id=0, class_name="person", bbox=(10, 20, 40, 80)
            )
        )
        attrs.append(
            _attr_row(
                run_id,
                video_id,
                fi,
                0,
                entity_type="human",
                role_label="unknown" if fi % 2 else "player",
            )
        )
        h_obs.append(
            _obs_row(
                run_id,
                video_id,
                fi,
                human_tid,
                detection_id=0,
                observation_state="observed",
                class_id=0,
            )
        )
        # Ball detection id 1
        dets.append(
            _det_row(
                run_id,
                video_id,
                fi,
                1,
                class_id=32,
                class_name="sports_ball",
                bbox=(50, 50, 58, 58),
            )
        )
        attrs.append(_attr_row(run_id, video_id, fi, 1, entity_type="ball", role_label="unknown"))
        state = "observed"
        did: int | None = 1
        flags: list[str] = []
        if fi == n_frames - 1 and n_frames >= 3:
            # Last frame predicted gap sample
            state = "predicted"
            did = None
            flags = ["physical_metric_ineligible", "event_ineligible"]
            # still keep a ball det unassigned for that frame — remove last ball det association
        b_obs.append(
            _obs_row(
                run_id,
                video_id,
                fi,
                ball_tid,
                detection_id=did,
                observation_state=state,
                class_id=32,
                quality_flags=flags,
                bbox=(50, 50, 58, 58),
                confidence=None if state == "predicted" else 0.9,
            )
        )
        if state == "predicted":
            # Remove the unused ball detection for predicted frame to avoid unassigned noise
            # Keep detection — unassigned is fine for quality metrics.
            primary.append(
                {
                    "frame_index": fi,
                    "status": "no_candidate",
                    "primary_track_id": None,
                    "primary_detection_id": None,
                }
            )
        elif fi == 1 and n_frames > 2:
            primary.append(
                {
                    "frame_index": fi,
                    "status": "ambiguous",
                    "primary_track_id": None,
                    "primary_detection_id": None,
                }
            )
        else:
            primary.append(
                {
                    "frame_index": fi,
                    "status": "primary",
                    "primary_track_id": ball_tid,
                    "primary_detection_id": 1,
                }
            )

    h_life = [
        _life_row(
            run_id,
            video_id,
            human_tid,
            0,
            0,
            times[0],
            "tentative",
            None,
            entity_type="human",
            transition_reason="birth",
            observation_source="detection_associated",
        ),
        _life_row(
            run_id,
            video_id,
            human_tid,
            1,
            min(1, n_frames - 1),
            times[min(1, n_frames - 1)],
            "confirmed",
            "tentative",
            entity_type="human",
            transition_reason="confirm",
            observation_source="detection_associated",
        ),
    ]
    b_life = [
        _life_row(
            run_id,
            video_id,
            ball_tid,
            0,
            0,
            times[0],
            "tentative",
            None,
            entity_type="ball",
            transition_reason="birth",
            observation_source="detection_associated",
        ),
        _life_row(
            run_id,
            video_id,
            ball_tid,
            1,
            min(1, n_frames - 1),
            times[min(1, n_frames - 1)],
            "confirmed",
            "tentative",
            entity_type="ball",
            transition_reason="confirm",
            observation_source="detection_associated",
        ),
    ]

    h_sum = [_summary_row(run_id, video_id, human_tid, h_obs)]
    b_sum = [_summary_row(run_id, video_id, ball_tid, b_obs)]

    return {
        "detections": dets,
        "detection_attributes": attrs,
        "frames": make_frame_rows(run_id, video_id, n_frames),
        "analysis_windows": [
            make_analysis_window_row(run_id, video_id, n_frames=n_frames),
        ],
        "human_observations": h_obs,
        "human_summaries": h_sum,
        "human_lifecycle": h_life,
        "ball_observations": b_obs,
        "ball_summaries": b_sum,
        "ball_lifecycle": b_life,
        "primary_sidecar": primary,
        "detection_receipt": make_detection_receipt(run_id=run_id, video_id=video_id),
        "human_receipt": make_tracking_receipt(
            run_id=run_id, video_id=video_id, tracker_id="human_mot_baseline"
        ),
        "ball_receipt": make_tracking_receipt(
            run_id=run_id, video_id=video_id, tracker_id="ball_mot_baseline"
        ),
        "fixture_fingerprint": hash_canonical_json(
            {"n_frames": n_frames, "collide": collide_track_ids, "run_id": run_id}
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "RUNTIME_ROOT",
    "SOURCE_SHA_A",
    "SOURCE_SHA_B",
    "TIMELINE_FP_A",
    "TIMELINE_FP_B",
    "DETECTION_FP_A",
    "DETECTION_FP_B",
    "WINDOW_FP_A",
    "WINDOW_FP_B",
    "assert_runtime_root",
    "make_frame_rows",
    "make_analysis_window_row",
    "make_detection_receipt",
    "make_tracking_receipt",
    "build_minimal_tracking_fusion_inputs",
    "write_json",
]
