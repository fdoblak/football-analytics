"""Synthetic fixtures for Stage 5E detection fusion (runtime / unit tests only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json

RUNTIME_ROOT = Path("/home/fdoblak/workspace/detection_pipeline_checks")

SOURCE_SHA_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SOURCE_SHA_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TIMELINE_FP_A = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
TIMELINE_FP_B = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/detection_pipeline_checks"):
        raise RuntimeError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    return resolved


def make_frame_rows(run_id: str, video_id: str, n: int, *, fps: int = 8) -> list[dict[str, Any]]:
    step = int(1_000_000 / fps)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
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
        )
    return rows


def make_analysis_window_row(
    run_id: str,
    video_id: str,
    *,
    n_frames: int = 4,
    playability: str = "playable",
    tracking: str = "eligible",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": "aw_fuse_01",
        "start_time_us": 0,
        "end_time_us": max(1, n_frames) * 125000,
        "start_frame_index": 0,
        "end_frame_index_exclusive": n_frames,
        "shot_id": "shot_01",
        "camera_segment_ids": ["cam_01"],
        "view_family": "main_broadcast",
        "framing_scale": "wide",
        "replay_status": "live",
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
        "source_refs": ["shot_01", "cam_01"],
        "policy_version": "1",
        "provenance_json": '{"stage":"5E"}',
        "contract_version": 1,
    }


def make_detection_row(
    run_id: str,
    video_id: str,
    *,
    frame_index: int,
    detection_id: int,
    bbox: list[float],
    score: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
    model_id: str = "human_yolo11n",
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": float(score),
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
        "bbox_x2": float(x2),
        "bbox_y2": float(y2),
        "model_id": model_id,
        "is_interpolated": False,
        "quality_flags": [],
    }


def make_human_attribute_row(
    run_id: str,
    video_id: str,
    *,
    frame_index: int,
    detection_id: int,
    role_label: str = "unknown",
    role_source: str = "downstream_classifier",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "entity_type": "human",
        "role_label": role_label,
        "role_source": role_source,
        "role_score": None,
        "occlusion": None,
        "truncation": None,
        "visibility": None,
        "review_status": "unreviewed",
        "attribute_source_ref": "role:baseline",
        "provenance_json": '{"stage":"5D"}',
        "contract_version": 1,
    }


def make_ball_attribute_row(
    run_id: str,
    video_id: str,
    *,
    frame_index: int,
    detection_id: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "entity_type": "ball",
        "role_label": "unknown",
        "role_source": "detector_native",
        "role_score": None,
        "occlusion": None,
        "truncation": None,
        "visibility": None,
        "review_status": "unreviewed",
        "attribute_source_ref": "detector:sports_ball",
        "provenance_json": '{"stage":"5C"}',
        "contract_version": 1,
    }


def make_frame_status_row(
    run_id: str,
    video_id: str,
    *,
    frame_index: int,
    processing_status: str = "processed",
    eligibility: str = "eligible",
    human_count: int = 0,
    ball_count: int = 0,
    detector_id: str = "human_yolo11n",
) -> dict[str, Any]:
    det_count = human_count + ball_count
    if processing_status in {
        "skipped",
        "failed",
        "not_eligible",
        "processed_no_detections",
    }:
        det_count = 0
        human_count = 0
        ball_count = 0
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "video_time_us": frame_index * 125000,
        "analysis_window_id": "aw_fuse_01",
        "processing_status": processing_status,
        "eligibility": eligibility,
        "detector_id": detector_id,
        "input_artifact_ref": None,
        "detection_count": det_count,
        "human_count": human_count,
        "ball_count": ball_count,
        "skip_reason": None if processing_status == "processed" else "test_skip",
        "error_code": "TEST_FAIL" if processing_status == "failed" else None,
        "coverage": 1.0,
        "provenance_json": '{"stage":"5E"}',
        "contract_version": 1,
    }


def make_detection_receipt(
    *,
    run_id: str,
    video_id: str,
    detector_id: str,
    frames_ref: str = "frames.parquet",
    source_sha: str = SOURCE_SHA_A,
    timeline_fp: str = TIMELINE_FP_A,
    config_fingerprint: str | None = None,
    human_count: int = 0,
    ball_count: int = 0,
    processed: int = 1,
) -> dict[str, Any]:
    cfg = config_fingerprint or ("e" * 64)
    return {
        "schema_version": 1,
        "receipt_id": f"receipt_{detector_id[:12]}",
        "run_id": run_id,
        "video_id": video_id,
        "detector_id": detector_id,
        "model_registry_id": None,
        "model_sha256": None,
        "adapter_id": "synthetic",
        "adapter_version": "1.0.0",
        "config_fingerprint": cfg,
        "taxonomy_version": "1",
        "source_video_ref": "/tmp/synthetic.mp4",
        "frames_ref": frames_ref,
        "analysis_windows_ref": "analysis_windows.parquet",
        "source_video_sha256": source_sha,
        "timeline_fingerprint": timeline_fp,
        "eligible_frame_count": processed,
        "processed_frame_count": processed,
        "skipped_frame_count": 0,
        "failed_frame_count": 0,
        "processed_no_detection_count": 0,
        "total_detection_count": human_count + ball_count,
        "human_detection_count": human_count,
        "ball_detection_count": ball_count,
        "pre_nms_count": None,
        "post_nms_count": human_count + ball_count,
        "started_at_utc": "2026-07-23T00:00:00.000000Z",
        "completed_at_utc": "2026-07-23T00:00:01.000000Z",
        "status": "succeeded",
        "warnings": [],
        "errors": [],
        "artifacts": {
            "source_video_sha256": source_sha,
            "timeline_fingerprint": timeline_fp,
        },
        "environment_ref": None,
        "provenance": {"stage": "5A", "label": "synthetic"},
    }


def make_role_receipt(
    *,
    run_id: str,
    video_id: str,
    source_sha: str = SOURCE_SHA_A,
    timeline_fp: str = TIMELINE_FP_A,
    config_fingerprint: str | None = None,
) -> dict[str, Any]:
    cfg = config_fingerprint or ("f" * 64)
    return {
        "schema_version": 1,
        "receipt_id": f"role_{run_id[:12]}",
        "run_id": run_id,
        "video_id": video_id,
        "classifier_id": "human_role_baseline_hsv",
        "config_fingerprint": cfg,
        "source_video_sha256": source_sha,
        "timeline_fingerprint": timeline_fp,
        "status": "succeeded",
        "assignment_counts": {"classified": 1, "abstained": 0},
        "role_counts": {"unknown": 0, "player": 1},
        "provenance": {"stage": "5D", "label": "synthetic"},
    }


def build_minimal_fusion_inputs(
    run_id: str,
    video_id: str = "video_01",
    *,
    n_frames: int = 4,
    collide_ball_id: bool = True,
) -> dict[str, Any]:
    """Build synthetic human/ball/role tables for fusion tests (≤20 frames)."""
    n_frames = min(n_frames, 20)
    human_dets = []
    human_attrs = []
    human_status = []
    ball_dets = []
    ball_attrs = []
    ball_status = []
    role_attrs = []

    for fi in range(n_frames):
        # Human det id 0
        human_dets.append(
            make_detection_row(
                run_id,
                video_id,
                frame_index=fi,
                detection_id=0,
                bbox=[10, 20, 40, 80],
                class_name="person",
            )
        )
        human_attrs.append(
            make_human_attribute_row(
                run_id, video_id, frame_index=fi, detection_id=0, role_label="unknown"
            )
        )
        role_attrs.append(
            make_human_attribute_row(
                run_id,
                video_id,
                frame_index=fi,
                detection_id=0,
                role_label="player" if fi % 2 == 0 else "unknown",
            )
        )
        human_status.append(
            make_frame_status_row(
                run_id,
                video_id,
                frame_index=fi,
                human_count=1,
                ball_count=0,
                detector_id="human_yolo11n",
            )
        )
        # Ball — optionally same detection_id to force remap
        ball_id = 0 if collide_ball_id else 1
        ball_dets.append(
            make_detection_row(
                run_id,
                video_id,
                frame_index=fi,
                detection_id=ball_id,
                bbox=[50, 50, 58, 58],
                class_id=32,
                class_name="sports_ball",
                model_id="ball_yolo11n",
            )
        )
        ball_attrs.append(
            make_ball_attribute_row(run_id, video_id, frame_index=fi, detection_id=ball_id)
        )
        ball_status.append(
            make_frame_status_row(
                run_id,
                video_id,
                frame_index=fi,
                human_count=0,
                ball_count=1,
                detector_id="ball_yolo11n",
            )
        )

    return {
        "human_detections": human_dets,
        "human_attributes": human_attrs,
        "human_frame_status": human_status,
        "ball_detections": ball_dets,
        "ball_attributes": ball_attrs,
        "ball_frame_status": ball_status,
        "role_attributes": role_attrs,
        "frames": make_frame_rows(run_id, video_id, n_frames),
        "analysis_windows": [make_analysis_window_row(run_id, video_id, n_frames=n_frames)],
        "human_receipt": make_detection_receipt(
            run_id=run_id,
            video_id=video_id,
            detector_id="human_yolo11n",
            human_count=n_frames,
            processed=n_frames,
        ),
        "ball_receipt": make_detection_receipt(
            run_id=run_id,
            video_id=video_id,
            detector_id="ball_yolo11n",
            ball_count=n_frames,
            processed=n_frames,
        ),
        "role_receipt": make_role_receipt(run_id=run_id, video_id=video_id),
        "fixture_fingerprint": hash_canonical_json(
            {"n_frames": n_frames, "collide": collide_ball_id, "run_id": run_id}
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
    "assert_runtime_root",
    "make_frame_rows",
    "make_analysis_window_row",
    "make_detection_row",
    "make_human_attribute_row",
    "make_ball_attribute_row",
    "make_frame_status_row",
    "make_detection_receipt",
    "make_role_receipt",
    "build_minimal_fusion_inputs",
    "write_json",
]
