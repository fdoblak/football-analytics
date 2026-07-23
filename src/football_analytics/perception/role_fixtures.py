"""Synthetic fixtures for Stage 5D human role baseline (runtime / unit tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_role_checks")

# Frozen synthetic scenarios — NOT real football performance claims.
FROZEN_ROLE_FIXTURES: dict[str, dict[str, Any]] = {
    "two_kits_players": {
        "frame_width": 200,
        "frame_height": 120,
        "humans": [
            {
                "detection_id": 0,
                "bbox": [20, 30, 40, 80],
                "kit_hue": 0.05,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 1,
                "bbox": [50, 30, 70, 80],
                "kit_hue": 0.05,
                "kit_saturation": 0.72,
                "kit_value": 0.68,
            },
            {
                "detection_id": 2,
                "bbox": [110, 30, 130, 80],
                "kit_hue": 0.55,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 3,
                "bbox": [140, 30, 160, 80],
                "kit_hue": 0.55,
                "kit_saturation": 0.68,
                "kit_value": 0.72,
            },
        ],
        "expect_player_min": 4,
    },
    "gk_needs_extra_evidence": {
        "frame_width": 200,
        "frame_height": 120,
        "humans": [
            {
                "detection_id": 0,
                "bbox": [40, 30, 60, 80],
                "kit_hue": 0.0,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 1,
                "bbox": [70, 30, 90, 80],
                "kit_hue": 0.0,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 2,
                "bbox": [100, 30, 120, 80],
                "kit_hue": 0.5,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 3,
                "bbox": [130, 30, 150, 80],
                "kit_hue": 0.5,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            # Distinct color only (center) — must NOT become GK without extra evidence.
            {
                "detection_id": 4,
                "bbox": [90, 25, 110, 85],
                "kit_hue": 0.95,
                "kit_saturation": 0.85,
                "kit_value": 0.85,
            },
            # Distinct + lateral → GK candidate.
            {
                "detection_id": 5,
                "bbox": [2, 25, 22, 85],
                "kit_hue": 0.25,
                "kit_saturation": 0.85,
                "kit_value": 0.75,
            },
        ],
        "color_only_detection_id": 4,
        "gk_candidate_detection_id": 5,
    },
    "referee_dark": {
        "frame_width": 200,
        "frame_height": 120,
        "humans": [
            {
                "detection_id": 0,
                "bbox": [40, 30, 60, 80],
                "kit_hue": 0.1,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 1,
                "bbox": [70, 30, 90, 80],
                "kit_hue": 0.1,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 2,
                "bbox": [110, 30, 130, 80],
                "kit_hue": 0.55,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 3,
                "bbox": [140, 30, 160, 80],
                "kit_hue": 0.55,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            # Dark alone without non-outfield margin handled in classification.
            {
                "detection_id": 4,
                "bbox": [95, 30, 115, 80],
                "kit_hue": 0.0,
                "kit_saturation": 0.1,
                "kit_value": 0.2,
            },
        ],
        "ref_detection_id": 4,
    },
    "tiny_crop_abstain": {
        "frame_width": 200,
        "frame_height": 120,
        "humans": [
            {
                "detection_id": 0,
                "bbox": [10, 10, 14, 18],
                "kit_hue": 0.2,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
                "crop_quality": 0.05,
            }
        ],
    },
    "conflict": {
        "frame_width": 200,
        "frame_height": 120,
        "humans": [
            {
                "detection_id": 0,
                "bbox": [40, 30, 60, 80],
                "kit_hue": 0.0,
                "kit_saturation": 0.15,
                "kit_value": 0.25,
            },
            {
                "detection_id": 1,
                "bbox": [70, 30, 90, 80],
                "kit_hue": 0.0,
                "kit_saturation": 0.12,
                "kit_value": 0.22,
            },
            {
                "detection_id": 2,
                "bbox": [110, 30, 130, 80],
                "kit_hue": 0.55,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
            {
                "detection_id": 3,
                "bbox": [140, 30, 160, 80],
                "kit_hue": 0.55,
                "kit_saturation": 0.7,
                "kit_value": 0.7,
            },
        ],
    },
}


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/human_role_checks"):
        raise RuntimeError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    return resolved


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
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "entity_type": "human",
        "role_label": "unknown",
        "role_source": "detector_native",
        "role_score": None,
        "occlusion": None,
        "truncation": None,
        "visibility": None,
        "review_status": "unreviewed",
        "attribute_source_ref": "detector:person",
        "provenance_json": '{"stage":"5B"}',
        "contract_version": 1,
    }


def make_frame_status_row(
    run_id: str,
    video_id: str,
    *,
    frame_index: int,
    processing_status: str = "processed",
    eligibility: str = "eligible",
    human_count: int = 1,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "video_time_us": frame_index * 125000,
        "analysis_window_id": "aw_role_01",
        "processing_status": processing_status,
        "eligibility": eligibility,
        "detector_id": "human_yolo11n",
        "input_artifact_ref": None,
        "detection_count": human_count,
        "human_count": human_count,
        "ball_count": 0,
        "skip_reason": None if processing_status == "processed" else "test_skip",
        "error_code": None,
        "coverage": 1.0,
        "provenance_json": '{"stage":"5B"}',
        "contract_version": 1,
    }


def make_analysis_window_row(
    run_id: str,
    video_id: str,
    *,
    n_frames: int = 1,
    playability: str = "playable",
    tracking: str = "eligible",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": "aw_role_01",
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
        "provenance_json": '{"stage":"5D"}',
        "contract_version": 1,
    }


__all__ = [
    "RUNTIME_ROOT",
    "FROZEN_ROLE_FIXTURES",
    "assert_runtime_root",
    "make_detection_row",
    "make_human_attribute_row",
    "make_frame_status_row",
    "make_analysis_window_row",
]
