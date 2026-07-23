"""Synthetic fixtures helpers for Stage 5C ball detection (runtime only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RUNTIME_ROOT = Path("/home/fdoblak/workspace/ball_detection_checks")


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/ball_detection_checks"):
        raise RuntimeError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    return resolved


def write_tiny_mp4_with_ball(
    path: Path, *, n_frames: int = 8, width: int = 128, height: int = 96
) -> Path:
    """Write a tiny synthetic BGR video with a small moving bright blob (ball-like)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")
    try:
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Green pitch-ish background
            frame[:, :] = (40, 110, 40)
            cx = 20 + (i % 6) * 8
            cy = 30 + (i % 4) * 6
            cv2.circle(frame, (cx, cy), 4, (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


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
    n_frames: int,
    tracking: str = "eligible",
    playability: str = "playable",
    ball_analysis: str = "eligible",
    identity: str = "unknown",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": "aw_ball_01",
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
        "identity_eligibility": identity,
        "ball_analysis_eligibility": ball_analysis,
        "live_event_eligibility": "eligible",
        "physical_metric_eligibility": "eligible",
        "decision_codes": ["BALL_ANALYSIS_ELIGIBLE"],
        "manual_review_required": False,
        "coverage": 1.0,
        "confidence": 1.0,
        "timeline_mapping_quality": "exact_identity",
        "source_refs": ["shot_01", "cam_01"],
        "policy_version": "1",
        "provenance_json": '{"stage":"5C"}',
        "contract_version": 1,
    }


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "write_tiny_mp4_with_ball",
    "make_frame_rows",
    "make_analysis_window_row",
]
