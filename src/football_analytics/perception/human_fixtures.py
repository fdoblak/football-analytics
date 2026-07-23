"""Synthetic fixtures helpers for Stage 5B human detection (runtime only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_detection_checks")


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/human_detection_checks"):
        raise RuntimeError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    return resolved


def write_tiny_mp4(path: Path, *, n_frames: int = 8, width: int = 64, height: int = 48) -> Path:
    """Write a tiny synthetic BGR video via OpenCV (no network)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")
    try:
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw a moving light rectangle so the detector may see something.
            x0 = 8 + (i % 5) * 4
            y0 = 8 + (i % 3) * 4
            cv2.rectangle(frame, (x0, y0), (x0 + 20, y0 + 36), (220, 180, 160), -1)
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
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": "aw_01",
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
        "ball_analysis_eligibility": "ineligible",
        "live_event_eligibility": "eligible",
        "physical_metric_eligibility": "eligible",
        "decision_codes": ["TRACKING_ELIGIBLE"],
        "manual_review_required": False,
        "coverage": 1.0,
        "confidence": 1.0,
        "timeline_mapping_quality": "exact_identity",
        "source_refs": ["shot_01", "cam_01"],
        "policy_version": "1",
        "provenance_json": '{"stage":"5B"}',
        "contract_version": 1,
    }


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "write_tiny_mp4",
    "make_frame_rows",
    "make_analysis_window_row",
]
