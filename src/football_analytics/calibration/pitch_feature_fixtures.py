"""Synthetic fixtures for Stage 8B pitch feature detection tests / smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RUNTIME_ROOT = Path("/home/fdoblak/workspace/pitch_feature_checks")


def assert_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not str(RUNTIME_ROOT).startswith("/home/fdoblak/workspace/"):
        raise RuntimeError("pitch feature runtime root escape")


def make_solid_rgb(
    *,
    width: int = 960,
    height: int = 540,
    color: tuple[int, int, int] = (34, 139, 34),
) -> np.ndarray:
    """Uint8 RGB image (optionally non-model size for preprocess tests)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = np.asarray(color, dtype=np.uint8)
    return arr


def make_pitch_like_rgb(*, width: int = 320, height: int = 180) -> np.ndarray:
    """Simple green field with white line strokes (not ground truth)."""
    img = make_solid_rgb(width=width, height=height, color=(40, 120, 40))
    # Horizontal + vertical white lines
    img[height // 2 - 1 : height // 2 + 1, :] = (240, 240, 240)
    img[:, width // 2 - 1 : width // 2 + 1] = (240, 240, 240)
    img[0:2, :] = (240, 240, 240)
    img[-2:, :] = (240, 240, 240)
    img[:, 0:2] = (240, 240, 240)
    img[:, -2:] = (240, 240, 240)
    return img


def make_analysis_window_row(
    *,
    run_id: str,
    video_id: str,
    window_id: str = "aw_01",
    start_frame: int = 0,
    end_frame_exclusive: int = 10,
    calibration_eligibility: str = "eligible",
    playability: str = "playable",
    graphics_status: str = "clean",
    view_class: str = "main_broadcast",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": window_id,
        "start_frame_index": int(start_frame),
        "end_frame_index_exclusive": int(end_frame_exclusive),
        "calibration_eligibility": calibration_eligibility,
        "playability": playability,
        "graphics_status": graphics_status,
        "view_class": view_class,
        "camera_motion": "static",
        "contract_version": 1,
    }


def make_frame_rows(
    *,
    run_id: str,
    video_id: str,
    n: int = 3,
    width: int = 320,
    height: int = 180,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": i,
                "video_time_us": i * 40_000,
                "width": width,
                "height": height,
                "contract_version": 1,
            }
        )
    return rows


def fixture_image_bundle() -> dict[str, Any]:
    """In-memory detect inputs for service smoke without video decode."""
    return {
        "run_id": "run_20260723T230000000001Z_c22200000001",
        "video_id": "video_pitch_fx",
        "images": [
            {
                "frame_index": 0,
                "video_time_us": 0,
                "rgb": make_pitch_like_rgb(width=320, height=180),
                "eligible": True,
            },
            {
                "frame_index": 1,
                "video_time_us": 40_000,
                "rgb": make_solid_rgb(width=320, height=180, color=(10, 10, 10)),
                "eligible": True,
            },
            {
                "frame_index": 2,
                "video_time_us": 80_000,
                "rgb": make_pitch_like_rgb(width=160, height=90),
                "eligible": False,
                "skip_reason": "not_eligible",
            },
        ],
    }


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "make_solid_rgb",
    "make_pitch_like_rgb",
    "make_analysis_window_row",
    "make_frame_rows",
    "fixture_image_bundle",
]
