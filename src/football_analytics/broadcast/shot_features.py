"""Streaming consecutive-frame features for shot boundary detection (OpenCV)."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ShotFeatureError(ValueError):
    """Feature extraction failure."""


@dataclass(frozen=True)
class FeatureFrame:
    frame_index: int
    video_time_us: int
    luma_mae: float
    hist_distance: float
    edge_change_ratio: float
    mean_luma: float


def _reject_non_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ShotFeatureError(f"{name} is non-finite")
    return float(value)


def _resize_bgr(frame: Any, *, width: int, height: int) -> Any:
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        raise ShotFeatureError("empty frame")
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _mean_luma(gray: Any) -> float:
    return _reject_non_finite("mean_luma", float(np.mean(gray)) / 255.0)


def _luma_mae(prev_gray: Any, gray: Any) -> float:
    diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
    return _reject_non_finite("luma_mae", float(np.mean(diff)) / 255.0)


def _hist_distance(prev_bgr: Any, bgr: Any) -> float:
    import cv2

    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist_prev = cv2.calcHist([prev_hsv], [0, 1, 2], None, [12, 16, 16], [0, 180, 0, 256, 0, 256])
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist, hist)
    corr = float(cv2.compareHist(hist_prev, hist, cv2.HISTCMP_CORREL))
    if not math.isfinite(corr):
        raise ShotFeatureError("hist correlation is non-finite")
    # Map correlation [-1,1] → distance [0,1]
    distance = max(0.0, min(1.0, 0.5 * (1.0 - corr)))
    return _reject_non_finite("hist_distance", distance)


def _edge_magnitude(gray: Any) -> Any:
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _edge_change_ratio(prev_gray: Any, gray: Any) -> float:
    prev_e = _edge_magnitude(prev_gray)
    cur_e = _edge_magnitude(gray)
    prev_mean = float(np.mean(prev_e))
    cur_mean = float(np.mean(cur_e))
    denom = max(prev_mean, cur_mean, 1.0)
    ratio = abs(cur_mean - prev_mean) / denom
    return _reject_non_finite("edge_change_ratio", max(0.0, min(1.0, ratio)))


def load_timeline_times(frames_table: Any) -> list[tuple[int, int]]:
    """Return sorted (frame_index, video_time_us) from a frames contract table."""
    if frames_table is None or frames_table.num_rows == 0:
        raise ShotFeatureError("frames timeline is empty")
    rows = frames_table.to_pylist()
    pairs = [(int(r["frame_index"]), int(r["video_time_us"])) for r in rows]
    pairs.sort(key=lambda x: x[0])
    for i, (idx, _t) in enumerate(pairs):
        if idx != i:
            raise ShotFeatureError("frames timeline must be contiguous from 0")
    return pairs


def build_cfr_timeline(
    *, frame_count: int, fps_num: int = 25, fps_den: int = 1
) -> list[tuple[int, int]]:
    """Build deterministic CFR (frame_index, video_time_us) pairs for synthetic fixtures."""
    if frame_count < 1:
        raise ShotFeatureError("frame_count must be >= 1")
    if fps_num <= 0 or fps_den <= 0:
        raise ShotFeatureError("fps must be positive")
    out: list[tuple[int, int]] = []
    for i in range(frame_count):
        time_us = int(round((i * fps_den * 1_000_000) / fps_num))
        out.append((i, time_us))
    return out


def iter_feature_frames(
    source: Path | str,
    timeline: Sequence[tuple[int, int]],
    *,
    analysis_width: int,
    analysis_height: int,
    max_frames: int,
    timeout_seconds: float,
) -> Iterator[FeatureFrame]:
    """Stream consecutive-frame features; times come from timeline, never fps invention."""
    import cv2

    path = Path(source)
    if not path.is_file() or path.is_symlink():
        raise ShotFeatureError("source must be a regular non-symlink file")
    if not timeline:
        raise ShotFeatureError("timeline required")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ShotFeatureError("failed to open video source")

    started = time.monotonic()
    prev_bgr = None
    prev_gray = None
    decoded = 0
    try:
        while decoded < len(timeline):
            if decoded >= max_frames:
                raise ShotFeatureError("max_frames exceeded")
            if (time.monotonic() - started) > timeout_seconds:
                raise ShotFeatureError("decode timeout exceeded")
            ok, frame = cap.read()
            if not ok:
                break
            frame_index, video_time_us = timeline[decoded]
            bgr = _resize_bgr(frame, width=analysis_width, height=analysis_height)
            import cv2 as _cv2

            gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
            mean_luma = _mean_luma(gray)
            if prev_bgr is None or prev_gray is None:
                luma_mae = 0.0
                hist_distance = 0.0
                edge_change = 0.0
            else:
                luma_mae = _luma_mae(prev_gray, gray)
                hist_distance = _hist_distance(prev_bgr, bgr)
                edge_change = _edge_change_ratio(prev_gray, gray)
            yield FeatureFrame(
                frame_index=frame_index,
                video_time_us=video_time_us,
                luma_mae=luma_mae,
                hist_distance=hist_distance,
                edge_change_ratio=edge_change,
                mean_luma=mean_luma,
            )
            prev_bgr = bgr
            prev_gray = gray
            decoded += 1
    finally:
        cap.release()

    if decoded == 0:
        raise ShotFeatureError("no frames decoded")
    if decoded != len(timeline):
        raise ShotFeatureError(f"decoded frame count {decoded} != timeline length {len(timeline)}")


def extract_feature_frames(
    source: Path | str,
    timeline: Sequence[tuple[int, int]],
    config: Mapping[str, Any],
) -> list[FeatureFrame]:
    """Materialize feature list under resource limits (tiny fixtures only)."""
    max_frames = min(
        int(config["decode"]["max_frames"]), int(config["resource_limits"]["max_frames"])
    )
    timeout = min(
        float(config["decode"]["timeout_seconds"]),
        float(config["resource_limits"]["timeout_seconds"]),
    )
    return list(
        iter_feature_frames(
            source,
            timeline,
            analysis_width=int(config["analysis_width"]),
            analysis_height=int(config["analysis_height"]),
            max_frames=max_frames,
            timeout_seconds=timeout,
        )
    )


__all__ = [
    "FeatureFrame",
    "ShotFeatureError",
    "load_timeline_times",
    "build_cfr_timeline",
    "iter_feature_frames",
    "extract_feature_frames",
]
