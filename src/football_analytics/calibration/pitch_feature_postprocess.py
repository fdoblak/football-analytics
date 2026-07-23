"""Heatmap peak NMS + line endpoint / component fitting (Stage 8B; no GPL copy).

Peak decode mirrors CenterNet/mmdet-style max-pool local maxima with scale=2
into 960×540 model space. Line SV models emit up to two peaks per channel
(endpoints). A connected-component + fitLine path supports synthetic masks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.calibration.pitch_feature_mapping import keypoint_mapping, line_mapping
from football_analytics.calibration.pitch_feature_preprocess import (
    StretchTransform,
    clip_point_to_source,
    model_point_to_source,
)


class PitchFeaturePostprocessError(ValueError):
    """Postprocess failure."""


@dataclass(frozen=True)
class DecodedKeypoint:
    channel_index: int
    x_model: float
    y_model: float
    score: float
    x_source: float | None
    y_source: float | None
    in_bounds: bool
    rejected: bool
    reason: str | None


@dataclass(frozen=True)
class DecodedLine:
    channel_index: int
    x1_model: float
    y1_model: float
    x2_model: float
    y2_model: float
    score: float
    x1_source: float | None
    y1_source: float | None
    x2_source: float | None
    y2_source: float | None
    length_source: float | None
    in_bounds: bool
    rejected: bool
    reason: str | None


def _finite(*vals: float) -> bool:
    return all(math.isfinite(float(v)) for v in vals)


def extract_peaks_maxpool(
    heatmap: Any,
    *,
    scale: int = 2,
    max_peaks: int = 1,
    min_distance: int = 15,
    border_pad_value: float | None = 1.0,
) -> list[list[list[tuple[float, float, float]]]]:
    """Return [batch][channel][(x,y,score), ...] in model pixel space (×scale).

    ``heatmap``: torch.Tensor (B,C,H,W). Does not mutate caller tensor.
    """
    import torch
    import torch.nn.functional as F

    if not isinstance(heatmap, torch.Tensor):
        raise PitchFeaturePostprocessError("heatmap must be a torch.Tensor")
    if heatmap.ndim != 4:
        raise PitchFeaturePostprocessError("heatmap must be BxCxHxW")
    if torch.isnan(heatmap).any() or torch.isinf(heatmap).any():
        raise PitchFeaturePostprocessError("NaN/Inf in heatmap")

    batch_size, n_channels, _, width = heatmap.shape
    work = heatmap
    kernel = int(min_distance) * 2 + 1
    if border_pad_value is not None:
        pad = int(min_distance)
        padded = F.pad(work, (pad, pad, pad, pad), mode="constant", value=float(border_pad_value))
        pooled = F.max_pool2d(padded, kernel, stride=1, padding=0)
    else:
        pad = (kernel - 1) // 2
        pooled = F.max_pool2d(work, kernel, stride=1, padding=pad)
    local_maxima = pooled == work
    suppressed = work * local_maxima
    scores, indices = torch.topk(
        suppressed.reshape(batch_size, n_channels, -1), int(max_peaks), sorted=True
    )
    rows = torch.div(indices, width, rounding_mode="floor")
    cols = indices % width
    # (u,v) = (col,row) * scale
    out: list[list[list[tuple[float, float, float]]]] = []
    scores_np = scores.detach().cpu().tolist()
    rows_np = rows.detach().cpu().tolist()
    cols_np = cols.detach().cpu().tolist()
    for b in range(batch_size):
        chans: list[list[tuple[float, float, float]]] = []
        for c in range(n_channels):
            peaks: list[tuple[float, float, float]] = []
            for k in range(int(max_peaks)):
                sc = float(scores_np[b][c][k])
                x = float(cols_np[b][c][k]) * float(scale)
                y = float(rows_np[b][c][k]) * float(scale)
                peaks.append((x, y, sc))
            chans.append(peaks)
        out.append(chans)
    return out


def decode_keypoints_from_heatmap(
    heatmap: Any,
    *,
    transform: StretchTransform,
    score_threshold: float,
    scale: int = 2,
    max_peaks: int = 1,
    min_distance: int = 15,
    expected_channels: int = 57,
    duplicate_distance_px: float = 6.0,
) -> list[DecodedKeypoint]:
    if heatmap.shape[1] != expected_channels:
        raise PitchFeaturePostprocessError(
            f"keypoint channel mismatch: got {heatmap.shape[1]} expected {expected_channels}"
        )
    peaks = extract_peaks_maxpool(
        heatmap,
        scale=scale,
        max_peaks=max_peaks,
        min_distance=min_distance,
        border_pad_value=1.0,
    )[0]
    decoded: list[DecodedKeypoint] = []
    kept_xy: list[tuple[float, float]] = []
    for ch, ch_peaks in enumerate(peaks):
        for x_m, y_m, score in ch_peaks:
            if not _finite(x_m, y_m, score):
                decoded.append(
                    DecodedKeypoint(ch, x_m, y_m, score, None, None, False, True, "non_finite")
                )
                continue
            if score <= float(score_threshold):
                decoded.append(
                    DecodedKeypoint(ch, x_m, y_m, score, None, None, False, True, "below_threshold")
                )
                continue
            xs, ys = model_point_to_source(x_m, y_m, transform)
            clipped = clip_point_to_source(xs, ys, transform)
            if clipped is None:
                decoded.append(
                    DecodedKeypoint(ch, x_m, y_m, score, xs, ys, False, True, "out_of_bounds")
                )
                continue
            xs, ys = clipped
            dup = False
            for px, py in kept_xy:
                if math.hypot(xs - px, ys - py) < float(duplicate_distance_px):
                    dup = True
                    break
            if dup:
                decoded.append(
                    DecodedKeypoint(ch, x_m, y_m, score, xs, ys, True, True, "duplicate")
                )
                continue
            kept_xy.append((xs, ys))
            decoded.append(DecodedKeypoint(ch, x_m, y_m, score, xs, ys, True, False, None))
    return decoded


def decode_lines_from_heatmap(
    heatmap: Any,
    *,
    transform: StretchTransform,
    score_threshold: float,
    scale: int = 2,
    max_peaks: int = 2,
    min_distance: int = 10,
    expected_channels: int = 23,
    minimum_length_px: float = 8.0,
    duplicate_endpoint_distance_px: float = 8.0,
) -> list[DecodedLine]:
    if heatmap.shape[1] != expected_channels:
        raise PitchFeaturePostprocessError(
            f"line channel mismatch: got {heatmap.shape[1]} expected {expected_channels}"
        )
    peaks = extract_peaks_maxpool(
        heatmap,
        scale=scale,
        max_peaks=max_peaks,
        min_distance=min_distance,
        border_pad_value=None,
    )[0]
    decoded: list[DecodedLine] = []
    kept: list[tuple[float, float, float, float]] = []
    for ch, ch_peaks in enumerate(peaks):
        if len(ch_peaks) < 2:
            continue
        (x1m, y1m, s1), (x2m, y2m, s2) = ch_peaks[0], ch_peaks[1]
        score = min(float(s1), float(s2))
        if not _finite(x1m, y1m, x2m, y2m, s1, s2):
            decoded.append(
                DecodedLine(
                    ch,
                    x1m,
                    y1m,
                    x2m,
                    y2m,
                    score,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                    True,
                    "non_finite",
                )
            )
            continue
        if float(s1) <= score_threshold or float(s2) <= score_threshold:
            decoded.append(
                DecodedLine(
                    ch,
                    x1m,
                    y1m,
                    x2m,
                    y2m,
                    score,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                    True,
                    "below_threshold",
                )
            )
            continue
        x1s, y1s = model_point_to_source(x1m, y1m, transform)
        x2s, y2s = model_point_to_source(x2m, y2m, transform)
        c1 = clip_point_to_source(x1s, y1s, transform)
        c2 = clip_point_to_source(x2s, y2s, transform)
        if c1 is None or c2 is None:
            decoded.append(
                DecodedLine(
                    ch,
                    x1m,
                    y1m,
                    x2m,
                    y2m,
                    score,
                    x1s,
                    y1s,
                    x2s,
                    y2s,
                    None,
                    False,
                    True,
                    "out_of_bounds",
                )
            )
            continue
        x1s, y1s = c1
        x2s, y2s = c2
        length = math.hypot(x2s - x1s, y2s - y1s)
        if length < float(minimum_length_px):
            decoded.append(
                DecodedLine(
                    ch,
                    x1m,
                    y1m,
                    x2m,
                    y2m,
                    score,
                    x1s,
                    y1s,
                    x2s,
                    y2s,
                    length,
                    True,
                    True,
                    "too_short",
                )
            )
            continue
        dup = False
        for a, b, c, d in kept:
            if (
                math.hypot(x1s - a, y1s - b) < duplicate_endpoint_distance_px
                and math.hypot(x2s - c, y2s - d) < duplicate_endpoint_distance_px
            ) or (
                math.hypot(x1s - c, y1s - d) < duplicate_endpoint_distance_px
                and math.hypot(x2s - a, y2s - b) < duplicate_endpoint_distance_px
            ):
                dup = True
                break
        if dup:
            decoded.append(
                DecodedLine(
                    ch,
                    x1m,
                    y1m,
                    x2m,
                    y2m,
                    score,
                    x1s,
                    y1s,
                    x2s,
                    y2s,
                    length,
                    True,
                    True,
                    "duplicate",
                )
            )
            continue
        kept.append((x1s, y1s, x2s, y2s))
        decoded.append(
            DecodedLine(
                ch,
                x1m,
                y1m,
                x2m,
                y2m,
                score,
                x1s,
                y1s,
                x2s,
                y2s,
                length,
                True,
                False,
                None,
            )
        )
    return decoded


def fit_line_from_mask(
    mask: Any,
    *,
    minimum_length_px: float = 8.0,
    min_component_area: int = 8,
) -> tuple[float, float, float, float, float] | None:
    """Connected components on binary mask → longest component fitLine endpoints.

    Returns (x1,y1,x2,y2,support_score) in the mask's pixel space, or None.
    """
    import cv2
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise PitchFeaturePostprocessError("mask must be 2D")
    binary = (arr > 0).astype("uint8")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best: tuple[float, float, float, float, float] | None = None
    best_len = -1.0
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        ys, xs = (labels == label).nonzero()
        if len(xs) < 2:
            continue
        pts = np.column_stack([xs.astype("float32"), ys.astype("float32")])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # Project extents along direction
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        t0, t1 = float(t.min()), float(t.max())
        x1, y1 = float(x0 + t0 * vx), float(y0 + t0 * vy)
        x2, y2 = float(x0 + t1 * vx), float(y0 + t1 * vy)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < float(minimum_length_px):
            continue
        support = min(1.0, area / 1000.0)
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2, support)
    return best


def make_synthetic_peak_heatmap(
    *,
    channels: int,
    height: int,
    width: int,
    peaks: Sequence[tuple[int, int, int, float]],
) -> Any:
    """Build (1,C,H,W) tensor with gaussian-ish peaks: (channel, row, col, score)."""
    import torch

    heat = torch.zeros((1, channels, height, width), dtype=torch.float32)
    for ch, row, col, score in peaks:
        if not (0 <= ch < channels and 0 <= row < height and 0 <= col < width):
            raise PitchFeaturePostprocessError("synthetic peak out of range")
        heat[0, ch, row, col] = float(score)
        # tiny neighbors for NMS uniqueness
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width:
                heat[0, ch, rr, cc] = max(float(heat[0, ch, rr, cc]), float(score) * 0.5)
    return heat


def accepted_keypoints(decoded: Sequence[DecodedKeypoint]) -> list[DecodedKeypoint]:
    return [d for d in decoded if not d.rejected and d.in_bounds]


def accepted_lines(decoded: Sequence[DecodedLine]) -> list[DecodedLine]:
    return [d for d in decoded if not d.rejected and d.in_bounds]


def mapping_for_keypoint(channel_index: int) -> Any:
    return keypoint_mapping(channel_index)


def mapping_for_line(channel_index: int) -> Any:
    return line_mapping(channel_index)


__all__ = [
    "PitchFeaturePostprocessError",
    "DecodedKeypoint",
    "DecodedLine",
    "extract_peaks_maxpool",
    "decode_keypoints_from_heatmap",
    "decode_lines_from_heatmap",
    "fit_line_from_mask",
    "make_synthetic_peak_heatmap",
    "accepted_keypoints",
    "accepted_lines",
    "mapping_for_keypoint",
    "mapping_for_line",
]
