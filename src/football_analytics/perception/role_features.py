"""Deterministic crop / synthetic role features (Stage 5D, OpenCV/NumPy)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class RoleFeatureError(ValueError):
    """Role feature extraction failure."""


@dataclass(frozen=True)
class RoleFeatures:
    detection_id: int
    frame_index: int
    upper_hist: tuple[float, ...]
    lower_hist: tuple[float, ...]
    mean_saturation: float
    mean_value: float
    color_signature: tuple[float, ...]
    bbox_area: float
    aspect_ratio: float
    norm_cx: float
    norm_cy: float
    crop_quality: float
    feature_source: str  # crop | synthetic | geometry_only

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "upper_hist": list(self.upper_hist),
            "lower_hist": list(self.lower_hist),
            "mean_saturation": self.mean_saturation,
            "mean_value": self.mean_value,
            "color_signature": list(self.color_signature),
            "bbox_area": self.bbox_area,
            "aspect_ratio": self.aspect_ratio,
            "norm_cx": self.norm_cx,
            "norm_cy": self.norm_cy,
            "crop_quality": self.crop_quality,
            "feature_source": self.feature_source,
        }


def _finite_unit(value: float, *, label: str) -> float:
    if not math.isfinite(value):
        raise RoleFeatureError(f"{label} must be finite")
    if value < 0.0 or value > 1.0:
        raise RoleFeatureError(f"{label} must be in [0,1]")
    return float(value)


def _l1_normalize(hist: np.ndarray) -> np.ndarray:
    s = float(np.sum(hist))
    if not math.isfinite(s) or s <= 0.0:
        out = np.zeros_like(hist, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    return (hist.astype(np.float64) / s).astype(np.float64)


def _hsv_hist(
    hsv: np.ndarray, *, h_bins: int, s_bins: int, v_bins: int
) -> tuple[np.ndarray, float, float]:
    import cv2

    h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).reshape(-1)
    s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).reshape(-1)
    v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).reshape(-1)
    hist = np.concatenate([_l1_normalize(h), _l1_normalize(s), _l1_normalize(v)])
    mean_s = float(np.mean(hsv[:, :, 1])) / 255.0
    mean_v = float(np.mean(hsv[:, :, 2])) / 255.0
    return hist, mean_s, mean_v


def _bbox_geometry(
    bbox_xyxy: Sequence[float],
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, frame_width, frame_height)):
        raise RoleFeatureError("bbox/frame geometry must be finite")
    if x2 <= x1 or y2 <= y1:
        raise RoleFeatureError("invalid bbox")
    w = x2 - x1
    h = y2 - y1
    area = w * h
    aspect = w / h if h > 0 else 0.0
    cx = ((x1 + x2) * 0.5) / max(frame_width, 1.0)
    cy = ((y1 + y2) * 0.5) / max(frame_height, 1.0)
    return (
        area,
        aspect,
        _finite_unit(min(1.0, max(0.0, cx)), label="norm_cx"),
        _finite_unit(min(1.0, max(0.0, cy)), label="norm_cy"),
        min(w, h),
    )


def crop_quality_from_array(bgr: np.ndarray) -> float:
    """Simple coverage/sharpness proxy in [0,1]."""
    if bgr is None or bgr.size == 0:
        return 0.0
    h, w = bgr.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = min(1.0, lap / 200.0)
    area_proxy = min(1.0, (h * w) / 10000.0)
    q = 0.5 * sharp + 0.5 * area_proxy
    if not math.isfinite(q):
        raise RoleFeatureError("crop_quality must be finite")
    return float(min(1.0, max(0.0, q)))


def extract_features_from_crop(
    bgr_crop: np.ndarray,
    *,
    detection_id: int,
    frame_index: int,
    bbox_xyxy: Sequence[float],
    frame_width: float,
    frame_height: float,
    config: Mapping[str, Any],
) -> RoleFeatures:
    """Extract HSV upper/lower histograms from a BGR crop (not persisted)."""
    import cv2

    if bgr_crop is None or not isinstance(bgr_crop, np.ndarray) or bgr_crop.size == 0:
        raise RoleFeatureError("empty crop")
    feats = config["features"]
    h_bins = int(feats["hsv_h_bins"])
    s_bins = int(feats["hsv_s_bins"])
    v_bins = int(feats["hsv_v_bins"])
    upper_frac = float(feats["upper_body_fraction"])

    area, aspect, norm_cx, norm_cy, _ = _bbox_geometry(
        bbox_xyxy, frame_width=frame_width, frame_height=frame_height
    )
    ch, cw = bgr_crop.shape[:2]
    split = max(1, min(ch - 1, int(round(ch * upper_frac))))
    upper = bgr_crop[:split, :, :]
    lower = bgr_crop[split:, :, :]
    hsv_u = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    hsv_l = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    uh, ms_u, mv_u = _hsv_hist(hsv_u, h_bins=h_bins, s_bins=s_bins, v_bins=v_bins)
    lh, ms_l, mv_l = _hsv_hist(hsv_l, h_bins=h_bins, s_bins=s_bins, v_bins=v_bins)
    mean_s = _finite_unit(0.5 * (ms_u + ms_l), label="mean_saturation")
    mean_v = _finite_unit(0.5 * (mv_u + mv_l), label="mean_value")
    # Compact signature: upper hue peak + mean S/V + lower hue peak.
    u_h = uh[:h_bins]
    l_h = lh[:h_bins]
    sig = (
        float(np.argmax(u_h)) / max(h_bins - 1, 1),
        mean_s,
        mean_v,
        float(np.argmax(l_h)) / max(h_bins - 1, 1),
    )
    quality = crop_quality_from_array(bgr_crop)
    return RoleFeatures(
        detection_id=int(detection_id),
        frame_index=int(frame_index),
        upper_hist=tuple(float(x) for x in uh.tolist()),
        lower_hist=tuple(float(x) for x in lh.tolist()),
        mean_saturation=mean_s,
        mean_value=mean_v,
        color_signature=sig,
        bbox_area=float(area),
        aspect_ratio=float(aspect),
        norm_cx=norm_cx,
        norm_cy=norm_cy,
        crop_quality=quality,
        feature_source="crop",
    )


def extract_synthetic_features(
    *,
    detection_id: int,
    frame_index: int,
    bbox_xyxy: Sequence[float],
    frame_width: float,
    frame_height: float,
    config: Mapping[str, Any],
    kit_hue: float = 0.0,
    kit_saturation: float = 0.6,
    kit_value: float = 0.6,
    lower_hue: float | None = None,
    crop_quality: float = 0.8,
) -> RoleFeatures:
    """Fixture-only path: geometry + synthetic kit color (no video/crops)."""
    feats = config["features"]
    h_bins = int(feats["hsv_h_bins"])
    s_bins = int(feats["hsv_s_bins"])
    v_bins = int(feats["hsv_v_bins"])
    area, aspect, norm_cx, norm_cy, _ = _bbox_geometry(
        bbox_xyxy, frame_width=frame_width, frame_height=frame_height
    )
    hue = _finite_unit(float(kit_hue), label="kit_hue")
    sat = _finite_unit(float(kit_saturation), label="kit_saturation")
    val = _finite_unit(float(kit_value), label="kit_value")
    lh_peak = _finite_unit(
        float(lower_hue if lower_hue is not None else kit_hue), label="lower_hue"
    )
    quality = _finite_unit(float(crop_quality), label="crop_quality")

    def _peak_hist(peak01: float, n: int) -> list[float]:
        hist = [0.0] * n
        idx = int(round(peak01 * (n - 1)))
        idx = min(max(idx, 0), n - 1)
        hist[idx] = 1.0
        return hist

    upper = _peak_hist(hue, h_bins) + _peak_hist(sat, s_bins) + _peak_hist(val, v_bins)
    lower = _peak_hist(lh_peak, h_bins) + _peak_hist(sat, s_bins) + _peak_hist(val, v_bins)
    sig = (hue, sat, val, lh_peak)
    return RoleFeatures(
        detection_id=int(detection_id),
        frame_index=int(frame_index),
        upper_hist=tuple(upper),
        lower_hist=tuple(lower),
        mean_saturation=sat,
        mean_value=val,
        color_signature=sig,
        bbox_area=float(area),
        aspect_ratio=float(aspect),
        norm_cx=norm_cx,
        norm_cy=norm_cy,
        crop_quality=quality,
        feature_source="synthetic",
    )


def color_l1_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise RoleFeatureError("signature length mismatch")
    dist = sum(abs(float(x) - float(y)) for x, y in zip(a, b, strict=True)) / max(len(a), 1)
    if not math.isfinite(dist):
        raise RoleFeatureError("color distance must be finite")
    return float(dist)


def kit_color_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Hue-weighted kit distance so distinct jerseys do not collapse under mean L1."""
    if len(a) < 4 or len(b) < 4:
        return color_l1_distance(a, b)
    dist = (
        0.40 * abs(float(a[0]) - float(b[0]))
        + 0.10 * abs(float(a[1]) - float(b[1]))
        + 0.10 * abs(float(a[2]) - float(b[2]))
        + 0.40 * abs(float(a[3]) - float(b[3]))
    )
    if not math.isfinite(dist):
        raise RoleFeatureError("kit color distance must be finite")
    return float(dist)


def map_user_other_to_staff(label: str) -> str:
    """Documented mapping: user-facing 'other' → canonical RoleLabel staff."""
    if label == "other":
        return "staff"
    return label


__all__ = [
    "RoleFeatureError",
    "RoleFeatures",
    "extract_features_from_crop",
    "extract_synthetic_features",
    "crop_quality_from_array",
    "color_l1_distance",
    "map_user_other_to_staff",
    "kit_color_distance",
]
