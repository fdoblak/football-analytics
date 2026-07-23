"""Per-sample camera-view features (OpenCV); finite-only; seek by frame index."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_analytics.broadcast.camera_sampling import SamplePoint


class CameraFeatureError(ValueError):
    """Camera feature extraction failure."""


@dataclass(frozen=True)
class CameraSampleFeatures:
    frame_index: int
    time_us: int
    pitch_green_fraction: float
    pitch_center_fraction: float
    pitch_spatial_spread: float
    hist_entropy: float
    edge_density: float
    texture_entropy: float
    center_periphery_ratio: float
    overlay_high_contrast_fraction: float
    skin_like_fraction: float
    mean_luma: float
    frame_diff_mean: float
    flow_mag_mean: float
    flow_mag_std: float
    flow_horizontal_ratio: float
    flow_radial_consistency: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "frame_index": self.frame_index,
            "time_us": self.time_us,
            "pitch_green_fraction": self.pitch_green_fraction,
            "pitch_center_fraction": self.pitch_center_fraction,
            "pitch_spatial_spread": self.pitch_spatial_spread,
            "hist_entropy": self.hist_entropy,
            "edge_density": self.edge_density,
            "texture_entropy": self.texture_entropy,
            "center_periphery_ratio": self.center_periphery_ratio,
            "overlay_high_contrast_fraction": self.overlay_high_contrast_fraction,
            "skin_like_fraction": self.skin_like_fraction,
            "mean_luma": self.mean_luma,
            "frame_diff_mean": self.frame_diff_mean,
            "flow_mag_mean": self.flow_mag_mean,
            "flow_mag_std": self.flow_mag_std,
            "flow_horizontal_ratio": self.flow_horizontal_ratio,
            "flow_radial_consistency": self.flow_radial_consistency,
        }


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise CameraFeatureError(f"{name} is non-finite")
    return float(value)


def _resize_bgr(frame: Any, *, width: int, height: int) -> Any:
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        raise CameraFeatureError("empty frame")
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _entropy(gray: Any) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0, 256))
    total = float(hist.sum())
    if total <= 0:
        return _finite("entropy", 0.0)
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    ent = float(-np.sum(p * np.log2(p)))
    return _finite("entropy", ent)


def _edge_density(gray: Any) -> float:
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    dens = float(np.mean(mag)) / 255.0
    return _finite("edge_density", max(0.0, min(1.0, dens)))


def _pitch_mask(hsv: Any, pitch_cfg: Mapping[str, Any]) -> Any:
    import cv2

    lower = np.array(
        [int(pitch_cfg["h_min"]), int(pitch_cfg["s_min"]), int(pitch_cfg["v_min"])],
        dtype=np.uint8,
    )
    upper = np.array([int(pitch_cfg["h_max"]), 255, 255], dtype=np.uint8)
    primary = cv2.inRange(hsv, lower, upper)
    alt_lower = np.array(
        [int(pitch_cfg["alt_h_min"]), int(pitch_cfg["alt_s_min"]), int(pitch_cfg["alt_v_min"])],
        dtype=np.uint8,
    )
    alt_upper = np.array([int(pitch_cfg["alt_h_max"]), 255, 255], dtype=np.uint8)
    alt = cv2.inRange(hsv, alt_lower, alt_upper)
    return cv2.bitwise_or(primary, alt)


def _skin_mask(hsv: Any, skin_cfg: Mapping[str, Any]) -> Any:
    import cv2

    lower = np.array(
        [int(skin_cfg["h_min"]), int(skin_cfg["s_min"]), int(skin_cfg["v_min"])],
        dtype=np.uint8,
    )
    upper = np.array(
        [int(skin_cfg["h_max"]), 255, int(skin_cfg["v_max"])],
        dtype=np.uint8,
    )
    return cv2.inRange(hsv, lower, upper)


def _spatial_spread(mask: Any) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return _finite("pitch_spatial_spread", 0.0)
    h, w = mask.shape[:2]
    nx = xs.astype(np.float64) / max(w - 1, 1)
    ny = ys.astype(np.float64) / max(h - 1, 1)
    spread = float(np.sqrt(np.var(nx) + np.var(ny)))
    return _finite("pitch_spatial_spread", max(0.0, min(1.0, spread * 2.0)))


def _center_fraction(mask: Any) -> float:
    h, w = mask.shape[:2]
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    center = mask[y0:y1, x0:x1]
    if center.size == 0:
        return _finite("pitch_center_fraction", 0.0)
    frac = float(np.mean(center > 0))
    return _finite("pitch_center_fraction", frac)


def _center_periphery_ratio(gray: Any) -> float:
    h, w = gray.shape[:2]
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    center = gray[y0:y1, x0:x1]
    peri_mask = np.ones_like(gray, dtype=bool)
    peri_mask[y0:y1, x0:x1] = False
    peri = gray[peri_mask]
    c_mean = float(np.mean(center)) + 1e-6
    p_mean = float(np.mean(peri)) + 1e-6
    ratio = c_mean / p_mean
    # Map to a bounded signal around 1.0
    bounded = max(0.0, min(3.0, ratio)) / 3.0
    return _finite("center_periphery_ratio", bounded)


def _overlay_fraction(bgr: Any, gray: Any, pitch_mask: Any) -> float:
    """High-contrast / saturated non-pitch fraction (graphics-like)."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    # Local contrast via Laplacian magnitude
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    contrast = np.abs(lap) / 255.0
    high = (contrast > 0.06) | ((sat > 0.45) & (val > 0.45))
    non_pitch = pitch_mask == 0
    overlay = high & non_pitch
    # Bright UI bars (near white) and saturated brand colors (not bare dark bg)
    bright = (val > 0.80) & (sat < 0.35) & non_pitch
    vivid = (sat > 0.50) & (val > 0.35) & non_pitch
    frac = float(np.mean(overlay | bright | vivid))
    return _finite("overlay_high_contrast_fraction", max(0.0, min(1.0, frac)))


def _flow_summary(
    prev_gray: Any,
    gray: Any,
    *,
    flow_cfg: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    import cv2

    if not bool(flow_cfg.get("enabled", True)):
        return (0.0, 0.0, 0.0, 0.0)
    flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
        prev_gray,
        gray,
        None,
        float(flow_cfg["pyr_scale"]),
        int(flow_cfg["levels"]),
        int(flow_cfg["winsize"]),
        int(flow_cfg["iterations"]),
        int(flow_cfg["poly_n"]),
        float(flow_cfg["poly_sigma"]),
        0,
    )
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    mag_mean = _finite("flow_mag_mean", float(np.mean(mag)))
    mag_std = _finite("flow_mag_std", float(np.std(mag)))
    abs_fx = np.abs(fx)
    abs_fy = np.abs(fy)
    denom = abs_fx + abs_fy + 1e-6
    horiz = float(np.mean(abs_fx / denom))
    horiz = _finite("flow_horizontal_ratio", max(0.0, min(1.0, horiz)))

    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rx = xx - cx
    ry = yy - cy
    rnorm = np.sqrt(rx * rx + ry * ry) + 1e-6
    # Radial unit direction
    ux = rx / rnorm
    uy = ry / rnorm
    radial = fx * ux + fy * uy
    # Consistency: mean radial / mean mag
    consistency = float(np.mean(radial)) / (mag_mean + 1e-6)
    consistency = _finite("flow_radial_consistency", max(-1.0, min(1.0, consistency)))
    # Use absolute for zoom detection (in/out)
    consistency_abs = abs(consistency)
    return mag_mean, mag_std, horiz, _finite("flow_radial_consistency", consistency_abs)


def extract_features_for_samples(
    source: Path | str,
    samples: Sequence[SamplePoint],
    config: Mapping[str, Any],
) -> list[CameraSampleFeatures]:
    """Decode only required sample frames (seek by index); compute finite features."""
    import cv2

    path = Path(source)
    if not path.is_file() or path.is_symlink():
        raise CameraFeatureError("source must be a regular non-symlink file")
    if not samples:
        raise CameraFeatureError("samples required")

    width = int(config["analysis_width"])
    height = int(config["analysis_height"])
    max_frames = min(
        int(config["decode"]["max_frames"]), int(config["resource_limits"]["max_frames"])
    )
    timeout = min(
        float(config["decode"]["timeout_seconds"]),
        float(config["resource_limits"]["timeout_seconds"]),
    )
    pitch_cfg = config["pitch_hsv"]
    skin_cfg = config["skin_hsv"]
    flow_cfg = config["optical_flow"]

    ordered = sorted(samples, key=lambda s: (s.frame_index, s.time_us))
    if len(ordered) > max_frames:
        raise CameraFeatureError("sample count exceeds max_frames")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CameraFeatureError("failed to open video source")

    started = time.monotonic()
    prev_gray = None
    out: list[CameraSampleFeatures] = []
    try:
        for sample in ordered:
            if (time.monotonic() - started) > timeout:
                raise CameraFeatureError("decode timeout exceeded")
            # Seek by absolute frame index (deterministic; avoids skip off-by-one).
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(sample.frame_index))
            ok, frame = cap.read()
            if not ok:
                raise CameraFeatureError(f"failed to decode frame_index={sample.frame_index}")
            bgr = _resize_bgr(frame, width=width, height=height)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            pitch = _pitch_mask(hsv, pitch_cfg)
            skin = _skin_mask(hsv, skin_cfg)
            pitch_frac = _finite("pitch_green_fraction", float(np.mean(pitch > 0)))
            skin_frac = _finite("skin_like_fraction", float(np.mean(skin > 0)))
            center_frac = _center_fraction(pitch)
            spread = _spatial_spread(pitch)
            hist_ent = _entropy(gray)
            edge = _edge_density(gray)
            tex = hist_ent  # same estimator; kept as distinct field for provenance
            cpr = _center_periphery_ratio(gray)
            overlay = _overlay_fraction(bgr, gray, pitch)
            mean_luma = _finite("mean_luma", float(np.mean(gray)) / 255.0)

            if prev_gray is None:
                diff = 0.0
                flow_mean = flow_std = flow_h = flow_r = 0.0
            else:
                diff = _finite(
                    "frame_diff_mean",
                    float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                    / 255.0,
                )
                flow_mean, flow_std, flow_h, flow_r = _flow_summary(
                    prev_gray, gray, flow_cfg=flow_cfg
                )

            feats = CameraSampleFeatures(
                frame_index=sample.frame_index,
                time_us=sample.time_us,
                pitch_green_fraction=pitch_frac,
                pitch_center_fraction=center_frac,
                pitch_spatial_spread=spread,
                hist_entropy=hist_ent,
                edge_density=edge,
                texture_entropy=tex,
                center_periphery_ratio=cpr,
                overlay_high_contrast_fraction=overlay,
                skin_like_fraction=skin_frac,
                mean_luma=mean_luma,
                frame_diff_mean=diff,
                flow_mag_mean=flow_mean,
                flow_mag_std=flow_std,
                flow_horizontal_ratio=flow_h,
                flow_radial_consistency=flow_r,
            )
            out.append(feats)
            prev_gray = gray
    finally:
        cap.release()

    if len(out) != len(ordered):
        raise CameraFeatureError("feature count mismatch")
    return out


__all__ = [
    "CameraFeatureError",
    "CameraSampleFeatures",
    "extract_features_for_samples",
]
