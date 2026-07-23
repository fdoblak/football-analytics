"""Handcrafted appearance descriptor (Stage 7B) — HSV/Lab spatial + edge/texture."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

EXTRACTOR_TYPE_HANDCRAFTED = "handcrafted"
DEFAULT_EMBEDDING_DIM = 88
L2_NORM_TOL = 1e-3
EPS = 1e-12


class AppearanceDescriptorError(ValueError):
    """Appearance descriptor failure."""


@dataclass(frozen=True)
class AppearanceDescriptor:
    vector: tuple[float, ...]
    quality: float
    dimension: int
    extractor_id: str
    extractor_version: str
    extractor_type: str
    reason_codes: tuple[str, ...]

    def as_list(self) -> list[float]:
        return list(self.vector)


def expected_embedding_dim(descriptor_cfg: Mapping[str, Any]) -> int:
    dim = int(descriptor_cfg["embedding_dim"])
    computed = (
        2
        * (
            int(descriptor_cfg["hsv_h_bins"])
            + int(descriptor_cfg["hsv_s_bins"])
            + int(descriptor_cfg["hsv_v_bins"])
            + int(descriptor_cfg["lab_l_bins"])
            + int(descriptor_cfg["lab_a_bins"])
            + int(descriptor_cfg["lab_b_bins"])
        )
        + int(descriptor_cfg["edge_bins"])
        + int(descriptor_cfg["texture_bins"])
    )
    if dim != computed:
        raise AppearanceDescriptorError(
            f"embedding_dim {dim} != computed descriptor size {computed}"
        )
    return dim


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise AppearanceDescriptorError("empty embedding")
    if not np.all(np.isfinite(arr)):
        raise AppearanceDescriptorError("embedding must be finite")
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n < EPS:
        out = np.zeros_like(arr)
        out[0] = 1.0
        return out
    return (arr / n).astype(np.float64)


def validate_embedding(vec: Sequence[float], *, expected_dim: int) -> tuple[float, ...]:
    if len(vec) != expected_dim:
        raise AppearanceDescriptorError(f"embedding dim {len(vec)} != expected {expected_dim}")
    arr = np.asarray(list(vec), dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise AppearanceDescriptorError("embedding contains non-finite values")
    n = float(np.linalg.norm(arr))
    if abs(n - 1.0) > L2_NORM_TOL:
        raise AppearanceDescriptorError(f"embedding L2 norm {n} outside tolerance")
    return tuple(float(x) for x in arr.tolist())


def _l1_normalize(hist: np.ndarray) -> np.ndarray:
    s = float(np.sum(hist))
    if not math.isfinite(s) or s <= 0.0:
        out = np.zeros_like(hist, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    return (hist.astype(np.float64) / s).astype(np.float64)


def _channel_hist(img: np.ndarray, channel: int, bins: int, ranges: list[float]) -> np.ndarray:
    import cv2

    h = cv2.calcHist([img], [channel], None, [bins], ranges).reshape(-1)
    return _l1_normalize(h)


def _hsv_block(bgr: np.ndarray, *, h_bins: int, s_bins: int, v_bins: int) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return np.concatenate(
        [
            _channel_hist(hsv, 0, h_bins, [0, 180]),
            _channel_hist(hsv, 1, s_bins, [0, 256]),
            _channel_hist(hsv, 2, v_bins, [0, 256]),
        ]
    )


def _lab_block(bgr: np.ndarray, *, l_bins: int, a_bins: int, b_bins: int) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return np.concatenate(
        [
            _channel_hist(lab, 0, l_bins, [0, 256]),
            _channel_hist(lab, 1, a_bins, [0, 256]),
            _channel_hist(lab, 2, b_bins, [0, 256]),
        ]
    )


def _edge_texture(
    bgr: np.ndarray, *, edge_bins: int, texture_bins: int
) -> tuple[np.ndarray, float]:
    """Bounded grayscale edge magnitude hist + texture energy bands (no face cues)."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Clip extreme edges for stability.
    mag = np.clip(mag, 0.0, 255.0)
    edge_hist, _ = np.histogram(mag.reshape(-1), bins=edge_bins, range=(0.0, 255.0))
    edge = _l1_normalize(edge_hist.astype(np.float64))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    abs_lap = np.abs(lap)
    # Spatial texture bands: mean energy over horizontal strips.
    h = abs_lap.shape[0]
    bands = []
    for i in range(texture_bins):
        y0 = int(round(i * h / texture_bins))
        y1 = int(round((i + 1) * h / texture_bins))
        y1 = max(y1, y0 + 1)
        band = abs_lap[y0:y1, :]
        bands.append(float(np.mean(band)) if band.size else 0.0)
    tex = _l1_normalize(np.asarray(bands, dtype=np.float64))
    sharp = float(min(1.0, float(np.var(np.asarray(lap, dtype=np.float64))) / 200.0))
    return np.concatenate([edge, tex]), sharp


def crop_quality(bgr: np.ndarray) -> float:
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
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
        raise AppearanceDescriptorError("crop_quality must be finite")
    return float(min(1.0, max(0.0, q)))


def extract_descriptor_from_bgr(
    bgr_crop: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> AppearanceDescriptor:
    """Extract fixed-dim L2-normalized handcrafted appearance vector from BGR crop.

    Does not encode track id, timestamp, or evaluation labels.
    """
    if bgr_crop is None or not isinstance(bgr_crop, np.ndarray) or bgr_crop.size == 0:
        raise AppearanceDescriptorError("empty crop")
    desc = config["descriptor"]
    dim = expected_embedding_dim(desc)
    h_bins = int(desc["hsv_h_bins"])
    s_bins = int(desc["hsv_s_bins"])
    v_bins = int(desc["hsv_v_bins"])
    l_bins = int(desc["lab_l_bins"])
    a_bins = int(desc["lab_a_bins"])
    b_bins = int(desc["lab_b_bins"])
    edge_bins = int(desc["edge_bins"])
    texture_bins = int(desc["texture_bins"])
    upper_frac = float(desc["upper_body_fraction"])

    ch, cw = bgr_crop.shape[:2]
    if ch < 2 or cw < 2:
        raise AppearanceDescriptorError("crop too small")
    split = max(1, min(ch - 1, int(round(ch * upper_frac))))
    upper = bgr_crop[:split, :, :]
    lower = bgr_crop[split:, :, :]

    parts = [
        _hsv_block(upper, h_bins=h_bins, s_bins=s_bins, v_bins=v_bins),
        _hsv_block(lower, h_bins=h_bins, s_bins=s_bins, v_bins=v_bins),
        _lab_block(upper, l_bins=l_bins, a_bins=a_bins, b_bins=b_bins),
        _lab_block(lower, l_bins=l_bins, a_bins=a_bins, b_bins=b_bins),
    ]
    edge_tex, _ = _edge_texture(bgr_crop, edge_bins=edge_bins, texture_bins=texture_bins)
    parts.append(edge_tex)
    raw = np.concatenate(parts)
    if raw.size != dim:
        raise AppearanceDescriptorError(f"raw size {raw.size} != dim {dim}")
    vec = l2_normalize(raw)
    validated = validate_embedding(vec.tolist(), expected_dim=dim)
    quality = crop_quality(bgr_crop)
    return AppearanceDescriptor(
        vector=validated,
        quality=quality,
        dimension=dim,
        extractor_id=str(config["extractor_id"]),
        extractor_version=str(config["extractor_version"]),
        extractor_type=str(config["extractor_type"]),
        reason_codes=(),
    )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise AppearanceDescriptorError("cosine dim mismatch")
    va = np.asarray(list(a), dtype=np.float64)
    vb = np.asarray(list(b), dtype=np.float64)
    if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
        raise AppearanceDescriptorError("cosine inputs must be finite")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < EPS or nb < EPS:
        return 0.0
    sim = float(np.dot(va, vb) / (na * nb))
    if not math.isfinite(sim):
        raise AppearanceDescriptorError("cosine similarity must be finite")
    return float(max(-1.0, min(1.0, sim)))


__all__ = [
    "EXTRACTOR_TYPE_HANDCRAFTED",
    "DEFAULT_EMBEDDING_DIM",
    "AppearanceDescriptorError",
    "AppearanceDescriptor",
    "expected_embedding_dim",
    "l2_normalize",
    "validate_embedding",
    "crop_quality",
    "extract_descriptor_from_bgr",
    "cosine_similarity",
]
