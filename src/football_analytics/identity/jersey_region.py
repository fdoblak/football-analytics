"""Jersey torso region candidates + quality metrics (Stage 7D)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class JerseyRegionCandidate:
    x0: int
    y0: int
    x1: int
    y1: int
    quality: float
    contrast: float
    blur_var: float
    area: int
    suitability: str
    rank_key: tuple[Any, ...]
    reason_codes: tuple[str, ...]


def _clamp_box(
    x0: int, y0: int, x1: int, y1: int, *, width: int, height: int
) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def _contrast_blur(gray: np.ndarray) -> tuple[float, float]:
    if gray.size == 0:
        return 0.0, 0.0
    contrast = float(np.std(gray.astype(np.float32)))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return contrast, blur


def propose_torso_regions(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    config: Mapping[str, Any],
) -> list[JerseyRegionCandidate]:
    """Propose bounded torso jersey-number regions inside a human bbox.

    Deterministic ranking: higher quality first, then larger area, then (y0,x0).
    Does not claim back orientation without a pose model.
    """
    if image.ndim == 3:
        h, w = image.shape[:2]
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        h, w = image.shape[:2]
        gray_full = image
    bx0, by0, bx1, by1 = [float(v) for v in bbox]
    bx0_i, by0_i, bx1_i, by1_i = _clamp_box(
        int(round(bx0)), int(round(by0)), int(round(bx1)), int(round(by1)), width=w, height=h
    )
    bw = bx1_i - bx0_i
    bh = by1_i - by0_i
    region_cfg = config["region"]
    elig = config["eligibility"]
    if (
        bw < int(elig["min_bbox_width"])
        or bh < int(elig["min_bbox_height"])
        or bw * bh < int(elig["min_bbox_area"])
    ):
        return []

    # Primary torso band + slightly shifted variants for ranking.
    fracs = [
        (
            float(region_cfg["torso_x0_frac"]),
            float(region_cfg["torso_y0_frac"]),
            float(region_cfg["torso_x1_frac"]),
            float(region_cfg["torso_y1_frac"]),
        ),
        (
            max(0.0, float(region_cfg["torso_x0_frac"]) - 0.04),
            max(0.0, float(region_cfg["torso_y0_frac"]) - 0.04),
            min(1.0, float(region_cfg["torso_x1_frac"]) + 0.04),
            min(1.0, float(region_cfg["torso_y1_frac"]) + 0.04),
        ),
        (
            float(region_cfg["torso_x0_frac"]) + 0.02,
            float(region_cfg["torso_y0_frac"]) + 0.06,
            float(region_cfg["torso_x1_frac"]) - 0.02,
            min(1.0, float(region_cfg["torso_y1_frac"]) + 0.06),
        ),
    ]
    candidates: list[JerseyRegionCandidate] = []
    for fx0, fy0, fx1, fy1 in fracs:
        if not (0.0 <= fx0 < fx1 <= 1.0 and 0.0 <= fy0 < fy1 <= 1.0):
            continue
        x0 = bx0_i + int(round(fx0 * bw))
        y0 = by0_i + int(round(fy0 * bh))
        x1 = bx0_i + int(round(fx1 * bw))
        y1 = by0_i + int(round(fy1 * bh))
        x0, y0, x1, y1 = _clamp_box(x0, y0, x1, y1, width=w, height=h)
        # Containment: must stay inside human bbox.
        if x0 < bx0_i or y0 < by0_i or x1 > bx1_i or y1 > by1_i:
            continue
        rw, rh = x1 - x0, y1 - y0
        area = rw * rh
        if (
            rw < int(region_cfg["min_region_width"])
            or rh < int(region_cfg["min_region_height"])
            or area < int(region_cfg["min_region_area"])
        ):
            continue
        patch = gray_full[y0:y1, x0:x1]
        contrast, blur = _contrast_blur(patch)
        reasons: list[str] = []
        suitability = str(region_cfg["orientation_claim"])
        if contrast < float(region_cfg["min_contrast"]):
            reasons.append("LOW_CONTRAST")
            suitability = "not_suitable"
        if blur < float(region_cfg["min_blur_laplacian_var"]):
            reasons.append("BLURRY")
            suitability = "not_suitable"
        # Quality score: prefer moderate blur variance + contrast (deterministic).
        quality = float(max(0.0, min(1.0, (contrast / 64.0) * 0.6 + min(1.0, blur / 200.0) * 0.4)))
        rank = (-quality, -area, y0, x0, y1, x1)
        candidates.append(
            JerseyRegionCandidate(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                quality=quality,
                contrast=contrast,
                blur_var=blur,
                area=area,
                suitability=suitability,
                rank_key=rank,
                reason_codes=tuple(reasons),
            )
        )
    candidates.sort(key=lambda c: c.rank_key)
    return candidates


def extract_region_crop(image: np.ndarray, candidate: JerseyRegionCandidate) -> np.ndarray:
    crop = image[candidate.y0 : candidate.y1, candidate.x0 : candidate.x1]
    return np.ascontiguousarray(crop.copy())


def region_metrics_payload(candidate: JerseyRegionCandidate) -> dict[str, Any]:
    return {
        "bbox": [candidate.x0, candidate.y0, candidate.x1, candidate.y1],
        "quality": candidate.quality,
        "contrast": candidate.contrast,
        "blur_var": candidate.blur_var,
        "area": candidate.area,
        "suitability": candidate.suitability,
        "reason_codes": list(candidate.reason_codes),
    }


__all__ = [
    "JerseyRegionCandidate",
    "propose_torso_regions",
    "extract_region_crop",
    "region_metrics_payload",
]
