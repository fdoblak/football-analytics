"""Stretch-resize preprocess for Stage 8B SV_kp / SV_lines (960×540, no mean/std)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from football_analytics.core.hashing import hash_canonical_json

MODEL_WIDTH = 960
MODEL_HEIGHT = 540


class PitchFeaturePreprocessError(ValueError):
    """Preprocess / inverse transform failure."""


@dataclass(frozen=True)
class StretchTransform:
    """Source → model stretch (independent sx/sy; no letterbox padding)."""

    source_width: int
    source_height: int
    model_width: int = MODEL_WIDTH
    model_height: int = MODEL_HEIGHT
    resize_mode: str = "stretch"
    color_order: str = "rgb"
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None
    tensor_dtype: str = "float32"
    to_tensor_lo: float = 0.0
    to_tensor_hi: float = 1.0
    pad_left: float = 0.0
    pad_top: float = 0.0
    pad_right: float = 0.0
    pad_bottom: float = 0.0

    @property
    def scale_x(self) -> float:
        return float(self.model_width) / float(self.source_width)

    @property
    def scale_y(self) -> float:
        return float(self.model_height) / float(self.source_height)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scale_x"] = self.scale_x
        d["scale_y"] = self.scale_y
        return d

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


def build_stretch_transform(
    *,
    source_width: int,
    source_height: int,
    model_width: int = MODEL_WIDTH,
    model_height: int = MODEL_HEIGHT,
) -> StretchTransform:
    if source_width <= 0 or source_height <= 0:
        raise PitchFeaturePreprocessError("source dimensions must be positive")
    if model_width <= 0 or model_height <= 0:
        raise PitchFeaturePreprocessError("model dimensions must be positive")
    return StretchTransform(
        source_width=int(source_width),
        source_height=int(source_height),
        model_width=int(model_width),
        model_height=int(model_height),
    )


def model_point_to_source(
    x_model: float, y_model: float, transform: StretchTransform
) -> tuple[float, float]:
    if not math.isfinite(x_model) or not math.isfinite(y_model):
        raise PitchFeaturePreprocessError("non-finite model coordinates")
    x = (float(x_model) - transform.pad_left) / transform.scale_x
    y = (float(y_model) - transform.pad_top) / transform.scale_y
    if not math.isfinite(x) or not math.isfinite(y):
        raise PitchFeaturePreprocessError("non-finite source coordinates")
    return x, y


def source_point_to_model(
    x_source: float, y_source: float, transform: StretchTransform
) -> tuple[float, float]:
    if not math.isfinite(x_source) or not math.isfinite(y_source):
        raise PitchFeaturePreprocessError("non-finite source coordinates")
    x = float(x_source) * transform.scale_x + transform.pad_left
    y = float(y_source) * transform.scale_y + transform.pad_top
    return x, y


def clip_point_to_source(
    x: float, y: float, transform: StretchTransform, *, eps: float = 1e-6
) -> tuple[float, float] | None:
    """Return clipped point if inside [0,W)×[0,H); else None when far outside."""
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    w = float(transform.source_width)
    h = float(transform.source_height)
    if x < -eps or y < -eps or x > w + eps or y > h + eps:
        return None
    return max(0.0, min(w - eps, x)), max(0.0, min(h - eps, y))


def preprocess_rgb_uint8_to_tensor(
    image_rgb: Any,
    *,
    transform: StretchTransform | None = None,
) -> tuple[Any, StretchTransform]:
    """Stretch HxWx3 uint8 RGB → float tensor (1,3,540,960) in [0,1]. No mean/std."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    arr = np.asarray(image_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise PitchFeaturePreprocessError("image must be HxWx3 RGB")
    if arr.dtype != np.uint8:
        raise PitchFeaturePreprocessError("image dtype must be uint8")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    tf = transform or build_stretch_transform(source_width=w, source_height=h)
    if tf.source_width != w or tf.source_height != h:
        raise PitchFeaturePreprocessError("transform source size mismatch")
    # NCHW float [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)
    t = t.unsqueeze(0)
    if (w, h) != (tf.model_width, tf.model_height):
        t = F.interpolate(
            t,
            size=(tf.model_height, tf.model_width),
            mode="bilinear",
            align_corners=False,
        )
    return t, tf


__all__ = [
    "MODEL_WIDTH",
    "MODEL_HEIGHT",
    "PitchFeaturePreprocessError",
    "StretchTransform",
    "build_stretch_transform",
    "model_point_to_source",
    "source_point_to_model",
    "clip_point_to_source",
    "preprocess_rgb_uint8_to_tensor",
]
