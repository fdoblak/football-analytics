"""Shared Ultralytics load / offline / predict helpers (no eager import)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.perception.adapters.base import RawDetectionBox

SPORTS_BALL_CANONICAL = "sports ball"
SPORTS_BALL_NORMALIZED = "sports_ball"


class UltralyticsCommonError(ValueError):
    """Shared Ultralytics adapter failure."""


def reject_network_weights_path(path: str) -> Path:
    text = str(path).strip()
    if not text:
        raise UltralyticsCommonError("weights_path empty")
    lowered = text.lower()
    if "://" in text or lowered.startswith(("http:", "https:", "ftp:", "s3:", "gs:")):
        raise UltralyticsCommonError("NETWORK_WEIGHTS_FORBIDDEN")
    if ".." in Path(text).parts:
        raise UltralyticsCommonError("WEIGHTS_PATH_ESCAPE")
    target = Path(text)
    if not target.is_absolute():
        raise UltralyticsCommonError("weights_path must be absolute")
    if target.is_symlink():
        raise UltralyticsCommonError("weights_path must not be a symlink")
    if not target.is_file():
        raise UltralyticsCommonError(f"weights missing: {target}")
    return target


def validate_sha256_hex(expected_sha256: str) -> str:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise UltralyticsCommonError("expected_sha256 must be 64-char hex")
    expected = expected_sha256.lower()
    if any(c not in "0123456789abcdef" for c in expected):
        raise UltralyticsCommonError("expected_sha256 must be lowercase hex")
    return expected


def set_ultralytics_offline_env() -> None:
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def parse_model_names(names: Any) -> dict[int, str]:
    """Return id → original lowercased name (preserve spaces for COCO display)."""
    if isinstance(names, dict):
        return {int(k): str(v).strip().lower() for k, v in names.items()}
    return {i: str(n).strip().lower() for i, n in enumerate(names)}


def load_yolo_local(weights_path: Path) -> Any:
    set_ultralytics_offline_env()
    from ultralytics import YOLO  # lazy

    return YOLO(str(weights_path), task="detect")


def verify_and_load_weights(
    weights_path: str, expected_sha256: str
) -> tuple[Any, Path, str, dict[int, str]]:
    expected = validate_sha256_hex(expected_sha256)
    path = reject_network_weights_path(weights_path)
    actual = sha256_file(path)
    if actual.lower() != expected:
        raise UltralyticsCommonError("MODEL_HASH_MISMATCH")
    model = load_yolo_local(path)
    names = parse_model_names(getattr(model, "names", None) or {})
    return model, path, actual.lower(), names


def software_versions_map() -> Mapping[str, str]:
    out: dict[str, str] = {}
    try:
        import ultralytics

        out["ultralytics"] = str(ultralytics.__version__)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        out["torch"] = str(torch.__version__)
    except Exception:  # noqa: BLE001
        pass
    return out


def prepare_bgr_image(image_bgr_or_rgb: Any, *, channel_order: str) -> Any:
    import numpy as np

    if channel_order not in {"bgr", "rgb"}:
        raise UltralyticsCommonError("channel_order must be bgr|rgb")
    arr = np.asarray(image_bgr_or_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise UltralyticsCommonError("image must be HxWx3")
    if channel_order == "rgb":
        return arr[:, :, ::-1].copy()
    return arr


def run_yolo_predict(
    model: Any,
    arr: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    half: bool,
    class_ids: Sequence[int] | None,
) -> Any:
    use_half = bool(half) and str(device).startswith("cuda")
    classes_arg = sorted({int(x) for x in class_ids}) if class_ids else None
    predict_kwargs: dict[str, Any] = {
        "source": arr,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "device": device,
        "verbose": False,
        "classes": classes_arg,
        "batch": 1,
    }
    if use_half:
        predict_kwargs["half"] = True
    results = model.predict(**predict_kwargs)
    if not results:
        return None
    return results[0]


def extract_xyxy_conf_cls(result: Any) -> tuple[Any, Any, Any] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy
    confs = boxes.conf
    clss = boxes.cls
    try:
        xyxy_np = xyxy.detach().cpu().numpy()
        conf_np = confs.detach().cpu().numpy()
        cls_np = clss.detach().cpu().numpy().astype(int)
    except Exception:  # noqa: BLE001
        xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
        conf_np = confs.cpu().numpy() if hasattr(confs, "cpu") else confs
        cls_np = clss.cpu().numpy().astype(int) if hasattr(clss, "cpu") else clss
    return xyxy_np, conf_np, cls_np


def boxes_from_result(
    result: Any,
    *,
    names: Mapping[int, str],
    allowed_ids: set[int],
    allowed_names_normalized: set[str],
    emit_class_name: str,
    reject_normalized: set[str] | None = None,
    require_exact_name: str | None = None,
) -> list[RawDetectionBox]:
    """Filter YOLO result boxes to an allowed class set."""
    extracted = extract_xyxy_conf_cls(result)
    if extracted is None:
        return []
    xyxy_np, conf_np, cls_np = extracted
    reject = reject_normalized or set()
    out: list[RawDetectionBox] = []
    for i in range(len(xyxy_np)):
        cid = int(cls_np[i])
        raw_name = str(names.get(cid, "")).strip().lower()
        cname_norm = normalize_class_name(raw_name)
        if cname_norm in reject:
            continue
        if require_exact_name is not None and raw_name != require_exact_name:
            continue
        id_ok = cid in allowed_ids
        name_ok = cname_norm in allowed_names_normalized or raw_name in {
            n.replace("_", " ") for n in allowed_names_normalized
        }
        if not (id_ok or name_ok):
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy_np[i])
        score = float(conf_np[i])
        out.append(
            RawDetectionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=score,
                class_id=cid if id_ok else next(iter(allowed_ids), cid),
                class_name=emit_class_name,
            )
        )
    return out


def require_sports_ball_in_names(names: Mapping[int, str]) -> int:
    """Verify COCO sports ball is present; return its class id. BLOCKED if missing."""
    for cid, name in names.items():
        if str(name).strip().lower() == SPORTS_BALL_CANONICAL:
            return int(cid)
    # Fallback: normalized match
    for cid, name in names.items():
        if normalize_class_name(str(name)) == SPORTS_BALL_NORMALIZED:
            return int(cid)
    raise UltralyticsCommonError("SPORTS_BALL_CLASS_MISSING")


__all__ = [
    "SPORTS_BALL_CANONICAL",
    "SPORTS_BALL_NORMALIZED",
    "UltralyticsCommonError",
    "reject_network_weights_path",
    "validate_sha256_hex",
    "set_ultralytics_offline_env",
    "normalize_class_name",
    "parse_model_names",
    "load_yolo_local",
    "verify_and_load_weights",
    "software_versions_map",
    "prepare_bgr_image",
    "run_yolo_predict",
    "extract_xyxy_conf_cls",
    "boxes_from_result",
    "require_sports_ball_in_names",
]
