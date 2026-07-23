"""Ultralytics YOLO person adapter — lazy import; local weights only; no network."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.perception.adapters.base import PersonDetectorAdapter, RawPersonBox

ADAPTER_ID = "ultralytics_person"
ADAPTER_VERSION = "1.0.0"


class UltralyticsPersonAdapterError(ValueError):
    """Ultralytics person adapter failure."""


def _reject_network_path(path: str) -> Path:
    text = str(path).strip()
    if not text:
        raise UltralyticsPersonAdapterError("weights_path empty")
    lowered = text.lower()
    if "://" in text or lowered.startswith(("http:", "https:", "ftp:", "s3:", "gs:")):
        raise UltralyticsPersonAdapterError("NETWORK_WEIGHTS_FORBIDDEN")
    if ".." in Path(text).parts:
        raise UltralyticsPersonAdapterError("WEIGHTS_PATH_ESCAPE")
    target = Path(text)
    if not target.is_absolute():
        raise UltralyticsPersonAdapterError("weights_path must be absolute")
    if target.is_symlink():
        raise UltralyticsPersonAdapterError("weights_path must not be a symlink")
    if not target.is_file():
        raise UltralyticsPersonAdapterError(f"weights missing: {target}")
    return target


class UltralyticsPersonAdapter(PersonDetectorAdapter):
    """YOLO11 COCO person (class 0) adapter. Ultralytics/torch imported only in load/predict."""

    def __init__(self) -> None:
        self._model: Any = None
        self._weights_path: Path | None = None
        self._weights_sha256: str | None = None
        self._names: dict[int, str] = {}

    @property
    def adapter_id(self) -> str:
        return ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return ADAPTER_VERSION

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, weights_path: str, expected_sha256: str) -> None:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise UltralyticsPersonAdapterError("expected_sha256 must be 64-char hex")
        expected = expected_sha256.lower()
        if any(c not in "0123456789abcdef" for c in expected):
            raise UltralyticsPersonAdapterError("expected_sha256 must be lowercase hex")

        path = _reject_network_path(weights_path)
        actual = sha256_file(path)
        if actual.lower() != expected:
            raise UltralyticsPersonAdapterError("MODEL_HASH_MISMATCH")

        # Offline / no auto-download at inference.
        os.environ.setdefault("YOLO_OFFLINE", "1")
        os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

        from ultralytics import YOLO  # lazy

        self._model = YOLO(str(path), task="detect")
        self._weights_path = path
        self._weights_sha256 = actual.lower()
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, dict):
            self._names = {int(k): str(v).lower() for k, v in names.items()}
        else:
            self._names = {i: str(n).lower() for i, n in enumerate(names)}

    def unload(self) -> None:
        self._model = None
        self._weights_path = None
        self._weights_sha256 = None
        self._names = {}

    def software_versions(self) -> Mapping[str, str]:
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

    def predict_persons(
        self,
        image_bgr_or_rgb: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        half: bool,
        class_ids: Sequence[int],
        class_names: Sequence[str],
        channel_order: str = "bgr",
    ) -> list[RawPersonBox]:
        if self._model is None:
            raise UltralyticsPersonAdapterError("MODEL_NOT_LOADED")
        if channel_order not in {"bgr", "rgb"}:
            raise UltralyticsPersonAdapterError("channel_order must be bgr|rgb")

        import numpy as np

        arr = np.asarray(image_bgr_or_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise UltralyticsPersonAdapterError("image must be HxWx3")
        # Ultralytics expects BGR ndarray for OpenCV-style inputs.
        if channel_order == "rgb":
            arr = arr[:, :, ::-1].copy()

        allowed_ids = {int(x) for x in class_ids}
        allowed_names = {str(n).strip().lower() for n in class_names}
        # Force person-only classes into predict when possible (COCO id 0).
        classes_arg = sorted(allowed_ids) if allowed_ids else None

        use_half = bool(half) and str(device).startswith("cuda")
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
        results = self._model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # Ultralytics maps boxes back to the original image size when predict()
        # is given the original image; xyxy here is source-frame space.
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

        out: list[RawPersonBox] = []
        for i in range(len(xyxy_np)):
            cid = int(cls_np[i])
            cname = str(self._names.get(cid, "")).strip().lower().replace(" ", "_")
            # Reject sports ball / ball and every non-person class (Stage 5C owns ball).
            if cname in {"sports_ball", "ball", "soccer_ball", "football"}:
                continue
            id_ok = cid in allowed_ids
            name_ok = cname in allowed_names or cname == "person"
            if not (id_ok or name_ok):
                continue
            if not id_ok and cname != "person":
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy_np[i])
            score = float(conf_np[i])
            out.append(
                RawPersonBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    class_id=cid if id_ok else 0,
                    class_name="person",
                )
            )
        return out


def get_person_adapter(adapter_id: str) -> PersonDetectorAdapter:
    if adapter_id == ADAPTER_ID:
        return UltralyticsPersonAdapter()
    raise UltralyticsPersonAdapterError(f"unknown adapter_id: {adapter_id}")


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "UltralyticsPersonAdapterError",
    "UltralyticsPersonAdapter",
    "get_person_adapter",
]
