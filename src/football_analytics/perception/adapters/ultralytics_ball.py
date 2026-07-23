"""Ultralytics YOLO sports-ball adapter — lazy import; local weights only; no network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_analytics.perception.adapters.base import BallDetectorAdapter, RawDetectionBox
from football_analytics.perception.adapters.ultralytics_common import (
    SPORTS_BALL_CANONICAL,
    UltralyticsCommonError,
    boxes_from_result,
    normalize_class_name,
    prepare_bgr_image,
    require_sports_ball_in_names,
    run_yolo_predict,
    software_versions_map,
    verify_and_load_weights,
)

ADAPTER_ID = "ultralytics_sports_ball"
ADAPTER_VERSION = "1.0.0"


class UltralyticsBallAdapterError(ValueError):
    """Ultralytics sports-ball adapter failure."""


class UltralyticsBallAdapter(BallDetectorAdapter):
    """YOLO11 COCO sports ball (class 32) adapter. Shares load/hash path with person."""

    def __init__(self) -> None:
        self._model: Any = None
        self._weights_path: Path | None = None
        self._weights_sha256: str | None = None
        self._names: dict[int, str] = {}
        self._sports_ball_id: int | None = None

    @property
    def adapter_id(self) -> str:
        return ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return ADAPTER_VERSION

    def is_loaded(self) -> bool:
        return self._model is not None

    def model_names(self) -> Mapping[int, str]:
        return dict(self._names)

    def load(self, weights_path: str, expected_sha256: str) -> None:
        try:
            model, path, sha, names = verify_and_load_weights(weights_path, expected_sha256)
            sports_id = require_sports_ball_in_names(names)
        except UltralyticsCommonError as exc:
            raise UltralyticsBallAdapterError(str(exc)) from exc
        self._model = model
        self._weights_path = path
        self._weights_sha256 = sha
        self._names = names
        self._sports_ball_id = sports_id

    def unload(self) -> None:
        self._model = None
        self._weights_path = None
        self._weights_sha256 = None
        self._names = {}
        self._sports_ball_id = None

    def software_versions(self) -> Mapping[str, str]:
        return software_versions_map()

    def predict_balls(
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
    ) -> list[RawDetectionBox]:
        if self._model is None or self._sports_ball_id is None:
            raise UltralyticsBallAdapterError("MODEL_NOT_LOADED")
        try:
            arr = prepare_bgr_image(image_bgr_or_rgb, channel_order=channel_order)
        except UltralyticsCommonError as exc:
            raise UltralyticsBallAdapterError(str(exc)) from exc

        allowed_ids = {int(x) for x in class_ids} or {int(self._sports_ball_id)}
        # Only accept verified sports-ball id from loaded names when intersecting.
        allowed_ids = allowed_ids & {int(self._sports_ball_id)}
        if not allowed_ids:
            raise UltralyticsBallAdapterError("SPORTS_BALL_CLASS_MISMATCH")

        allowed_names = {normalize_class_name(n) for n in class_names}
        # Config may list "sports ball" or "sports_ball".
        allowed_names.add("sports_ball")
        result = run_yolo_predict(
            self._model,
            arr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            half=half,
            class_ids=sorted(allowed_ids),
        )
        if result is None:
            return []

        # Accept ONLY verified sports ball; reject person and all others.
        return boxes_from_result(
            result,
            names=self._names,
            allowed_ids=allowed_ids,
            allowed_names_normalized=allowed_names,
            emit_class_name=SPORTS_BALL_CANONICAL,
            reject_normalized={"person", "human", "pedestrian"},
            require_exact_name=SPORTS_BALL_CANONICAL,
        )


def get_ball_adapter(adapter_id: str) -> BallDetectorAdapter:
    if adapter_id in {ADAPTER_ID, "ultralytics_coco_sports_ball"}:
        return UltralyticsBallAdapter()
    raise UltralyticsBallAdapterError(f"unknown adapter_id: {adapter_id}")


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "UltralyticsBallAdapterError",
    "UltralyticsBallAdapter",
    "get_ball_adapter",
]
