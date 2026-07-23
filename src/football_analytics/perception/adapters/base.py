"""Abstract person-detector adapter protocol (no network, no eager model load)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawPersonBox:
    """Raw person detection in the coordinate space documented by the adapter."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class PersonDetectorAdapter(ABC):
    """Detector adapter for generic person boxes only (Stage 5B)."""

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Stable adapter identifier."""

    @property
    @abstractmethod
    def adapter_version(self) -> str:
        """Adapter version string."""

    @abstractmethod
    def load(self, weights_path: str, expected_sha256: str) -> None:
        """Load local weights after SHA-256 verification. Network forbidden."""

    @abstractmethod
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
        """Return person-class boxes only (class_id/name/score + xyxy)."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    def close(self) -> None:
        self.unload()

    def is_loaded(self) -> bool:
        return False

    def software_versions(self) -> Mapping[str, str]:
        return {}


__all__ = ["RawPersonBox", "PersonDetectorAdapter"]
