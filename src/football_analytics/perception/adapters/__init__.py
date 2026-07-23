"""Perception detector adapters (lazy-load friendly; no eager Ultralytics import)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "PersonDetectorAdapter",
    "UltralyticsPersonAdapter",
    "get_person_adapter",
]


def __getattr__(name: str) -> Any:
    if name == "PersonDetectorAdapter":
        from football_analytics.perception.adapters.base import PersonDetectorAdapter

        return PersonDetectorAdapter
    if name == "UltralyticsPersonAdapter":
        from football_analytics.perception.adapters.ultralytics_person import (
            UltralyticsPersonAdapter,
        )

        return UltralyticsPersonAdapter
    if name == "get_person_adapter":
        from football_analytics.perception.adapters.ultralytics_person import get_person_adapter

        return get_person_adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
