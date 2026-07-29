"""Perception detector adapters (lazy-load friendly; no eager Ultralytics import)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "PersonDetectorAdapter",
    "BallDetectorAdapter",
    "RawDetectionBox",
    "RawPersonBox",
    "UltralyticsPersonAdapter",
    "UltralyticsBallAdapter",
    "SoccerNetGameStateDetectorAdapter",
    "get_person_adapter",
    "get_ball_adapter",
    "get_soccernet_gamestate_detector_adapter",
]


def __getattr__(name: str) -> Any:
    if name in {"PersonDetectorAdapter", "BallDetectorAdapter", "RawDetectionBox", "RawPersonBox"}:
        from football_analytics.perception.adapters import base as _base

        return getattr(_base, name)
    if name == "UltralyticsPersonAdapter":
        from football_analytics.perception.adapters.ultralytics_person import (
            UltralyticsPersonAdapter,
        )

        return UltralyticsPersonAdapter
    if name == "get_person_adapter":
        from football_analytics.perception.adapters.ultralytics_person import get_person_adapter

        return get_person_adapter
    if name == "UltralyticsBallAdapter":
        from football_analytics.perception.adapters.ultralytics_ball import UltralyticsBallAdapter

        return UltralyticsBallAdapter
    if name == "get_ball_adapter":
        from football_analytics.perception.adapters.ultralytics_ball import get_ball_adapter

        return get_ball_adapter
    if name == "SoccerNetGameStateDetectorAdapter":
        from football_analytics.perception.adapters.soccernet_gamestate_detector import (
            SoccerNetGameStateDetectorAdapter,
        )

        return SoccerNetGameStateDetectorAdapter
    if name == "get_soccernet_gamestate_detector_adapter":
        from football_analytics.perception.adapters.soccernet_gamestate_detector import (
            get_soccernet_gamestate_detector_adapter,
        )

        return get_soccernet_gamestate_detector_adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
