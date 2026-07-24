"""Panoramic domain classification for SoccerTrack v2 acceptance."""

from __future__ import annotations

from football_analytics.acceptance.contracts import (
    BROADCAST_ACCEPTANCE_STATUS,
    CAMERA_DOMAIN_PANORAMIC,
    FULL_PITCH_TRACKING_ACCEPTANCE,
)


def classify_camera_domain() -> dict[str, str]:
    return {
        "camera_domain": CAMERA_DOMAIN_PANORAMIC,
        "broadcast_acceptance": BROADCAST_ACCEPTANCE_STATUS,
        "full_pitch_tracking_acceptance": FULL_PITCH_TRACKING_ACCEPTANCE,
        "finding": (
            "SoccerTrack v2 provides panoramic full-pitch video, not classic TV broadcast. "
            "Shot/camera/replay broadcast modules are not validated by this dataset; "
            "do not invent broadcast cuts from continuous panorama."
        ),
    }
