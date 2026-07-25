"""Stage 18 own-video perception package."""

from football_analytics.acceptance.stage18_own_video.pipeline import (
    OWN_VIDEO_CFG,
    classify_human_role,
    compute_pitch_masks,
    detect_balls,
    detect_persons,
)

__all__ = [
    "OWN_VIDEO_CFG",
    "classify_human_role",
    "compute_pitch_masks",
    "detect_balls",
    "detect_persons",
]
