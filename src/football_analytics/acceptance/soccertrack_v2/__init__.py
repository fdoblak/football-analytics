"""SoccerTrack v2 acceptance subpackage."""

from football_analytics.acceptance.soccertrack_v2.formats import (
    BasEvent,
    GsrPlayerObservation,
    VideoHalfMeta,
)
from football_analytics.acceptance.soccertrack_v2.loader import (
    iter_gsr_player_observations,
    load_bas_events,
    probe_video,
)

__all__ = [
    "BasEvent",
    "GsrPlayerObservation",
    "VideoHalfMeta",
    "iter_gsr_player_observations",
    "load_bas_events",
    "probe_video",
]
