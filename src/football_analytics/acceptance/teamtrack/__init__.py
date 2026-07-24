"""TeamTrack (official) MOT sequence adapter for Stage 16-R2 real-video pilot."""

from football_analytics.acceptance.teamtrack.loader import (
    TeamTrackSequence,
    load_mot_gt,
    load_sequence,
    parse_seqinfo,
)
from football_analytics.acceptance.teamtrack.target_selection import (
    select_anonymous_track,
    write_teamtrack_target_receipt,
)

__all__ = [
    "TeamTrackSequence",
    "load_mot_gt",
    "load_sequence",
    "parse_seqinfo",
    "select_anonymous_track",
    "write_teamtrack_target_receipt",
]
