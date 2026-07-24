"""Namespace constants for Stage 16-R4 three-track acceptance."""

from __future__ import annotations

NAMESPACE_TEAMTRACK_REAL_VIDEO = "teamtrack_real_video_pilot"
NAMESPACE_SOCCERTRACK_REFERENCE = "soccertrack_v2_reference_analysis"
NAMESPACE_SELF_CONTAINED = "self_contained_deterministic_acceptance"

AUTHORITATIVE_SOCCERTRACK_TARGET = {
    "match_id": "128057",
    "team_side": "left",
    "jersey_number": 24,
    "player_id": "506469",
}

DEPRECATED_INVALID_TARGET = {
    "jersey_number": 11,
    "player_id": "506466",
    "status": "deprecated_invalid_delayed_result",
}

TEAMTRACK_PILOT_TARGET = {
    "dataset": "TeamTrack",
    "sequence": "F_20200220_1_0330_0360",
    "anonymous_track_id": 7,
}

EVIDENCE_LEVELS = (
    "REAL_VIDEO_VALIDATED",
    "REFERENCE_ANNOTATION_DERIVED",
    "SELF_CONTAINED_TESTED",
    "NOT_EVALUABLE",
)

GATE_TECHNICAL_PREVIEW = (
    "PASS_WITH_FINDINGS — SELF-CONTAINED TECHNICAL ACCEPTANCE COMPLETE; "
    "REAL-VIDEO TRACKING VALIDATED; VIDEO-EVENT ACCURACY NOT VALIDATED"
)

__all__ = [
    "AUTHORITATIVE_SOCCERTRACK_TARGET",
    "DEPRECATED_INVALID_TARGET",
    "EVIDENCE_LEVELS",
    "GATE_TECHNICAL_PREVIEW",
    "NAMESPACE_SELF_CONTAINED",
    "NAMESPACE_SOCCERTRACK_REFERENCE",
    "NAMESPACE_TEAMTRACK_REAL_VIDEO",
    "TEAMTRACK_PILOT_TARGET",
]
