"""Customer metric evaluability taxonomy for Stage 16."""

from __future__ import annotations

from football_analytics.acceptance.contracts import (
    EVALUATED_AGAINST_EXTERNAL_GT,
    MANUAL_REVIEW_REQUIRED,
    NOT_EVALUABLE,
    PIPELINE_DERIVED_NOT_DIRECTLY_GT_SUPPORTED,
)

# Map customer-facing metric keys → evaluability given SoccerTrack v2 GSR+BAS support
DEFAULT_METRIC_TAXONOMY: dict[str, str] = {
    "visibility_coverage": EVALUATED_AGAINST_EXTERNAL_GT,
    "track_continuity": EVALUATED_AGAINST_EXTERNAL_GT,
    "role_team_jersey_agreement": EVALUATED_AGAINST_EXTERNAL_GT,
    "pitch_coordinate_error": EVALUATED_AGAINST_EXTERNAL_GT,
    "trajectory_distance_error": EVALUATED_AGAINST_EXTERNAL_GT,
    "speed_distribution_error": PIPELINE_DERIVED_NOT_DIRECTLY_GT_SUPPORTED,
    "measured_distance": PIPELINE_DERIVED_NOT_DIRECTLY_GT_SUPPORTED,
    "sprint_count": PIPELINE_DERIVED_NOT_DIRECTLY_GT_SUPPORTED,
    "heatmap": PIPELINE_DERIVED_NOT_DIRECTLY_GT_SUPPORTED,
    "pass_events": EVALUATED_AGAINST_EXTERNAL_GT,
    "drive_events": EVALUATED_AGAINST_EXTERNAL_GT,
    "header_events": EVALUATED_AGAINST_EXTERNAL_GT,
    "high_pass_events": EVALUATED_AGAINST_EXTERNAL_GT,
    "tackle_events": EVALUATED_AGAINST_EXTERNAL_GT,
    "pass_accuracy": NOT_EVALUABLE,  # BAS has no pass outcome
    "duel_win_rate": NOT_EVALUABLE,
    "clearance_outcome": NOT_EVALUABLE,
    "recovery_turnover_outcome": NOT_EVALUABLE,
    "detection_map": NOT_EVALUABLE,  # no MOT bbox pack on selected mirror subset
    "broadcast_shot_segmentation": NOT_EVALUABLE,
    "identity_confidence": MANUAL_REVIEW_REQUIRED,
}


def classify_metric(metric_key: str) -> str:
    return DEFAULT_METRIC_TAXONOMY.get(metric_key, NOT_EVALUABLE)


def taxonomy_table() -> list[dict[str, str]]:
    return [{"metric": k, "evaluability": v} for k, v in sorted(DEFAULT_METRIC_TAXONOMY.items())]
