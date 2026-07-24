"""Stage 16 final report + visual helpers (acceptance-only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ATTRIBUTION_FOOTER = (
    "Source: SoccerTrack v2 — CC BY 4.0\n"
    "Project-generated analysis; not official Opta data."
)


def build_acceptance_report(
    *,
    match_id: str,
    target_display: str,
    target_player_id: str,
    team_side: str,
    jersey_number: int,
    camera_domain: dict[str, str],
    evaluability_table: list[dict[str, str]],
    gsr_eval: dict[str, Any],
    bas_eval: dict[str, Any],
    pipeline_metrics: dict[str, Any],
    coverage: dict[str, Any],
    findings: list[str],
    identity_source: str,
) -> dict[str, Any]:
    """Canonical single-player acceptance report (honest nulls / not_evaluable)."""

    def metric(value: Any, evaluability: str, **extra: Any) -> dict[str, Any]:
        return {
            "value": value,
            "evaluability": evaluability,
            "provenance": extra.get("provenance", "pipeline_or_external_gt"),
            "coverage": extra.get("coverage"),
            "confidence": extra.get("confidence"),
            "definition_version": extra.get("definition_version", "stage16_v1"),
            "notes": extra.get("notes"),
        }

    report = {
        "schema": "single_player_acceptance_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "SoccerTrack v2",
            "license": "CC-BY-4.0",
            "match_id": match_id,
            "attribution": ATTRIBUTION_FOOTER,
            "official_urls": [
                "https://atomscott.github.io/SoccerTrack-v2/",
                "https://creativecommons.org/licenses/by/4.0/",
            ],
        },
        "target": {
            "display_name": target_display,
            "player_id": target_player_id,
            "team_side": team_side,
            "jersey_number": jersey_number,
            "identity_source": identity_source,
            "identity_confidence": pipeline_metrics.get("identity_confidence", "external_reference_confirmation"),
        },
        "camera_domain": camera_domain,
        "coverage": coverage,
        "metrics": {
            "analyzed_duration_s": metric(
                pipeline_metrics.get("analyzed_duration_s"),
                "pipeline_derived_not_directly_gt_supported",
            ),
            "visibility_coverage": metric(
                coverage.get("visibility_coverage"),
                "evaluated_against_external_gt",
                coverage=coverage.get("visibility_coverage"),
            ),
            "measured_distance_m": metric(
                pipeline_metrics.get("measured_distance_m"),
                "pipeline_derived_not_directly_gt_supported",
            ),
            "robust_mean_speed_mps": metric(
                pipeline_metrics.get("robust_mean_speed_mps"),
                "pipeline_derived_not_directly_gt_supported",
            ),
            "peak_speed_mps": metric(
                pipeline_metrics.get("peak_speed_mps"),
                "pipeline_derived_not_directly_gt_supported",
            ),
            "sprint_count": metric(
                pipeline_metrics.get("sprint_count"),
                "pipeline_derived_not_directly_gt_supported",
            ),
            "pass_events": metric(
                bas_eval.get("per_label", {}).get("Pass"),
                "evaluated_against_external_gt",
            ),
            "pass_accuracy": metric(
                None,
                "not_evaluable",
                notes="BAS has no pass outcome labels",
            ),
            "drive_events": metric(
                bas_eval.get("per_label", {}).get("Drive"),
                "evaluated_against_external_gt",
            ),
            "duel_win_rate": metric(
                None,
                "not_evaluable",
                notes="reference_gt_not_available for duel outcomes",
            ),
            "heatmap": metric(
                pipeline_metrics.get("heatmap_summary"),
                "pipeline_derived_not_directly_gt_supported",
            ),
        },
        "external_gt_comparison": {
            "gsr": gsr_eval,
            "bas": bas_eval,
        },
        "evaluability_table": evaluability_table,
        "warnings_findings": findings,
        "metric_origin": "project_generated",
        "definition_style": "opta_style_metric_definition",
        "official_opta_claim": False,
    }
    return report


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")  # ensure flushed
