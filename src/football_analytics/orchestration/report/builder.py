"""Canonical single-player report builder (Stage 14C). NO team summary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.orchestration.contracts import (
    load_report_json_schema,
    validate_against_json_schema,
)
from football_analytics.orchestration.stage_handlers import write_stage_receipt


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _metric(
    metric_id: str,
    *,
    value: Any,
    status: str = "ok",
    confidence: float | None = 0.7,
    coverage: float | None = 0.8,
    provenance: str = "project_generated",
    unit: str | None = None,
    reason_not_evaluable: str | None = None,
    components: dict[str, Any] | None = None,
    display_name_en: str | None = None,
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "display_name_tr": None,
        "display_name_en": display_name_en or metric_id,
        "version": 1,
        "value": value,
        "unit": unit,
        "status": status,
        "confidence": confidence,
        "coverage": coverage,
        "reason_not_evaluable": reason_not_evaluable,
        "provenance": provenance,
        "components": components,
    }


def synthetic_metric_bundle(
    *,
    identity_uncertain: bool = False,
    include_not_evaluable: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate physical/spatial/interaction/pass/duel style metrics (synthetic)."""
    id_status = "identity_uncertain" if identity_uncertain else "ok"
    id_conf = 0.4 if identity_uncertain else 0.85
    rows = [
        _metric(
            "distance_covered_m",
            value=None if identity_uncertain else 812.5,
            status="identity_uncertain" if identity_uncertain else "ok",
            confidence=id_conf,
            coverage=0.82,
            provenance="video_derived",
            unit="m",
            reason_not_evaluable="identity_uncertain" if identity_uncertain else None,
        ),
        _metric(
            "speed_avg_mps",
            value=None if identity_uncertain else 2.1,
            status=id_status if identity_uncertain else "ok",
            confidence=id_conf,
            coverage=0.8,
            provenance="video_derived",
            unit="m/s",
            reason_not_evaluable="identity_uncertain" if identity_uncertain else None,
        ),
        _metric(
            "sprint_count",
            value=None if identity_uncertain else 4,
            status=id_status if identity_uncertain else "ok",
            confidence=id_conf,
            coverage=0.78,
            provenance="video_derived",
        ),
        _metric(
            "sprint_max_speed_mps",
            value=None if identity_uncertain else 7.4,
            status=id_status if identity_uncertain else "ok",
            confidence=id_conf,
            coverage=0.78,
            provenance="video_derived",
            unit="m/s",
        ),
        _metric(
            "heatmap",
            value={"grid": "synthetic_8x12", "peak_zone": "mid_third"},
            status="ok",
            coverage=0.75,
            provenance="video_derived",
        ),
        _metric(
            "activation",
            value={"low": 0.2, "medium": 0.5, "high": 0.3},
            status="ok",
            coverage=0.75,
            provenance="project_generated",
        ),
        _metric(
            "pass_attempts",
            value=12,
            status="ok",
            coverage=0.7,
            provenance="opta_style_project",
        ),
        _metric(
            "pass_completion_rate",
            value=0.67,
            status="ok",
            coverage=0.7,
            provenance="opta_style_project",
        ),
        _metric(
            "dribbles_successful",
            value=3,
            status="ok",
            coverage=0.65,
            provenance="opta_style_project",
        ),
        _metric(
            "dribbles_failed",
            value=2,
            status="ok",
            coverage=0.65,
            provenance="opta_style_project",
        ),
        _metric(
            "take_on_success_rate",
            value=0.6,
            status="ok",
            coverage=0.65,
            provenance="opta_style_project",
        ),
        _metric(
            "duels_won",
            value=5,
            status="ok",
            coverage=0.6,
            provenance="opta_style_project",
        ),
        _metric(
            "duel_win_rate",
            value=0.55,
            status="ok",
            coverage=0.6,
            provenance="opta_style_project",
        ),
        _metric(
            "tackles_interceptions_recoveries",
            value={"tackles": 2, "recoveries": 3, "interceptions": 1},
            status="ok",
            coverage=0.6,
            provenance="opta_style_project",
        ),
        _metric(
            "ball_losses",
            value=4,
            status="ok",
            coverage=0.6,
            provenance="opta_style_project",
        ),
        _metric(
            "aerial_duels",
            value=3,
            status="ok",
            coverage=0.55,
            provenance="opta_style_project",
        ),
        _metric(
            "aerial_win_rate",
            value=0.33,
            status="ok",
            coverage=0.55,
            provenance="opta_style_project",
        ),
        _metric(
            "clearances",
            value=1,
            status="ok",
            coverage=0.55,
            provenance="opta_style_project",
        ),
        _metric(
            "penalty_area_ball_touches",
            value=2,
            status="ok",
            coverage=0.5,
            provenance="video_derived",
        ),
        _metric(
            "coverage_reliability",
            value=0.72,
            status="ok",
            coverage=1.0,
            provenance="project_generated",
        ),
    ]
    if include_not_evaluable:
        rows.append(
            _metric(
                "progressive_passes_def_to_mid",
                value=None,
                status="not_evaluable",
                confidence=None,
                coverage=0.4,
                provenance="opta_style_project",
                reason_not_evaluable="attack_direction_unknown",
            )
        )
        rows.append(
            _metric(
                "progressive_passes_mid_to_att",
                value=None,
                status="not_evaluable",
                confidence=None,
                coverage=0.4,
                provenance="opta_style_project",
                reason_not_evaluable="attack_direction_unknown",
            )
        )
        rows.append(
            _metric(
                "long_pass_attempts",
                value=None,
                status="calibration_insufficient",
                confidence=None,
                coverage=0.3,
                provenance="video_derived",
                reason_not_evaluable="calibration_insufficient",
            )
        )
        rows.append(
            _metric(
                "long_pass_completion_rate",
                value=None,
                status="calibration_insufficient",
                confidence=None,
                coverage=0.3,
                provenance="video_derived",
                reason_not_evaluable="calibration_insufficient",
            )
        )
        rows.append(
            _metric(
                "opta_style_events",
                value=None,
                status="event_uncertain",
                confidence=None,
                coverage=0.5,
                provenance="opta_style_project",
                reason_not_evaluable="no_reviewed_ground_truth",
            )
        )
    return rows


def build_single_player_report(
    *,
    run_id: str,
    git_commit: str,
    target_player_id: str,
    display_name: str,
    match_id: str,
    video_id: str,
    metrics: Sequence[Mapping[str, Any]] | None = None,
    identity_confidence: float | None = 0.85,
    manual_identity_status: str = "confirmed",
    coverage: Mapping[str, Any] | None = None,
    source_artifact_index: Sequence[Mapping[str, Any]] | None = None,
    warnings: Sequence[str] | None = None,
    analyzed_duration_s: float | None = 5400.0,
) -> dict[str, Any]:
    metric_rows = [dict(m) for m in (metrics or synthetic_metric_bundle())]
    not_eval = [
        str(m["metric_id"])
        for m in metric_rows
        if m.get("status") != "ok" or m.get("value") is None
    ]
    body: dict[str, Any] = {
        "schema_version": 1,
        "dictionary_version": 1,
        "run_id": run_id,
        "git_commit": git_commit,
        "generated_at_utc": _utc_now(),
        "target_player": {
            "target_player_id": target_player_id,
            "display_name": display_name,
            "team_hint": None,
            "jersey_number_hint": None,
            "match_id": match_id,
            "reference_images": [],
            "manual_identity_status": manual_identity_status,
            "identity_confidence": identity_confidence,
        },
        "match": {
            "match_id": match_id,
            "competition_hint": None,
            "home_team_hint": None,
            "away_team_hint": None,
            "kickoff_local": None,
            "video_path": None,
            "analysis_duration_s": analyzed_duration_s,
            "tracked_duration_s": analyzed_duration_s,
        },
        "coverage": dict(
            coverage
            or {
                "track_coverage": 0.82,
                "calibration_coverage": 0.55,
                "ball_tracking_coverage": 0.7,
                "metrics_evaluable_fraction": 0.72,
                "notes": "synthetic stage 14 report",
            }
        ),
        "identity_confidence": identity_confidence,
        "models_versions": {"stage": "14", "mode": "synthetic"},
        "code_versions": {"orchestration": "1"},
        "metrics": metric_rows,
        "evidence_refs": [],
        "warnings": list(warnings or ["REAL FOOTBALL ACCURACY NOT YET VALIDATED"]),
        "source_artifact_index": [dict(x) for x in (source_artifact_index or [])],
        "not_evaluable_metric_ids": not_eval,
        "team_summary_forbidden": True,
        "reproducibility_fingerprint": "",
    }
    # Team summary must never appear.
    if "team_summary" in body:
        raise RuntimeError("team_summary forbidden in single-player report")
    body["reproducibility_fingerprint"] = hash_canonical_json(
        {
            "run_id": run_id,
            "target_player_id": target_player_id,
            "match_id": match_id,
            "video_id": video_id,
            "metrics": metric_rows,
            "coverage": body["coverage"],
            "source_artifact_index": body["source_artifact_index"],
        }
    )
    schema = load_report_json_schema()
    validate_against_json_schema(body, schema)
    return body


def build_and_write_report_stage(
    *,
    stage_dir: Path,
    run_id: str,
    cache_key: str,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage_states: Sequence[Mapping[str, Any]],
    out_root: Path,
) -> dict[str, Any]:
    import subprocess

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[4]),
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        git_commit = "0" * 40

    source_index = []
    for st in stage_states:
        if st.get("artifact_path"):
            source_index.append(
                {
                    "logical_name": str(st["name"]),
                    "relative_path": str(st["artifact_path"]),
                    "sha256": str(st.get("cache_key") or ("0" * 64)),
                    "stage": str(st["name"]),
                }
            )
    report = build_single_player_report(
        run_id=str(run_id),
        git_commit=git_commit if len(git_commit) == 40 else "0" * 40,
        target_player_id=str(request["target_player_id"]),
        display_name=str(request.get("display_name") or request["target_player_id"]),
        match_id=str(request.get("match_id") or request["video_id"]),
        video_id=str(request["video_id"]),
        source_artifact_index=source_index,
        warnings=[
            "REAL FOOTBALL ACCURACY NOT YET VALIDATED",
            f"plan_fingerprint={plan.get('plan_fingerprint', '')[:16]}",
        ],
    )
    report_path = out_root / "single_player_report.json"
    write_json_record(report_path, report, overwrite=True)
    return write_stage_receipt(
        stage_dir,
        stage_name="report",
        run_id=str(run_id),
        cache_key=cache_key,
        mode="report_builder",
        extras={"report_json": str(report_path), "team_summary": False},
        overwrite=True,
    )


__all__ = [
    "synthetic_metric_bundle",
    "build_single_player_report",
    "build_and_write_report_stage",
]
