"""Stage 13E target-events fusion pipeline (13A→13B→13C→13D)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.records import write_json_record
from football_analytics.data.registry import default_project_root
from football_analytics.events.attack_direction import resolve_match_attack_directions
from football_analytics.events.evaluation import NOT_EVALUATED, evaluate_events
from football_analytics.events.ledger_service import build_target_event_ledger
from football_analytics.events.metrics_service import compute_target_event_metrics
from football_analytics.events.pipeline_config import (
    EventsPipelineConfigError,
    events_pipeline_config_fingerprint,
    load_events_pipeline_config,
)
from football_analytics.events.policy import load_events_policy, policy_fingerprint
from football_analytics.events.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
    build_synthetic_review_queue,
)
from football_analytics.events.replay_service import compute_replay_candidates
from football_analytics.events.validation import validate_events_bundle


class EventsPipelineError(RuntimeError):
    """Events pipeline fusion failure."""


@dataclass
class EventsPipelineResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    review_queue_json: str | None
    replay_parquet: str | None
    ledger_parquet: str | None
    revisions_parquet: str | None
    metrics_json: str | None
    summary: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    output_dir: Path,
    config_fp: str,
    code: str,
    summary: dict[str, Any],
) -> EventsPipelineResult:
    err_path = output_dir / "failure_receipt.json"
    write_json_record(
        err_path,
        {
            "schema_version": 1,
            "status": "failed",
            "error_code": code,
            "config_fingerprint": config_fp,
            "created_at_utc": _utc_now(),
            "summary": summary,
        },
        overwrite=True,
    )
    return EventsPipelineResult(
        accepted=False,
        exit_code=1,
        error_code=code,
        config_fingerprint=config_fp,
        summary_json=None,
        receipt_json=str(err_path),
        quality_json=None,
        evaluation_json=None,
        review_queue_json=None,
        replay_parquet=None,
        ledger_parquet=None,
        revisions_parquet=None,
        metrics_json=None,
        summary=summary,
    )


def integrate_target_events(
    *,
    output_dir: Path,
    sources: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    replay_contexts: Sequence[Mapping[str, Any]] | None = None,
    attack_periods: Sequence[Mapping[str, Any]] | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    interaction_coverage: float | None = 0.85,
    project_root: Path | None = None,
) -> EventsPipelineResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_events_pipeline_config(config_path, project_root=root)
    except EventsPipelineConfigError as exc:
        raise EventsPipelineError(str(exc)) from exc
    config_fp = events_pipeline_config_fingerprint(config)
    policy = load_events_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)

    sources = dict(sources or {})
    replay_contexts = list(replay_contexts or [])
    if not sources and not replay_contexts:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="EMPTY_INPUT",
            summary={"source_count": 0},
        )

    inputs = config.get("inputs") or {}
    replay_cfg = root / str(inputs.get("replay_config"))
    ledger_cfg = root / str(inputs.get("ledger_config"))
    metrics_cfg = root / str(inputs.get("metrics_config"))

    stage_13a = output_dir / "stage_13a"
    stage_13c = output_dir / "stage_13c"
    stage_13d = output_dir / "stage_13d"

    replay_parquet = None
    replays: list[dict[str, Any]] = []
    if replay_contexts:
        replay_res = compute_replay_candidates(
            output_dir=stage_13a,
            contexts=replay_contexts,
            config_path=replay_cfg,
            run_id=run_id,
            video_id=video_id,
            project_root=root,
        )
        if not replay_res.accepted:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code=f"REPLAY_FAILED:{replay_res.error_code}",
                summary=dict(replay_res.summary),
            )
        replays = [dict(r) for r in replay_res.replays]
        replay_parquet = replay_res.replay_parquet
        run_id = str(run_id or replay_res.summary.get("run_id"))
        video_id = str(video_id or replay_res.summary.get("video_id"))

    # 13B attack directions
    periods = list(attack_periods or [])
    if not periods:
        periods = [
            {
                "period_id": "period_1",
                "half_id": "first_half",
                "anonymous_team_id": "anon_team_a",
                "config_direction": "unknown",
            }
        ]
    run_id = run_id or "run_stage13_placeholder"
    video_id = video_id or "synthetic_video_13"
    # Prefer run/video from sources
    if sources:
        first = next(iter(sources.values()))[0]
        run_id = str(run_id or first.get("run_id"))
        video_id = str(video_id or first.get("video_id"))

    attack_dirs = resolve_match_attack_directions(
        run_id=str(run_id), video_id=str(video_id), periods=periods
    )
    attack_path = output_dir / "attack_direction_evidence.json"
    write_json_record(attack_path, {"periods": attack_dirs}, overwrite=True)
    # Manual override for metrics from first resolved period if available
    first_dir = str(attack_dirs[0].get("attack_direction")) if attack_dirs else "unknown"
    manual_for_metrics = first_dir if first_dir != "unknown" else None

    ledger_parquet = revisions_parquet = None
    ledger_rows: list[dict[str, Any]] = []
    revision_rows: list[dict[str, Any]] = []
    if sources:
        ledger_res = build_target_event_ledger(
            output_dir=stage_13c,
            sources=sources,
            config_path=ledger_cfg,
            run_id=run_id,
            video_id=video_id,
            project_root=root,
        )
        if not ledger_res.accepted:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code=f"LEDGER_FAILED:{ledger_res.error_code}",
                summary=dict(ledger_res.summary),
            )
        ledger_rows = [dict(r) for r in ledger_res.ledger]
        revision_rows = [dict(r) for r in ledger_res.revisions]
        ledger_parquet = ledger_res.ledger_parquet
        revisions_parquet = ledger_res.revisions_parquet

    validate_events_bundle(ledger=ledger_rows, revisions=revision_rows, replays=replays)

    metrics_json = None
    metrics_summary: dict[str, Any] = {}
    if ledger_rows:
        metrics_res = compute_target_event_metrics(
            output_dir=stage_13d,
            ledger=ledger_rows,
            config_path=metrics_cfg,
            run_id=run_id,
            video_id=video_id,
            project_root=root,
            attack_direction_manual=manual_for_metrics,
            interaction_coverage=interaction_coverage,
        )
        if not metrics_res.accepted:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code=f"METRICS_FAILED:{metrics_res.error_code}",
                summary=dict(metrics_res.summary),
            )
        metrics_json = metrics_res.metrics_json
        metrics_summary = dict(metrics_res.summary)

    evaluation = evaluate_events(ledger_rows=ledger_rows)
    eval_path = output_dir / "evaluation.json"
    write_json_record(eval_path, evaluation, overwrite=True)

    request = build_synthetic_request(run_id=str(run_id), video_id=str(video_id))
    receipt = build_synthetic_receipt(
        run_id=str(run_id),
        video_id=str(video_id),
        ledger_count=len(ledger_rows),
        revision_count=len(revision_rows),
        replay_count=len(replays),
    )
    quality = build_synthetic_quality(
        run_id=str(run_id),
        video_id=str(video_id),
        coverage={
            "never_invent_live_when_replay_uncertain": True,
            "append_only_ledger": True,
            "interaction_coverage": interaction_coverage,
            "metric_evaluable_count": metrics_summary.get("evaluable_count"),
        },
    )
    review = build_synthetic_review_queue(
        queue_id=f"review_{run_id}",
        run_id=str(run_id),
        video_id=str(video_id),
        entries=[
            {"kind": "attack_direction", "status": "unreviewed"}
            for d in attack_dirs
            if d.get("attack_direction") == "unknown" or d.get("conflict")
        ],
    )
    req_path = output_dir / "request.json"
    rec_path = output_dir / "receipt.json"
    qual_path = output_dir / "quality.json"
    rev_path = output_dir / "manual_review_queue.json"
    write_json_record(req_path, request, overwrite=True)
    write_json_record(rec_path, receipt, overwrite=True)
    write_json_record(qual_path, quality, overwrite=True)
    write_json_record(rev_path, review, overwrite=True)

    gate_hint = str(
        config.get("gate_hint")
        or (
            "PASS_WITH_FINDINGS — TARGET EVENTS PIPELINE ACTIVE; "
            "STAGE 13 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
        )
    )
    summary = {
        "run_id": run_id,
        "video_id": video_id,
        "stage": "13E",
        "ledger_count": len(ledger_rows),
        "revision_count": len(revision_rows),
        "replay_count": len(replays),
        "attack_periods": len(attack_dirs),
        "metrics_evaluable_count": metrics_summary.get("evaluable_count"),
        "metrics_not_evaluable_count": metrics_summary.get("not_evaluable_count"),
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "evaluation_status": evaluation["evaluation_status"],
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "gate_hint": gate_hint,
        "NOT_EVALUATED": NOT_EVALUATED,
        "created_at_utc": _utc_now(),
    }
    summary_path = output_dir / "summary.json"
    write_json_record(summary_path, summary, overwrite=True)
    return EventsPipelineResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        summary_json=str(summary_path),
        receipt_json=str(rec_path),
        quality_json=str(qual_path),
        evaluation_json=str(eval_path),
        review_queue_json=str(rev_path),
        replay_parquet=replay_parquet,
        ledger_parquet=ledger_parquet,
        revisions_parquet=revisions_parquet,
        metrics_json=metrics_json,
        summary=summary,
    )


__all__ = [
    "EventsPipelineError",
    "EventsPipelineResult",
    "integrate_target_events",
]
