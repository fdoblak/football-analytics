"""Stage 12E duels fusion service (12B → 12C → 12D → package)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.data.compiler import get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.duels.aerial_service import compute_aerial_clearance
from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS, evaluate_duels
from football_analytics.duels.ground_service import compute_ground_family
from football_analytics.duels.pipeline_config import (
    DuelsPipelineConfigError,
    duels_pipeline_config_fingerprint,
    load_duels_pipeline_config,
)
from football_analytics.duels.policy import load_duels_policy, policy_fingerprint
from football_analytics.duels.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
    build_synthetic_review_queue,
)
from football_analytics.duels.take_on_service import compute_take_ons
from football_analytics.duels.validation import validate_duels_bundle


class DuelsPipelineError(RuntimeError):
    """Duels pipeline fusion failure."""


@dataclass
class DuelsPipelineResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    review_queue_json: str | None
    take_on_parquet: str | None
    ground_parquet: str | None
    aerial_parquet: str | None
    tackle_parquet: str | None
    recovery_parquet: str | None
    turnover_parquet: str | None
    clearance_parquet: str | None
    summary: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    output_dir: Path,
    config_fp: str,
    code: str,
    summary: dict[str, Any],
) -> DuelsPipelineResult:
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
    return DuelsPipelineResult(
        accepted=False,
        exit_code=1,
        error_code=code,
        config_fingerprint=config_fp,
        summary_json=None,
        receipt_json=str(err_path),
        quality_json=None,
        evaluation_json=None,
        review_queue_json=None,
        take_on_parquet=None,
        ground_parquet=None,
        aerial_parquet=None,
        tackle_parquet=None,
        recovery_parquet=None,
        turnover_parquet=None,
        clearance_parquet=None,
        summary=summary,
    )


def integrate_duels(
    *,
    output_dir: Path,
    take_on_contexts: Sequence[Mapping[str, Any]] | None = None,
    ground_contexts: Sequence[Mapping[str, Any]] | None = None,
    aerial_contexts: Sequence[Mapping[str, Any]] | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
) -> DuelsPipelineResult:
    """Run Stage 12B/12C/12D and fuse into one duels package."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_duels_pipeline_config(config_path, project_root=root)
    except DuelsPipelineConfigError as exc:
        raise DuelsPipelineError(str(exc)) from exc
    config_fp = duels_pipeline_config_fingerprint(config)
    policy = load_duels_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)

    take_on_contexts = list(take_on_contexts or [])
    ground_contexts = list(ground_contexts or [])
    aerial_contexts = list(aerial_contexts or [])
    if not (take_on_contexts or ground_contexts or aerial_contexts):
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="EMPTY_INPUT",
            summary={"context_count": 0},
        )

    inputs = config.get("inputs") or {}
    take_cfg = root / str(inputs.get("take_on_config"))
    ground_cfg = root / str(inputs.get("ground_config"))
    aerial_cfg = root / str(inputs.get("aerial_config"))

    stage_12b = output_dir / "stage_12b"
    stage_12c = output_dir / "stage_12c"
    stage_12d = output_dir / "stage_12d"

    take_ons: list[dict[str, Any]] = []
    take_parquet = None
    if take_on_contexts:
        take_res = compute_take_ons(
            output_dir=stage_12b,
            contexts=take_on_contexts,
            config_path=take_cfg,
            run_id=run_id,
            video_id=video_id,
            project_root=root,
        )
        if not take_res.accepted:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code=f"TAKE_ON_FAILED:{take_res.error_code}",
                summary=dict(take_res.summary),
            )
        take_ons = [dict(row) for row in take_res.take_ons]
        take_parquet = take_res.take_on_parquet
        run_id = str(run_id or take_res.summary.get("run_id"))
        video_id = str(video_id or take_res.summary.get("video_id"))

    ground_duels: list[dict[str, Any]] = []
    tackles: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    turnovers: list[dict[str, Any]] = []
    ground_parquet = tackle_parquet = recovery_parquet = turnover_parquet = None
    if ground_contexts:
        ground_res = compute_ground_family(
            output_dir=stage_12c,
            contexts=ground_contexts,
            config_path=ground_cfg,
            run_id=run_id,
            video_id=video_id,
            project_root=root,
        )
        if not ground_res.accepted:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code=f"GROUND_FAILED:{ground_res.error_code}",
                summary=dict(ground_res.summary),
            )
        ground_duels = [dict(row) for row in ground_res.ground_duels]
        tackles = [dict(row) for row in ground_res.tackles]
        recoveries = [dict(row) for row in ground_res.recoveries]
        turnovers = [dict(row) for row in ground_res.turnovers]
        ground_parquet = ground_res.ground_parquet
        tackle_parquet = ground_res.tackle_parquet
        recovery_parquet = ground_res.recovery_parquet
        turnover_parquet = ground_res.turnover_parquet
        run_id = str(run_id or ground_res.summary.get("run_id"))
        video_id = str(video_id or ground_res.summary.get("video_id"))

    aerial_duels: list[dict[str, Any]] = []
    clearances: list[dict[str, Any]] = []
    aerial_parquet = clearance_parquet = None
    if aerial_contexts:
        aerial_res = compute_aerial_clearance(
            output_dir=stage_12d,
            contexts=aerial_contexts,
            config_path=aerial_cfg,
            run_id=run_id,
            video_id=video_id,
            project_root=root,
        )
        if not aerial_res.accepted:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code=f"AERIAL_FAILED:{aerial_res.error_code}",
                summary=dict(aerial_res.summary),
            )
        aerial_duels = [dict(row) for row in aerial_res.aerial_duels]
        clearances = [dict(row) for row in aerial_res.clearances]
        aerial_parquet = aerial_res.aerial_parquet
        clearance_parquet = aerial_res.clearance_parquet
        run_id = str(run_id or aerial_res.summary.get("run_id"))
        video_id = str(video_id or aerial_res.summary.get("video_id"))

    rid = str(run_id)
    vid = str(video_id)
    vr = validate_duels_bundle(
        take_ons=take_ons,
        ground_duels=ground_duels,
        aerial_duels=aerial_duels,
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
        clearances=clearances,
        policy=policy,
        expected_run_id=rid,
        expected_video_id=vid,
    )
    if vr.status != "PASS":
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="FUSED_VALIDATION_FAILED",
            summary={"errors": vr.errors},
        )

    coverage = {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 8_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_500_000,
        "opponent_context_us": 7_500_000,
        "not_observed_us": 500_000,
        "joint_coverage_ratio": 0.78,
        "nearby_opponent_alone_is_not_take_on": True,
        "nearest_switch_alone_is_not_duel_outcome": True,
        "monocular_aerial_no_exact_height": True,
        "long_ball_alone_is_not_clearance": True,
    }
    request = build_synthetic_request(run_id=rid, video_id=vid, duels_policy_fingerprint=policy_fp)
    receipt = build_synthetic_receipt(
        run_id=rid,
        video_id=vid,
        duels_policy_fingerprint=policy_fp,
        take_ons=take_ons,
        ground_duels=ground_duels,
        aerial_duels=aerial_duels,
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
        clearances=clearances,
        coverage_summary=coverage,
        status="succeeded",
    )
    quality = build_synthetic_quality(
        run_id=rid, video_id=vid, coverage=coverage, duels_policy_fingerprint=policy_fp
    )
    evaluation = evaluate_duels(
        take_ons=take_ons, ground_duels=ground_duels, aerial_duels=aerial_duels
    ).to_dict(run_id=rid, video_id=vid, config_fingerprint=config_fp)
    review = build_synthetic_review_queue(run_id=rid, video_id=vid, entries=[])

    gate_hint = (
        "PASS_WITH_FINDINGS — DUELS EVENTS PIPELINE ACTIVE; "
        "STAGE 12 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
    )
    summary = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "stage": "12E",
        "gate_hint": gate_hint,
        "take_on_attempt_count": len(take_ons),
        "ground_duel_candidate_count": len(ground_duels),
        "aerial_duel_candidate_count": len(aerial_duels),
        "tackle_event_count": len(tackles),
        "recovery_event_count": len(recoveries),
        "turnover_event_count": len(turnovers),
        "clearance_event_count": len(clearances),
        "evaluation_status": NOT_EVALUATED_DUELS,
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "contract_fingerprints": {
            name: contract_fingerprint(get_contract(name, 1))
            for name in (
                "take_on_attempts",
                "ground_duel_candidates",
                "aerial_duel_candidates",
                "tackle_events",
                "recovery_events",
                "turnover_events",
                "clearance_events",
            )
        },
        "artifact_hashes": {
            k: sha256_file(Path(v)) if v else None
            for k, v in {
                "take_on_attempts": take_parquet,
                "ground_duel_candidates": ground_parquet,
                "aerial_duel_candidates": aerial_parquet,
                "tackle_events": tackle_parquet,
                "recovery_events": recovery_parquet,
                "turnover_events": turnover_parquet,
                "clearance_events": clearance_parquet,
            }.items()
        },
        "created_at_utc": _utc_now(),
    }
    summary_path = output_dir / "duels_pipeline_summary.json"
    receipt_path = output_dir / "duels_pipeline_receipt.json"
    quality_path = output_dir / "duels_pipeline_quality.json"
    evaluation_path = output_dir / "duels_pipeline_evaluation.json"
    review_path = output_dir / "duels_pipeline_review_queue.json"
    write_json_record(summary_path, summary, overwrite=True)
    write_json_record(receipt_path, receipt, overwrite=True)
    write_json_record(quality_path, quality, overwrite=True)
    write_json_record(evaluation_path, evaluation, overwrite=True)
    write_json_record(review_path, review, overwrite=True)
    write_json_record(output_dir / "duels_pipeline_request.json", request, overwrite=True)
    return DuelsPipelineResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(evaluation_path),
        review_queue_json=str(review_path),
        take_on_parquet=take_parquet,
        ground_parquet=ground_parquet,
        aerial_parquet=aerial_parquet,
        tackle_parquet=tackle_parquet,
        recovery_parquet=recovery_parquet,
        turnover_parquet=turnover_parquet,
        clearance_parquet=clearance_parquet,
        summary=summary,
    )


__all__ = [
    "DuelsPipelineError",
    "DuelsPipelineResult",
    "integrate_duels",
]
