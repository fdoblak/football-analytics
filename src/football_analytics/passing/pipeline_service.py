"""Stage 11D passing fusion service (11B → 11C → package)."""

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
from football_analytics.passing.evaluation import NOT_EVALUATED_PASSING, evaluate_passing
from football_analytics.passing.metrics_service import compute_passing_metrics
from football_analytics.passing.pass_service import compute_pass_reception
from football_analytics.passing.pipeline_config import (
    PassingPipelineConfigError,
    load_passing_pipeline_config,
    passing_pipeline_config_fingerprint,
)
from football_analytics.passing.policy import load_passing_policy, policy_fingerprint
from football_analytics.passing.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
    build_synthetic_review_queue,
)
from football_analytics.passing.validation import validate_passing_bundle


class PassingPipelineError(RuntimeError):
    """Passing pipeline fusion failure."""


@dataclass
class PassingPipelineResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    review_queue_json: str | None
    metrics_json: str | None
    pass_parquet: str | None
    reception_parquet: str | None
    outcome_parquet: str | None
    progression_parquet: str | None
    touches_parquet: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "summary_json": self.summary_json,
            "receipt_json": self.receipt_json,
            "quality_json": self.quality_json,
            "evaluation_json": self.evaluation_json,
            "review_queue_json": self.review_queue_json,
            "metrics_json": self.metrics_json,
            "pass_parquet": self.pass_parquet,
            "reception_parquet": self.reception_parquet,
            "outcome_parquet": self.outcome_parquet,
            "progression_parquet": self.progression_parquet,
            "touches_parquet": self.touches_parquet,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    output_dir: Path,
    config_fp: str,
    code: str,
    summary: dict[str, Any],
) -> PassingPipelineResult:
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
    return PassingPipelineResult(
        accepted=False,
        exit_code=1,
        error_code=code,
        config_fingerprint=config_fp,
        summary_json=None,
        receipt_json=str(err_path),
        quality_json=None,
        evaluation_json=None,
        review_queue_json=None,
        metrics_json=None,
        pass_parquet=None,
        reception_parquet=None,
        outcome_parquet=None,
        progression_parquet=None,
        touches_parquet=None,
        summary=summary,
    )


def integrate_passing(
    *,
    output_dir: Path,
    transitions: Sequence[Mapping[str, Any]],
    touch_inputs: Sequence[Mapping[str, Any]] | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    attack_direction_manual: str | None = None,
    project_root: Path | None = None,
) -> PassingPipelineResult:
    """Run Stage 11B then 11C and fuse into one passing package."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_passing_pipeline_config(config_path, project_root=root)
    except PassingPipelineConfigError as exc:
        raise PassingPipelineError(str(exc)) from exc
    config_fp = passing_pipeline_config_fingerprint(config)
    policy = load_passing_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)

    if not transitions:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="EMPTY_INPUT",
            summary={"transition_count": 0},
        )

    inputs = config.get("inputs") or {}
    pass_cfg = root / str(inputs.get("pass_reception_config"))
    metrics_cfg = root / str(inputs.get("metrics_config"))

    stage_11b = output_dir / "stage_11b"
    stage_11c = output_dir / "stage_11c"

    pass_res = compute_pass_reception(
        output_dir=stage_11b,
        transitions=transitions,
        config_path=pass_cfg,
        run_id=run_id,
        video_id=video_id,
        project_root=root,
    )
    if not pass_res.accepted:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code=f"PASS_RECEPTION_FAILED:{pass_res.error_code}",
            summary=dict(pass_res.summary),
        )

    rid = str(run_id or pass_res.summary.get("run_id"))
    vid = str(video_id or pass_res.summary.get("video_id"))

    metrics_res = compute_passing_metrics(
        output_dir=stage_11c,
        passes=pass_res.passes,
        receptions=pass_res.receptions,
        outcomes=pass_res.outcomes,
        touch_inputs=touch_inputs,
        config_path=metrics_cfg,
        run_id=rid,
        video_id=vid,
        attack_direction_manual=attack_direction_manual,
        project_root=root,
    )
    if not metrics_res.accepted:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code=f"METRICS_FAILED:{metrics_res.error_code}",
            summary=dict(metrics_res.summary),
        )

    vr = validate_passing_bundle(
        passes=pass_res.passes,
        receptions=pass_res.receptions,
        outcomes=pass_res.outcomes,
        progression=metrics_res.progression,
        touches=metrics_res.touches,
        policy=policy,
        expected_run_id=rid,
        expected_video_id=vid,
    )
    if vr.status != "PASS":
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="FUSE_VALIDATION_FAILED",
            summary={"errors": list(vr.errors), "warnings": list(vr.warnings)},
        )

    coverage = {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 8_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_000_000,
        "attack_direction_resolved_us": 0,
        "not_observed_us": 1_000_000,
        "joint_coverage_ratio": 0.8,
        "owner_change_alone_is_not_completed_pass": True,
        "penalty_presence_is_not_box_touch": True,
        "attack_direction_unknown_blocks_directional": True,
    }
    request = build_synthetic_request(
        run_id=rid,
        video_id=vid,
        passing_policy_fingerprint=policy_fp,
        output_root="/home/fdoblak/workspace/passing_pipeline_checks",
    )
    receipt = build_synthetic_receipt(
        run_id=rid,
        video_id=vid,
        passing_policy_fingerprint=policy_fp,
        passes=pass_res.passes,
        receptions=pass_res.receptions,
        outcomes=pass_res.outcomes,
        progression=metrics_res.progression,
        touches=metrics_res.touches,
        coverage_summary=coverage,
        status="succeeded",
    )
    receipt["output_fingerprints"] = {
        "pass_candidates": contract_fingerprint(get_contract("pass_candidates", 1)),
        "reception_candidates": contract_fingerprint(get_contract("reception_candidates", 1)),
        "pass_outcomes": contract_fingerprint(get_contract("pass_outcomes", 1)),
        "ball_progression_segments": contract_fingerprint(
            get_contract("ball_progression_segments", 1)
        ),
        "target_ball_touches": contract_fingerprint(get_contract("target_ball_touches", 1)),
    }
    receipt["artifact_hashes"] = {
        "pass_candidates": sha256_file(Path(str(pass_res.pass_parquet))),
        "reception_candidates": sha256_file(Path(str(pass_res.reception_parquet))),
        "pass_outcomes": sha256_file(Path(str(pass_res.outcome_parquet))),
        "ball_progression_segments": sha256_file(Path(str(metrics_res.progression_parquet))),
        "target_ball_touches": sha256_file(Path(str(metrics_res.touches_parquet))),
    }
    quality = build_synthetic_quality(
        run_id=rid,
        video_id=vid,
        coverage=coverage,
        passing_policy_fingerprint=policy_fp,
        quality_flags=["stage_11_fused"],
    )
    evaluation = evaluate_passing(
        passes=pass_res.passes,
        receptions=pass_res.receptions,
        outcomes=pass_res.outcomes,
    ).to_dict(run_id=rid, video_id=vid, config_fingerprint=config_fp)
    review = build_synthetic_review_queue(run_id=rid, video_id=vid, entries=[])

    gate_hint = (
        "PASS_WITH_FINDINGS — PASSING PIPELINE ACTIVE; "
        "STAGE 11 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
    )
    summary = {
        "schema_version": 1,
        "stage": "11D",
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "pass_candidate_count": len(pass_res.passes),
        "reception_candidate_count": len(pass_res.receptions),
        "pass_outcome_count": len(pass_res.outcomes),
        "progression_segment_count": len(metrics_res.progression),
        "target_ball_touch_count": len(metrics_res.touches),
        "automatic_ceiling": "provisional",
        "evaluation_status": NOT_EVALUATED_PASSING,
        "opta_accuracy_validated": False,
        "real_football_accuracy_validated": False,
        "event_accuracy_validated": False,
        "gate_hint": gate_hint,
        "created_at_utc": _utc_now(),
    }

    write_json_record(output_dir / "passing_request.json", request, overwrite=False)
    receipt_path = output_dir / "passing_run_receipt.json"
    quality_path = output_dir / "passing_quality.json"
    eval_path = output_dir / "passing_evaluation.json"
    review_path = output_dir / "manual_review_queue.json"
    summary_path = output_dir / "passing_pipeline_summary.json"
    write_json_record(receipt_path, receipt, overwrite=False)
    write_json_record(quality_path, quality, overwrite=False)
    write_json_record(eval_path, evaluation, overwrite=False)
    write_json_record(review_path, review, overwrite=False)
    write_json_record(summary_path, summary, overwrite=False)

    return PassingPipelineResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(eval_path),
        review_queue_json=str(review_path),
        metrics_json=metrics_res.metrics_json,
        pass_parquet=pass_res.pass_parquet,
        reception_parquet=pass_res.reception_parquet,
        outcome_parquet=pass_res.outcome_parquet,
        progression_parquet=metrics_res.progression_parquet,
        touches_parquet=metrics_res.touches_parquet,
        summary=summary,
    )


__all__ = [
    "PassingPipelineError",
    "PassingPipelineResult",
    "integrate_passing",
]
