"""Stage 10D human-ball interaction fusion service (10B → 10C → package)."""

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
from football_analytics.interaction.evaluation import (
    NOT_EVALUATED_INTERACTION,
    evaluate_human_ball_interaction,
)
from football_analytics.interaction.pipeline_config import (
    InteractionPipelineConfigError,
    interaction_pipeline_config_fingerprint,
    load_interaction_pipeline_config,
)
from football_analytics.interaction.policy import (
    load_interaction_policy,
    policy_fingerprint,
)
from football_analytics.interaction.possession_service import compute_possession_control
from football_analytics.interaction.proximity_service import compute_human_ball_proximity_contact
from football_analytics.interaction.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
    build_synthetic_review_queue,
)
from football_analytics.interaction.validation import validate_interaction_bundle


class InteractionPipelineError(RuntimeError):
    """Interaction pipeline fusion failure."""


@dataclass
class InteractionPipelineResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    review_queue_json: str | None
    proximity_parquet: str | None
    contact_parquet: str | None
    possession_parquet: str | None
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
            "proximity_parquet": self.proximity_parquet,
            "contact_parquet": self.contact_parquet,
            "possession_parquet": self.possession_parquet,
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
) -> InteractionPipelineResult:
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
    return InteractionPipelineResult(
        accepted=False,
        exit_code=1,
        error_code=code,
        config_fingerprint=config_fp,
        summary_json=None,
        receipt_json=str(err_path),
        quality_json=None,
        evaluation_json=None,
        review_queue_json=None,
        proximity_parquet=None,
        contact_parquet=None,
        possession_parquet=None,
        summary=summary,
    )


def integrate_human_ball_interaction(
    *,
    output_dir: Path,
    points: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    identity_status: str = "confirmed",
    project_root: Path | None = None,
) -> InteractionPipelineResult:
    """Run Stage 10B then 10C and fuse into one interaction package."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_interaction_pipeline_config(config_path, project_root=root)
    except InteractionPipelineConfigError as exc:
        raise InteractionPipelineError(str(exc)) from exc
    config_fp = interaction_pipeline_config_fingerprint(config)
    policy = load_interaction_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)

    if not points:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="EMPTY_INPUT",
            summary={"point_count": 0},
        )

    if identity_status == "revoked" and bool(
        (config.get("integrity") or {}).get("fail_on_source_mix", True)
    ):
        # Revoked target: still compute operational outputs but mark not_evaluable overall
        pass

    inputs = config.get("inputs") or {}
    prox_cfg_path = root / str(inputs.get("proximity_config"))
    poss_cfg_path = root / str(inputs.get("possession_config"))

    prox_dir = output_dir / "stage_10b"
    poss_dir = output_dir / "stage_10c"

    prox = compute_human_ball_proximity_contact(
        output_dir=prox_dir,
        points=points,
        config_path=prox_cfg_path,
        run_id=run_id,
        video_id=video_id,
    )
    if not prox.accepted:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code=f"PROXIMITY_FAILED:{prox.error_code}",
            summary=dict(prox.summary),
        )

    import pyarrow.parquet as pq

    proximity_rows = pq.read_table(prox.proximity_parquet).to_pylist()
    contact_rows = pq.read_table(prox.contact_parquet).to_pylist()

    poss = compute_possession_control(
        output_dir=poss_dir,
        proximity_rows=proximity_rows,
        contact_rows=contact_rows,
        config_path=poss_cfg_path,
        proximity_config_path=prox_cfg_path,
        run_id=run_id or prox.summary.get("run_id"),
        video_id=video_id or prox.summary.get("video_id"),
        write_upstream_tables=False,
    )
    if not poss.accepted:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code=f"POSSESSION_FAILED:{poss.error_code}",
            summary=dict(poss.summary),
        )

    rid = str(run_id or prox.summary.get("run_id"))
    vid = str(video_id or prox.summary.get("video_id"))
    possessions = list(poss.possessions)

    # Scope integrity
    for rows in (proximity_rows, contact_rows, possessions):
        for r in rows:
            if str(r.get("run_id")) != rid or str(r.get("video_id")) != vid:
                return _fail(
                    output_dir=output_dir,
                    config_fp=config_fp,
                    code="SOURCE_MIX",
                    summary={"run_id": rid, "video_id": vid, "row": dict(r)},
                )

    vr = validate_interaction_bundle(
        proximity=proximity_rows,
        contacts=contact_rows,
        possessions=possessions,
        policy=policy,
        expected_run_id=rid,
        expected_video_id=vid,
        event_metrics={
            "pass": False,
            "dribble": False,
            "duel": False,
            "aerial": False,
            "turnover": False,
            "box_touch": False,
        },
    )
    if vr.status != "PASS":
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="VALIDATION_FAILED",
            summary={"errors": list(vr.errors), "warnings": list(vr.warnings)},
        )

    coverage = {
        "human_observed_us": sum(
            40_000 for r in proximity_rows if str(r.get("human_observation_state")) == "observed"
        ),
        "ball_observed_us": sum(
            40_000 for r in proximity_rows if str(r.get("ball_observation_state")) == "observed"
        ),
        "joint_observed_us": sum(
            40_000
            for r in proximity_rows
            if str(r.get("human_observation_state")) == "observed"
            and str(r.get("ball_observation_state")) == "observed"
        ),
        "calibration_valid_us": sum(
            40_000 for r in proximity_rows if str(r.get("calibration_status")) == "valid"
        ),
        "playable_us": sum(
            40_000 for r in proximity_rows if str(r.get("playability_status")) == "playable"
        ),
        "target_confirmed_us": sum(
            40_000
            for r in proximity_rows
            if str(r.get("target_relationship")) == "confirmed_target"
        ),
        "ambiguous_ball_us": sum(
            40_000 for r in proximity_rows if str(r.get("ball_candidate_status")) == "ambiguous"
        ),
        "contested_us": sum(
            int(h["end_time_us"]) - int(h["start_time_us"])
            for h in possessions
            if str(h.get("possession_state")) == "contested"
        ),
        "not_observed_us": 0,
        "joint_coverage_ratio": (
            (
                sum(
                    1
                    for r in proximity_rows
                    if str(r.get("human_observation_state")) == "observed"
                    and str(r.get("ball_observation_state")) == "observed"
                )
                / len(proximity_rows)
            )
            if proximity_rows
            else 0.0
        ),
        "low_joint_coverage_is_not_evaluable": True,
        "missing_ball_is_not_no_possession": True,
    }

    overall_status = "succeeded"
    findings: list[str] = []
    if identity_status == "revoked":
        overall_status = "not_evaluable"
        findings.append("identity_revoked")
    if any(str(h.get("possession_state")) == "contested" for h in possessions):
        findings.append("contested_intervals_present")
    if any(str(h.get("possession_state")) == "not_evaluable" for h in possessions):
        findings.append("not_evaluable_intervals_present")

    req = build_synthetic_request(
        run_id=rid,
        video_id=vid,
        interaction_policy_fingerprint=policy_fp,
        output_root=str(config["runtime_root"]),
    )
    receipt = build_synthetic_receipt(
        run_id=rid,
        video_id=vid,
        interaction_policy_fingerprint=policy_fp,
        proximity=proximity_rows,
        contacts=contact_rows,
        possessions=possessions,
        coverage_summary=coverage,
        status=overall_status,
    )
    receipt["input_fingerprints"]["pipeline_config"] = config_fp
    receipt["input_fingerprints"]["proximity_config"] = prox.config_fingerprint
    receipt["input_fingerprints"]["possession_config"] = poss.config_fingerprint
    receipt["output_fingerprints"]["human_ball_proximity"] = (
        sha256_file(Path(prox.proximity_parquet)) if prox.proximity_parquet else None
    )
    receipt["output_fingerprints"]["ball_contact_candidates"] = (
        sha256_file(Path(prox.contact_parquet)) if prox.contact_parquet else None
    )
    receipt["output_fingerprints"]["possession_hypotheses"] = (
        sha256_file(Path(poss.possession_parquet)) if poss.possession_parquet else None
    )
    receipt["provenance"]["stage"] = "10D"
    receipt["provenance"]["label"] = "human_ball_interaction_pipeline"
    receipt["lineage"] = {
        "stage_10b_summary": prox.summary_json,
        "stage_10c_summary": poss.summary_json,
        "stage_10b_receipt": prox.receipt_json,
        "stage_10c_receipt": poss.receipt_json,
    }

    quality = build_synthetic_quality(
        run_id=rid,
        video_id=vid,
        coverage=coverage,
        interaction_policy_fingerprint=policy_fp,
        quality_flags=list(findings),
        not_evaluable_reasons=(["identity_revoked"] if identity_status == "revoked" else []),
    )

    review_entries = []
    for h in possessions:
        if str(h.get("possession_state")) in {"contested", "unknown"} or h.get(
            "manual_review_required"
        ):
            review_entries.append(
                {
                    "entry_id": f"review_{h['hypothesis_id']}",
                    "subject_type": "possession_hypothesis",
                    "subject_id": h["hypothesis_id"],
                    "reason": str(h.get("possession_state")),
                    "priority": "normal",
                    "status": "queued",
                }
            )
    review = build_synthetic_review_queue(run_id=rid, video_id=vid, entries=review_entries)

    ev = evaluate_human_ball_interaction(
        proximity=proximity_rows,
        contacts=contact_rows,
        possessions=possessions,
        has_reviewed_ground_truth=False,
    )
    eval_payload = ev.to_dict(run_id=rid, video_id=vid, config_fingerprint=config_fp)
    eval_payload["adapter_notes"] = "stage_10d_fusion_operational_not_event_accuracy"
    eval_payload["findings"] = list(eval_payload.get("findings") or []) + findings
    eval_payload["operational_vs_event_accuracy"] = {
        "operational_metrics_produced": True,
        "event_accuracy_validated": False,
        "real_football_accuracy_validated": False,
        "opta_claim": False,
    }

    poss_state_counts: dict[str, int] = {}
    for h in possessions:
        st = str(h.get("possession_state"))
        poss_state_counts[st] = poss_state_counts.get(st, 0) + 1

    summary = {
        "schema_version": 1,
        "stage": "10D",
        "stage_closed": "10",
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "proximity_config_fingerprint": prox.config_fingerprint,
        "possession_config_fingerprint": poss.config_fingerprint,
        "identity_status": identity_status,
        "overall_status": overall_status,
        "findings": findings,
        "proximity_row_count": len(proximity_rows),
        "contact_candidate_count": len(contact_rows),
        "possession_hypothesis_count": len(possessions),
        "possession_state_counts": poss_state_counts,
        "review_queue_count": len(review_entries),
        "nearest_implies_possession": False,
        "event_metrics_produced": False,
        "operational_metrics_produced": True,
        "event_accuracy_validated": False,
        "real_football_accuracy_validated": False,
        "evaluation_status": NOT_EVALUATED_INTERACTION,
        "contract_fingerprints": {
            "human_ball_proximity": contract_fingerprint(get_contract("human_ball_proximity", 1)),
            "ball_contact_candidates": contract_fingerprint(
                get_contract("ball_contact_candidates", 1)
            ),
            "possession_hypotheses": contract_fingerprint(get_contract("possession_hypotheses", 1)),
        },
        "created_at_utc": _utc_now(),
        "request": req,
        "gate_hint": (
            "PASS_WITH_FINDINGS — HUMAN BALL INTERACTION PIPELINE ACTIVE; "
            "STAGE 10 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
        ),
    }

    summary_path = output_dir / "interaction_pipeline_summary.json"
    receipt_path = output_dir / "interaction_pipeline_receipt.json"
    quality_path = output_dir / "interaction_pipeline_quality.json"
    eval_path = output_dir / "interaction_pipeline_evaluation.json"
    review_path = output_dir / "interaction_pipeline_review_queue.json"
    write_json_record(summary_path, summary, overwrite=False)
    write_json_record(receipt_path, receipt, overwrite=False)
    write_json_record(quality_path, quality, overwrite=False)
    write_json_record(eval_path, eval_payload, overwrite=False)
    write_json_record(review_path, review, overwrite=False)

    return InteractionPipelineResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(eval_path),
        review_queue_json=str(review_path),
        proximity_parquet=prox.proximity_parquet,
        contact_parquet=prox.contact_parquet,
        possession_parquet=poss.possession_parquet,
        summary=summary,
    )


__all__ = [
    "InteractionPipelineError",
    "InteractionPipelineResult",
    "integrate_human_ball_interaction",
]
