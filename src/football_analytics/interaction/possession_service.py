"""Stage 10C possession / control hypothesis computation service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.interaction.contact_candidates import extract_contact_candidates
from football_analytics.interaction.evaluation import (
    NOT_EVALUATED_INTERACTION,
    evaluate_human_ball_interaction,
)
from football_analytics.interaction.policy import (
    load_interaction_policy,
    policy_fingerprint,
)
from football_analytics.interaction.possession import build_possession_hypotheses
from football_analytics.interaction.possession_config import (
    PossessionConfigError,
    load_possession_baseline_config,
    possession_baseline_config_fingerprint,
)
from football_analytics.interaction.proximity import mark_nearest_and_build
from football_analytics.interaction.proximity_config import load_proximity_baseline_config
from football_analytics.interaction.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
)
from football_analytics.interaction.validation import validate_interaction_bundle


class PossessionServiceError(RuntimeError):
    """Possession/control baseline service failure."""


@dataclass
class PossessionServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    possession_parquet: str | None
    proximity_parquet: str | None
    contact_parquet: str | None
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    possessions: Sequence[Mapping[str, Any]]
    proximity: Sequence[Mapping[str, Any]]
    contacts: Sequence[Mapping[str, Any]]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "possession_parquet": self.possession_parquet,
            "proximity_parquet": self.proximity_parquet,
            "contact_parquet": self.contact_parquet,
            "summary_json": self.summary_json,
            "receipt_json": self.receipt_json,
            "quality_json": self.quality_json,
            "evaluation_json": self.evaluation_json,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def _fail(
    *,
    output_dir: Path,
    config_fp: str,
    code: str,
    summary: dict[str, Any],
) -> PossessionServiceResult:
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
    return PossessionServiceResult(
        accepted=False,
        exit_code=1,
        error_code=code,
        config_fingerprint=config_fp,
        possession_parquet=None,
        proximity_parquet=None,
        contact_parquet=None,
        summary_json=None,
        receipt_json=str(err_path),
        quality_json=None,
        evaluation_json=None,
        summary=summary,
        possessions=[],
        proximity=[],
        contacts=[],
    )


def compute_possession_control(
    *,
    output_dir: Path,
    points: Sequence[Mapping[str, Any]] | None = None,
    proximity_rows: Sequence[Mapping[str, Any]] | None = None,
    contact_rows: Sequence[Mapping[str, Any]] | None = None,
    config_path: Path | None = None,
    proximity_config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    write_upstream_tables: bool = True,
) -> PossessionServiceResult:
    """Build possession hypotheses from points or precomputed proximity/contacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = load_possession_baseline_config(config_path)
    except PossessionConfigError as exc:
        raise PossessionServiceError(str(exc)) from exc
    config_fp = possession_baseline_config_fingerprint(config)
    policy = load_interaction_policy()
    policy_fp = policy_fingerprint(policy)

    prox_rows: list[dict[str, Any]]
    cont_rows: list[dict[str, Any]]
    if proximity_rows is not None:
        prox_rows = [dict(r) for r in proximity_rows]
        cont_rows = [dict(r) for r in (contact_rows or [])]
        if contact_rows is None:
            cont_rows = extract_contact_candidates(
                prox_rows,
                config=load_proximity_baseline_config(proximity_config_path),
                policy_fingerprint=policy_fp,
            )
    else:
        if not points:
            return _fail(
                output_dir=output_dir,
                config_fp=config_fp,
                code="EMPTY_INPUT",
                summary={"point_count": 0},
            )
        prox_cfg = load_proximity_baseline_config(proximity_config_path)
        prox_rows = mark_nearest_and_build(points, config=prox_cfg, policy_fingerprint=policy_fp)
        cont_rows = extract_contact_candidates(
            prox_rows, config=prox_cfg, policy_fingerprint=policy_fp
        )

    rid = run_id or str(prox_rows[0]["run_id"])
    vid = video_id or str(prox_rows[0]["video_id"])

    possessions = build_possession_hypotheses(
        prox_rows, cont_rows, config=config, policy_fingerprint=policy_fp
    )

    vr = validate_interaction_bundle(
        proximity=prox_rows,
        contacts=cont_rows,
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

    poss_path = output_dir / "possession_hypotheses.parquet"
    write_contract_parquet(
        _cast("possession_hypotheses", possessions),
        poss_path,
        get_contract("possession_hypotheses", 1),
        overwrite=False,
    )
    prox_path = None
    contact_path = None
    if write_upstream_tables:
        prox_path = output_dir / "human_ball_proximity.parquet"
        contact_path = output_dir / "ball_contact_candidates.parquet"
        write_contract_parquet(
            _cast("human_ball_proximity", prox_rows),
            prox_path,
            get_contract("human_ball_proximity", 1),
            overwrite=False,
        )
        write_contract_parquet(
            _cast("ball_contact_candidates", cont_rows),
            contact_path,
            get_contract("ball_contact_candidates", 1),
            overwrite=False,
        )

    coverage = {
        "human_observed_us": sum(
            40_000 for r in prox_rows if str(r.get("human_observation_state")) == "observed"
        ),
        "ball_observed_us": sum(
            40_000 for r in prox_rows if str(r.get("ball_observation_state")) == "observed"
        ),
        "joint_observed_us": sum(
            40_000
            for r in prox_rows
            if str(r.get("human_observation_state")) == "observed"
            and str(r.get("ball_observation_state")) == "observed"
        ),
        "calibration_valid_us": sum(
            40_000 for r in prox_rows if str(r.get("calibration_status")) == "valid"
        ),
        "playable_us": sum(
            40_000 for r in prox_rows if str(r.get("playability_status")) == "playable"
        ),
        "target_confirmed_us": sum(
            40_000 for r in prox_rows if str(r.get("target_relationship")) == "confirmed_target"
        ),
        "ambiguous_ball_us": sum(
            40_000 for r in prox_rows if str(r.get("ball_candidate_status")) == "ambiguous"
        ),
        "contested_us": sum(
            int(h["end_time_us"]) - int(h["start_time_us"])
            for h in possessions
            if str(h.get("possession_state")) == "contested"
        ),
        "not_observed_us": sum(
            int(h["end_time_us"]) - int(h["start_time_us"])
            for h in possessions
            if str(h.get("possession_state")) == "not_evaluable"
            and "NOT_OBSERVED" in list(h.get("reason_codes") or [])
        ),
        "joint_coverage_ratio": (
            (
                sum(
                    1
                    for r in prox_rows
                    if str(r.get("human_observation_state")) == "observed"
                    and str(r.get("ball_observation_state")) == "observed"
                )
                / len(prox_rows)
            )
            if prox_rows
            else 0.0
        ),
        "low_joint_coverage_is_not_evaluable": True,
        "missing_ball_is_not_no_possession": True,
    }

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
        proximity=prox_rows,
        contacts=cont_rows,
        possessions=possessions,
        coverage_summary=coverage,
        status="succeeded",
    )
    receipt["input_fingerprints"]["baseline_config"] = config_fp
    receipt["output_fingerprints"]["possession_hypotheses"] = sha256_file(poss_path)
    if prox_path:
        receipt["output_fingerprints"]["human_ball_proximity"] = sha256_file(prox_path)
    if contact_path:
        receipt["output_fingerprints"]["ball_contact_candidates"] = sha256_file(contact_path)
    receipt["provenance"]["stage"] = "10C"
    receipt["provenance"]["label"] = "possession_control_baseline"
    quality = build_synthetic_quality(
        run_id=rid,
        video_id=vid,
        coverage=coverage,
        interaction_policy_fingerprint=policy_fp,
    )
    ev = evaluate_human_ball_interaction(
        proximity=prox_rows,
        contacts=cont_rows,
        possessions=possessions,
        has_reviewed_ground_truth=False,
    )
    eval_payload = ev.to_dict(run_id=rid, video_id=vid, config_fingerprint=config_fp)

    state_counts: dict[str, int] = {}
    for h in possessions:
        st = str(h.get("possession_state"))
        state_counts[st] = state_counts.get(st, 0) + 1

    summary = {
        "schema_version": 1,
        "stage": "10C",
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "proximity_row_count": len(prox_rows),
        "contact_candidate_count": len(cont_rows),
        "possession_hypothesis_count": len(possessions),
        "possession_state_counts": state_counts,
        "automatic_ceiling": "provisional",
        "nearest_implies_possession": False,
        "event_metrics_produced": False,
        "evaluation_status": NOT_EVALUATED_INTERACTION,
        "contract_fingerprints": {
            "possession_hypotheses": contract_fingerprint(get_contract("possession_hypotheses", 1)),
        },
        "created_at_utc": _utc_now(),
        "request": req,
    }

    summary_path = output_dir / "possession_control_summary.json"
    receipt_path = output_dir / "possession_control_receipt.json"
    quality_path = output_dir / "possession_control_quality.json"
    eval_path = output_dir / "possession_control_evaluation.json"
    write_json_record(summary_path, summary, overwrite=False)
    write_json_record(receipt_path, receipt, overwrite=False)
    write_json_record(quality_path, quality, overwrite=False)
    write_json_record(eval_path, eval_payload, overwrite=False)

    return PossessionServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        possession_parquet=str(poss_path),
        proximity_parquet=str(prox_path) if prox_path else None,
        contact_parquet=str(contact_path) if contact_path else None,
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(eval_path),
        summary=summary,
        possessions=possessions,
        proximity=prox_rows,
        contacts=cont_rows,
    )


__all__ = [
    "PossessionServiceError",
    "PossessionServiceResult",
    "compute_possession_control",
]
