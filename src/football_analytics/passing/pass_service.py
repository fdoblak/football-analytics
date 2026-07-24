"""Stage 11B synthetic pass/reception/outcome computation from possession transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import default_project_root
from football_analytics.passing.evaluation import (
    NOT_EVALUATED_PASSING,
    evaluate_passing,
)
from football_analytics.passing.pass_config import (
    PassConfigError,
    load_pass_reception_config,
    pass_reception_config_fingerprint,
)
from football_analytics.passing.policy import load_passing_policy, policy_fingerprint
from football_analytics.passing.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
)
from football_analytics.passing.semantics import (
    cut_replay_gap_allows_pass,
    neutral_transition,
    neutral_zone_from_x,
    owner_change_alone_is_completed_pass,
)
from football_analytics.passing.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN
from football_analytics.passing.validation import validate_passing_bundle


class PassServiceError(RuntimeError):
    """Pass/reception baseline service failure."""


@dataclass
class PassServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    pass_parquet: str | None
    reception_parquet: str | None
    outcome_parquet: str | None
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    passes: Sequence[Mapping[str, Any]]
    receptions: Sequence[Mapping[str, Any]]
    outcomes: Sequence[Mapping[str, Any]]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "pass_parquet": self.pass_parquet,
            "reception_parquet": self.reception_parquet,
            "outcome_parquet": self.outcome_parquet,
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
) -> PassServiceResult:
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
    return PassServiceResult(
        accepted=False,
        exit_code=1,
        error_code=code,
        config_fingerprint=config_fp,
        pass_parquet=None,
        reception_parquet=None,
        outcome_parquet=None,
        summary_json=None,
        receipt_json=str(err_path),
        quality_json=None,
        evaluation_json=None,
        summary=summary,
        passes=[],
        receptions=[],
        outcomes=[],
    )


def _dist(x0: float | None, y0: float | None, x1: float | None, y1: float | None) -> float | None:
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    d = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return d if math.isfinite(d) else None


def build_pass_reception_from_transitions(
    *,
    transitions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit pass/reception/outcome rows from synthetic possession transitions."""
    thr = dict(config.get("thresholds") or {})
    long_thr = float(thr.get("long_pass_distance_m", 30.0))
    min_dist = float(thr.get("min_pass_distance_m", 1.0))
    recv_window = int(thr.get("reception_window_us", 2_000_000))
    target = dict(config.get("target") or {})
    target_track = int(target.get("track_id", 1))
    target_rel = str(target.get("relationship", "confirmed_target"))

    passes: list[dict[str, Any]] = []
    receptions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    for i, tr in enumerate(transitions):
        idx = i + 1
        cut = bool(tr.get("cut_or_replay"))
        gap = bool(tr.get("hard_gap"))
        owner_alone = bool(tr.get("owner_change_alone", False))
        from_owner = tr.get("from_owner_track_id")
        to_owner = tr.get("to_owner_track_id")
        start_us = int(tr.get("start_time_us", idx * 1_000_000))
        end_us = int(tr.get("end_time_us", start_us + 1_000_000))
        sx = tr.get("start_x_m")
        sy = tr.get("start_y_m")
        ex = tr.get("end_x_m")
        ey = tr.get("end_y_m")
        dist = _dist(sx, sy, ex, ey)
        start_zone = str(tr.get("start_zone_neutral") or neutral_zone_from_x(x_m=sx))
        end_zone = str(tr.get("end_zone_neutral") or neutral_zone_from_x(x_m=ex))
        contact_ids = list(tr.get("contact_candidate_ids") or [])
        evidence = list(tr.get("evidence_refs") or [])
        playability = str(tr.get("playability_status", "playable"))
        calibration = str(tr.get("calibration_status", "valid"))

        pass_id = f"pass_{idx:02d}"
        allowed = cut_replay_gap_allows_pass(cut_or_replay=cut, hard_gap=gap)
        if owner_alone and not evidence:
            state = "not_evaluable"
            reasons = ["OWNER_CHANGE_ALONE_NOT_PASS"]
        elif not allowed:
            state = "rejected"
            reasons = ["CUT_REPLAY_GAP_NO_PASS"]
        elif dist is not None and dist < min_dist:
            state = "rejected"
            reasons = ["below_min_pass_distance"]
        else:
            state = "provisional"
            reasons = []

        passer_rel = (
            target_rel
            if from_owner == target_track
            else str(tr.get("passer_relationship", "non_target"))
        )
        pass_row = {
            "run_id": run_id,
            "video_id": video_id,
            "pass_candidate_id": pass_id,
            "passer_human_track_id": int(from_owner if from_owner is not None else target_track),
            "intended_receiver_track_id": int(to_owner) if to_owner is not None else None,
            "passer_team_id": tr.get("passer_team_id", "team_a"),
            "start_time_us": start_us,
            "end_time_us": end_us,
            "start_x_m": float(sx) if sx is not None else None,
            "start_y_m": float(sy) if sy is not None else None,
            "end_x_m": float(ex) if ex is not None else None,
            "end_y_m": float(ey) if ey is not None else None,
            "pass_distance_m": float(dist) if dist is not None else None,
            "start_zone_neutral": start_zone,
            "end_zone_neutral": end_zone,
            "candidate_state": state,
            "target_relationship": passer_rel,
            "possession_hypothesis_id": tr.get("possession_hypothesis_id"),
            "contact_candidate_ids": contact_ids,
            "evidence_refs": evidence,
            "cut_or_replay": cut,
            "hard_gap": gap,
            "owner_change_alone": owner_alone,
            "implies_completed_pass": (
                False if owner_change_alone_is_completed_pass(owner_changed=owner_alone) else False
            ),
            "playability_status": playability,
            "calibration_status": calibration,
            "automatic_ceiling": "provisional",
            "review_status": "unreviewed",
            "manual_review_required": False,
            "uncertainty": float(tr.get("uncertainty", 0.4)),
            "reason_codes": reasons,
            "quality_flags": [],
            "metric_origin": METRIC_ORIGIN,
            "definition_style": DEFINITION_STYLE,
            "policy_fingerprint": policy_fp,
            "provenance_json": None,
            "contract_version": CONTRACT_VERSION,
        }
        passes.append(pass_row)

        recv_id = None
        if state == "provisional" and to_owner is not None and contact_ids:
            recv_id = f"recv_{idx:02d}"
            receptions.append(
                {
                    "run_id": run_id,
                    "video_id": video_id,
                    "reception_candidate_id": recv_id,
                    "receiver_human_track_id": int(to_owner),
                    "source_pass_candidate_id": pass_id,
                    "receiver_team_id": tr.get("receiver_team_id", "team_a"),
                    "start_time_us": end_us,
                    "end_time_us": end_us + recv_window,
                    "reception_x_m": float(ex) if ex is not None else None,
                    "reception_y_m": float(ey) if ey is not None else None,
                    "zone_neutral": end_zone,
                    "candidate_state": "provisional",
                    "target_relationship": (
                        target_rel if to_owner == target_track else "non_target"
                    ),
                    "possession_hypothesis_id": tr.get("to_possession_hypothesis_id"),
                    "contact_candidate_ids": contact_ids,
                    "evidence_refs": evidence,
                    "cut_or_replay": cut,
                    "hard_gap": gap,
                    "implies_completed_pass": False,
                    "playability_status": playability,
                    "calibration_status": calibration,
                    "automatic_ceiling": "provisional",
                    "review_status": "unreviewed",
                    "manual_review_required": False,
                    "uncertainty": float(tr.get("uncertainty", 0.35)),
                    "reason_codes": [],
                    "quality_flags": [],
                    "metric_origin": METRIC_ORIGIN,
                    "definition_style": DEFINITION_STYLE,
                    "policy_fingerprint": policy_fp,
                    "provenance_json": None,
                    "contract_version": CONTRACT_VERSION,
                }
            )

        if state == "provisional" and recv_id and bool(tr.get("same_team", True)):
            outcome = "completed"
        elif state == "provisional" and not recv_id:
            outcome = "incomplete"
        elif state in {"rejected", "not_evaluable"}:
            outcome = "not_evaluable"
        else:
            outcome = "uncertain"

        is_long = bool(dist is not None and dist >= long_thr and calibration == "valid")
        outcomes.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "outcome_id": f"out_{idx:02d}",
                "pass_candidate_id": pass_id,
                "reception_candidate_id": recv_id,
                "outcome": outcome,
                "outcome_state": (
                    "provisional" if outcome in {"completed", "incomplete", "uncertain"} else state
                ),
                "passer_is_target": int(from_owner if from_owner is not None else -1)
                == target_track,
                "receiver_is_teammate": bool(tr.get("same_team")) if to_owner is not None else None,
                "is_long_pass": is_long,
                "pass_distance_m": float(dist) if dist is not None else None,
                "long_pass_threshold_m": long_thr,
                "start_zone_neutral": start_zone,
                "end_zone_neutral": end_zone,
                "attack_relative_evaluable": False,
                "progression_1_to_2": "not_evaluable",
                "progression_2_to_3": "not_evaluable",
                "cut_or_replay": cut,
                "hard_gap": gap,
                "owner_change_alone": owner_alone,
                "automatic_ceiling": "provisional",
                "review_status": "unreviewed",
                "manual_review_required": False,
                "uncertainty": float(tr.get("uncertainty", 0.4)),
                "evidence_refs": evidence,
                "reason_codes": list(reasons),
                "quality_flags": [f"neutral_transition:{neutral_transition(start_zone, end_zone)}"],
                "metric_origin": METRIC_ORIGIN,
                "definition_style": DEFINITION_STYLE,
                "policy_fingerprint": policy_fp,
                "provenance_json": None,
                "contract_version": CONTRACT_VERSION,
            }
        )

    return passes, receptions, outcomes


def compute_pass_reception(
    *,
    output_dir: Path,
    transitions: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
) -> PassServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_pass_reception_config(config_path, project_root=root)
    except PassConfigError as exc:
        raise PassServiceError(str(exc)) from exc
    config_fp = pass_reception_config_fingerprint(config)
    policy = load_passing_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)

    if not transitions:
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="EMPTY_INPUT",
            summary={"transition_count": 0},
        )

    rid = run_id or generate_run_id()
    vid = video_id or str(transitions[0].get("video_id", "video_synth_01"))
    passes, receptions, outcomes = build_pass_reception_from_transitions(
        transitions=transitions,
        config=config,
        policy_fp=policy_fp,
        run_id=rid,
        video_id=vid,
    )

    vr = validate_passing_bundle(
        passes=passes,
        receptions=receptions,
        outcomes=outcomes,
        progression=[],
        touches=[],
        policy=policy,
        expected_run_id=rid,
        expected_video_id=vid,
    )
    if vr.status != "PASS":
        return _fail(
            output_dir=output_dir,
            config_fp=config_fp,
            code="VALIDATION_FAILED",
            summary={"errors": list(vr.errors), "warnings": list(vr.warnings)},
        )

    pass_tbl = _cast("pass_candidates", passes)
    recv_tbl = _cast("reception_candidates", receptions)
    out_tbl = _cast("pass_outcomes", outcomes)
    pass_path = output_dir / "pass_candidates.parquet"
    recv_path = output_dir / "reception_candidates.parquet"
    out_path = output_dir / "pass_outcomes.parquet"
    write_contract_parquet(pass_tbl, pass_path, get_contract("pass_candidates", 1), overwrite=False)
    write_contract_parquet(
        recv_tbl, recv_path, get_contract("reception_candidates", 1), overwrite=False
    )
    write_contract_parquet(out_tbl, out_path, get_contract("pass_outcomes", 1), overwrite=False)

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
        run_id=rid, video_id=vid, passing_policy_fingerprint=policy_fp
    )
    receipt = build_synthetic_receipt(
        run_id=rid,
        video_id=vid,
        passing_policy_fingerprint=policy_fp,
        passes=passes,
        receptions=receptions,
        outcomes=outcomes,
        progression=[],
        touches=[],
        coverage_summary=coverage,
        status="succeeded",
    )
    receipt["output_fingerprints"] = {
        "pass_candidates": contract_fingerprint(get_contract("pass_candidates", 1)),
        "reception_candidates": contract_fingerprint(get_contract("reception_candidates", 1)),
        "pass_outcomes": contract_fingerprint(get_contract("pass_outcomes", 1)),
        "ball_progression_segments": None,
        "target_ball_touches": None,
    }
    receipt["artifact_hashes"] = {
        "pass_candidates": sha256_file(pass_path),
        "reception_candidates": sha256_file(recv_path),
        "pass_outcomes": sha256_file(out_path),
    }
    quality = build_synthetic_quality(
        run_id=rid,
        video_id=vid,
        coverage=coverage,
        passing_policy_fingerprint=policy_fp,
    )
    evaluation = evaluate_passing(passes=passes, receptions=receptions, outcomes=outcomes).to_dict(
        run_id=rid, video_id=vid, config_fingerprint=config_fp
    )

    req_path = output_dir / "passing_request.json"
    receipt_path = output_dir / "passing_run_receipt.json"
    quality_path = output_dir / "passing_quality.json"
    eval_path = output_dir / "passing_evaluation.json"
    summary_path = output_dir / "pass_reception_summary.json"
    write_json_record(req_path, request, overwrite=False)
    write_json_record(receipt_path, receipt, overwrite=False)
    write_json_record(quality_path, quality, overwrite=False)
    write_json_record(eval_path, evaluation, overwrite=False)

    summary = {
        "schema_version": 1,
        "stage": "11B",
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "pass_candidate_count": len(passes),
        "reception_candidate_count": len(receptions),
        "pass_outcome_count": len(outcomes),
        "automatic_ceiling": "provisional",
        "evaluation_status": NOT_EVALUATED_PASSING,
        "opta_accuracy_validated": False,
        "real_football_accuracy_validated": False,
        "created_at_utc": _utc_now(),
    }
    write_json_record(summary_path, summary, overwrite=False)

    return PassServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        pass_parquet=str(pass_path),
        reception_parquet=str(recv_path),
        outcome_parquet=str(out_path),
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(eval_path),
        summary=summary,
        passes=passes,
        receptions=receptions,
        outcomes=outcomes,
    )


__all__ = [
    "PassServiceError",
    "PassServiceResult",
    "build_pass_reception_from_transitions",
    "compute_pass_reception",
]
