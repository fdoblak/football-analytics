"""Stage 12D synthetic aerial duel / clearance computation."""

from __future__ import annotations

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
from football_analytics.duels.aerial_config import (
    AerialConfigError,
    aerial_config_fingerprint,
    load_aerial_config,
)
from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS, evaluate_duels
from football_analytics.duels.policy import load_duels_policy, policy_fingerprint
from football_analytics.duels.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
)
from football_analytics.duels.semantics import (
    cut_replay_gap_allows_event,
    long_ball_alone_is_clearance,
    monocular_aerial_evaluability,
    neutral_zone_from_x,
)
from football_analytics.duels.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN
from football_analytics.duels.validation import validate_duels_bundle


class AerialServiceError(RuntimeError):
    """Aerial/clearance baseline service failure."""


@dataclass
class AerialServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    aerial_parquet: str | None
    clearance_parquet: str | None
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    aerial_duels: Sequence[Mapping[str, Any]]
    clearances: Sequence[Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def build_aerial_clearance_from_contexts(
    *,
    contexts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = dict(config.get("target") or {})
    target_rel = str(target.get("relationship", "confirmed_target"))
    aerial: list[dict[str, Any]] = []
    clearances: list[dict[str, Any]] = []
    a_i = c_i = 0
    for ctx in contexts:
        kind = str(ctx.get("kind", "aerial"))
        cut = bool(ctx.get("cut_or_replay"))
        gap = bool(ctx.get("hard_gap"))
        allowed = cut_replay_gap_allows_event(cut_or_replay=cut, hard_gap=gap)
        x_m = ctx.get("x_m")
        base = {
            "run_id": run_id,
            "video_id": video_id,
            "target_human_track_id": int(ctx.get("target_track_id", 1)),
            "opponent_human_track_id": (
                int(ctx["opponent_track_id"]) if ctx.get("opponent_track_id") is not None else None
            ),
            "start_time_us": int(ctx.get("start_time_us", 0)),
            "end_time_us": int(ctx.get("end_time_us", 0)),
            "x_m": float(x_m) if x_m is not None else None,
            "y_m": float(ctx["y_m"]) if ctx.get("y_m") is not None else None,
            "zone_neutral": str(ctx.get("zone_neutral") or neutral_zone_from_x(x_m=x_m)),
            "target_relationship": target_rel,
            "possession_hypothesis_id": ctx.get("possession_hypothesis_id"),
            "contact_candidate_ids": list(ctx.get("contact_candidate_ids") or []),
            "evidence_refs": list(ctx.get("evidence_refs") or []),
            "cut_or_replay": cut,
            "hard_gap": gap,
            "playability_status": str(ctx.get("playability_status", "playable")),
            "calibration_status": str(ctx.get("calibration_status", "valid")),
            "automatic_ceiling": "provisional",
            "review_status": "unreviewed",
            "manual_review_required": False,
            "uncertainty": float(ctx.get("uncertainty", 0.45)),
            "quality_flags": [],
            "metric_origin": METRIC_ORIGIN,
            "definition_style": DEFINITION_STYLE,
            "policy_fingerprint": policy_fp,
            "provenance_json": None,
            "contract_version": CONTRACT_VERSION,
        }
        if kind == "aerial":
            a_i += 1
            monocular = bool(ctx.get("monocular_only", True))
            reasons = ["MONOCULAR_AERIAL_NO_EXACT_HEIGHT"] if monocular else []
            if not allowed:
                state = "rejected"
                outcome = "rejected"
                reasons = ["CUT_REPLAY_GAP_NO_EVENT"]
                eval_status = "not_evaluable"
            else:
                eval_status = monocular_aerial_evaluability(monocular_only=monocular)
                if monocular:
                    state = eval_status if eval_status != "provisional" else "candidate"
                    outcome = "uncertain" if state == "candidate" else "not_evaluable"
                else:
                    state = "provisional"
                    outcome = str(ctx.get("outcome", "uncertain"))
            row = dict(base)
            row.update(
                {
                    "aerial_duel_candidate_id": f"aduels_{a_i:02d}",
                    "event_state": state,
                    "monocular_only": monocular,
                    "exact_3d_height_claimed": False,
                    "exact_3d_height_m": None,
                    "aerial_evaluability": eval_status if allowed else "not_evaluable",
                    "implies_aerial_outcome": False,
                    "outcome": outcome,
                    "reason_codes": reasons,
                }
            )
            aerial.append(row)
        elif kind == "clearance":
            c_i += 1
            long_alone = bool(ctx.get("long_ball_alone", False))
            reasons = []
            if long_alone or long_ball_alone_is_clearance(long_ball_alone=long_alone):
                state = "not_evaluable"
                implies = False
                outcome = "not_evaluable"
                reasons.append("LONG_BALL_ALONE_NOT_CLEARANCE")
            elif not allowed:
                state = "rejected"
                implies = False
                outcome = "rejected"
                reasons.append("CUT_REPLAY_GAP_NO_EVENT")
            elif not ctx.get("defensive_context") and not ctx.get("evidence_refs"):
                state = "not_evaluable"
                implies = False
                outcome = "not_evaluable"
                reasons.append("insufficient_clearance_evidence")
            else:
                state = "provisional"
                implies = True
                outcome = "cleared"
            row = dict(base)
            row.update(
                {
                    "clearance_event_id": f"clr_{c_i:02d}",
                    "event_state": state,
                    "long_ball_alone": long_alone,
                    "ball_distance_m": (
                        float(ctx["ball_distance_m"])
                        if ctx.get("ball_distance_m") is not None
                        else None
                    ),
                    "implies_clearance": implies,
                    "outcome": outcome,
                    "reason_codes": reasons,
                }
            )
            clearances.append(row)
    return aerial, clearances


def compute_aerial_clearance(
    *,
    output_dir: Path,
    contexts: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
) -> AerialServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_aerial_config(config_path, project_root=root)
    except AerialConfigError as exc:
        raise AerialServiceError(str(exc)) from exc
    config_fp = aerial_config_fingerprint(config)
    policy = load_duels_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)
    rid = run_id or generate_run_id()
    vid = video_id or "video_synth_01"
    if not contexts:
        err = output_dir / "failure_receipt.json"
        write_json_record(
            err,
            {
                "schema_version": 1,
                "status": "failed",
                "error_code": "EMPTY_INPUT",
                "config_fingerprint": config_fp,
                "created_at_utc": _utc_now(),
            },
            overwrite=True,
        )
        return AerialServiceResult(
            accepted=False,
            exit_code=1,
            error_code="EMPTY_INPUT",
            config_fingerprint=config_fp,
            aerial_parquet=None,
            clearance_parquet=None,
            summary_json=None,
            receipt_json=str(err),
            quality_json=None,
            evaluation_json=None,
            summary={"context_count": 0},
            aerial_duels=[],
            clearances=[],
        )

    aerial, clearances = build_aerial_clearance_from_contexts(
        contexts=contexts,
        config=config,
        policy_fp=policy_fp,
        run_id=rid,
        video_id=vid,
    )
    vr = validate_duels_bundle(
        aerial_duels=aerial,
        clearances=clearances,
        policy=policy,
        expected_run_id=rid,
        expected_video_id=vid,
    )
    if vr.status != "PASS":
        err = output_dir / "failure_receipt.json"
        write_json_record(
            err,
            {
                "schema_version": 1,
                "status": "failed",
                "error_code": "VALIDATION_FAILED",
                "errors": vr.errors,
                "config_fingerprint": config_fp,
                "created_at_utc": _utc_now(),
            },
            overwrite=True,
        )
        return AerialServiceResult(
            accepted=False,
            exit_code=1,
            error_code="VALIDATION_FAILED",
            config_fingerprint=config_fp,
            aerial_parquet=None,
            clearance_parquet=None,
            summary_json=None,
            receipt_json=str(err),
            quality_json=None,
            evaluation_json=None,
            summary={"errors": vr.errors},
            aerial_duels=aerial,
            clearances=clearances,
        )

    a_path = output_dir / "aerial_duel_candidates.parquet"
    c_path = output_dir / "clearance_events.parquet"
    write_contract_parquet(
        _cast("aerial_duel_candidates", aerial),
        a_path,
        get_contract("aerial_duel_candidates", 1),
        overwrite=False,
    )
    write_contract_parquet(
        _cast("clearance_events", clearances),
        c_path,
        get_contract("clearance_events", 1),
        overwrite=False,
    )
    coverage = {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 7_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_500_000,
        "opponent_context_us": 6_000_000,
        "not_observed_us": 1_000_000,
        "joint_coverage_ratio": 0.7,
        "nearby_opponent_alone_is_not_take_on": True,
        "nearest_switch_alone_is_not_duel_outcome": True,
        "monocular_aerial_no_exact_height": True,
        "long_ball_alone_is_not_clearance": True,
    }
    receipt = build_synthetic_receipt(
        run_id=rid,
        video_id=vid,
        duels_policy_fingerprint=policy_fp,
        take_ons=[],
        ground_duels=[],
        aerial_duels=aerial,
        tackles=[],
        recoveries=[],
        turnovers=[],
        clearances=clearances,
        coverage_summary=coverage,
        status="succeeded",
    )
    quality = build_synthetic_quality(
        run_id=rid, video_id=vid, coverage=coverage, duels_policy_fingerprint=policy_fp
    )
    evaluation = evaluate_duels(aerial_duels=aerial).to_dict(
        run_id=rid, video_id=vid, config_fingerprint=config_fp
    )
    summary = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "stage": "12D",
        "aerial_duel_candidate_count": len(aerial),
        "clearance_event_count": len(clearances),
        "evaluation_status": NOT_EVALUATED_DUELS,
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "contract_fingerprints": {
            "aerial_duel_candidates": contract_fingerprint(
                get_contract("aerial_duel_candidates", 1)
            ),
            "clearance_events": contract_fingerprint(get_contract("clearance_events", 1)),
        },
        "artifact_hashes": {
            "aerial_duel_candidates": sha256_file(a_path),
            "clearance_events": sha256_file(c_path),
        },
        "created_at_utc": _utc_now(),
    }
    summary_path = output_dir / "aerial_summary.json"
    receipt_path = output_dir / "aerial_receipt.json"
    quality_path = output_dir / "aerial_quality.json"
    evaluation_path = output_dir / "aerial_evaluation.json"
    write_json_record(summary_path, summary, overwrite=True)
    write_json_record(receipt_path, receipt, overwrite=True)
    write_json_record(quality_path, quality, overwrite=True)
    write_json_record(evaluation_path, evaluation, overwrite=True)
    return AerialServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        aerial_parquet=str(a_path),
        clearance_parquet=str(c_path),
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(evaluation_path),
        summary=summary,
        aerial_duels=aerial,
        clearances=clearances,
    )


__all__ = [
    "AerialServiceError",
    "AerialServiceResult",
    "build_aerial_clearance_from_contexts",
    "compute_aerial_clearance",
]
