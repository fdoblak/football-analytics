"""Stage 12C synthetic ground duel / tackle / recovery / turnover computation."""

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
from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS, evaluate_duels
from football_analytics.duels.ground_config import (
    GroundConfigError,
    ground_config_fingerprint,
    load_ground_config,
)
from football_analytics.duels.policy import load_duels_policy, policy_fingerprint
from football_analytics.duels.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
)
from football_analytics.duels.semantics import (
    cut_replay_gap_allows_event,
    nearest_switch_alone_is_duel_outcome,
    neutral_zone_from_x,
)
from football_analytics.duels.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN
from football_analytics.duels.validation import validate_duels_bundle


class GroundServiceError(RuntimeError):
    """Ground duel family service failure."""


@dataclass
class GroundServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    ground_parquet: str | None
    tackle_parquet: str | None
    recovery_parquet: str | None
    turnover_parquet: str | None
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    ground_duels: Sequence[Mapping[str, Any]]
    tackles: Sequence[Mapping[str, Any]]
    recoveries: Sequence[Mapping[str, Any]]
    turnovers: Sequence[Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def _base_row(
    *,
    run_id: str,
    video_id: str,
    policy_fp: str,
    ctx: Mapping[str, Any],
    target_rel: str,
    event_state: str,
    reasons: list[str],
) -> dict[str, Any]:
    x_m = ctx.get("x_m")
    return {
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
        "event_state": event_state,
        "target_relationship": target_rel,
        "possession_hypothesis_id": ctx.get("possession_hypothesis_id"),
        "contact_candidate_ids": list(ctx.get("contact_candidate_ids") or []),
        "evidence_refs": list(ctx.get("evidence_refs") or []),
        "cut_or_replay": bool(ctx.get("cut_or_replay")),
        "hard_gap": bool(ctx.get("hard_gap")),
        "playability_status": str(ctx.get("playability_status", "playable")),
        "calibration_status": str(ctx.get("calibration_status", "valid")),
        "automatic_ceiling": "provisional",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "uncertainty": float(ctx.get("uncertainty", 0.4)),
        "reason_codes": reasons,
        "quality_flags": [],
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fp,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def build_ground_family_from_contexts(
    *,
    contexts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target = dict(config.get("target") or {})
    target_rel = str(target.get("relationship", "confirmed_target"))
    ground: list[dict[str, Any]] = []
    tackles: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    turnovers: list[dict[str, Any]] = []
    g_i = t_i = r_i = u_i = 0
    for ctx in contexts:
        kind = str(ctx.get("kind", "ground_duel"))
        cut = bool(ctx.get("cut_or_replay"))
        gap = bool(ctx.get("hard_gap"))
        allowed = cut_replay_gap_allows_event(cut_or_replay=cut, hard_gap=gap)
        if kind == "ground_duel":
            g_i += 1
            nearest_alone = bool(ctx.get("nearest_switch_alone"))
            reasons: list[str] = []
            if nearest_alone or nearest_switch_alone_is_duel_outcome(
                nearest_switch_alone=nearest_alone
            ):
                state = "not_evaluable"
                implies = False
                outcome = "not_evaluable"
                reasons.append("NEAREST_SWITCH_ALONE_NOT_DUEL_OUTCOME")
            elif not allowed:
                state = "rejected"
                implies = False
                outcome = "rejected"
                reasons.append("CUT_REPLAY_GAP_NO_EVENT")
            else:
                state = "provisional"
                implies = bool(ctx.get("implies_duel_outcome", False))
                outcome = str(ctx.get("outcome", "uncertain"))
            row = _base_row(
                run_id=run_id,
                video_id=video_id,
                policy_fp=policy_fp,
                ctx=ctx,
                target_rel=target_rel,
                event_state=state,
                reasons=reasons,
            )
            row.update(
                {
                    "ground_duel_candidate_id": f"gduel_{g_i:02d}",
                    "nearest_switch_alone": nearest_alone,
                    "contested_possession": bool(ctx.get("contested_possession", False)),
                    "implies_duel_outcome": implies,
                    "outcome": outcome,
                }
            )
            ground.append(row)
        elif kind == "tackle":
            t_i += 1
            reasons = []
            if not allowed:
                state = "rejected"
                outcome = "rejected"
                reasons.append("CUT_REPLAY_GAP_NO_EVENT")
                implies = False
            else:
                state = "provisional"
                outcome = str(ctx.get("outcome", "uncertain"))
                implies = bool(ctx.get("implies_tackle", True))
            row = _base_row(
                run_id=run_id,
                video_id=video_id,
                policy_fp=policy_fp,
                ctx=ctx,
                target_rel=target_rel,
                event_state=state,
                reasons=reasons,
            )
            row.update(
                {
                    "tackle_event_id": f"tack_{t_i:02d}",
                    "related_ground_duel_candidate_id": ctx.get(
                        "related_ground_duel_candidate_id", "gduel_01"
                    ),
                    "implies_tackle": implies,
                    "implies_tackle_success": bool(ctx.get("implies_tackle_success", False)),
                    "outcome": outcome,
                }
            )
            tackles.append(row)
        elif kind == "turnover":
            u_i += 1
            reasons = []
            state = "rejected" if not allowed else "provisional"
            outcome = "rejected" if not allowed else str(ctx.get("outcome", "lost"))
            if not allowed:
                reasons.append("CUT_REPLAY_GAP_NO_EVENT")
            row = _base_row(
                run_id=run_id,
                video_id=video_id,
                policy_fp=policy_fp,
                ctx=ctx,
                target_rel=target_rel,
                event_state=state,
                reasons=reasons,
            )
            row.update(
                {
                    "turnover_event_id": f"turn_{u_i:02d}",
                    "related_recovery_event_id": ctx.get("related_recovery_event_id", "rec_01"),
                    "implies_turnover": bool(ctx.get("implies_turnover", True)) and allowed,
                    "outcome": outcome,
                }
            )
            turnovers.append(row)
        elif kind == "recovery":
            r_i += 1
            reasons = []
            state = "rejected" if not allowed else "provisional"
            outcome = "rejected" if not allowed else str(ctx.get("outcome", "recovered"))
            if not allowed:
                reasons.append("CUT_REPLAY_GAP_NO_EVENT")
            row = _base_row(
                run_id=run_id,
                video_id=video_id,
                policy_fp=policy_fp,
                ctx=ctx,
                target_rel=target_rel,
                event_state=state,
                reasons=reasons,
            )
            row.update(
                {
                    "recovery_event_id": f"rec_{r_i:02d}",
                    "related_turnover_event_id": ctx.get("related_turnover_event_id", "turn_01"),
                    "implies_recovery": bool(ctx.get("implies_recovery", True)) and allowed,
                    "outcome": outcome,
                }
            )
            recoveries.append(row)
    return ground, tackles, recoveries, turnovers


def compute_ground_family(
    *,
    output_dir: Path,
    contexts: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
) -> GroundServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_ground_config(config_path, project_root=root)
    except GroundConfigError as exc:
        raise GroundServiceError(str(exc)) from exc
    config_fp = ground_config_fingerprint(config)
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
        return GroundServiceResult(
            accepted=False,
            exit_code=1,
            error_code="EMPTY_INPUT",
            config_fingerprint=config_fp,
            ground_parquet=None,
            tackle_parquet=None,
            recovery_parquet=None,
            turnover_parquet=None,
            summary_json=None,
            receipt_json=str(err),
            quality_json=None,
            evaluation_json=None,
            summary={"context_count": 0},
            ground_duels=[],
            tackles=[],
            recoveries=[],
            turnovers=[],
        )

    ground, tackles, recoveries, turnovers = build_ground_family_from_contexts(
        contexts=contexts,
        config=config,
        policy_fp=policy_fp,
        run_id=rid,
        video_id=vid,
    )
    # Align linked ids between synthetic turnover/recovery pair when both present.
    if turnovers and recoveries:
        turnovers[0]["related_recovery_event_id"] = recoveries[0]["recovery_event_id"]
        recoveries[0]["related_turnover_event_id"] = turnovers[0]["turnover_event_id"]
    if ground and tackles:
        for t in tackles:
            if t.get("related_ground_duel_candidate_id") in {None, "gduel_01"}:
                t["related_ground_duel_candidate_id"] = ground[0]["ground_duel_candidate_id"]

    vr = validate_duels_bundle(
        ground_duels=ground,
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
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
        return GroundServiceResult(
            accepted=False,
            exit_code=1,
            error_code="VALIDATION_FAILED",
            config_fingerprint=config_fp,
            ground_parquet=None,
            tackle_parquet=None,
            recovery_parquet=None,
            turnover_parquet=None,
            summary_json=None,
            receipt_json=str(err),
            quality_json=None,
            evaluation_json=None,
            summary={"errors": vr.errors},
            ground_duels=ground,
            tackles=tackles,
            recoveries=recoveries,
            turnovers=turnovers,
        )

    g_path = output_dir / "ground_duel_candidates.parquet"
    t_path = output_dir / "tackle_events.parquet"
    r_path = output_dir / "recovery_events.parquet"
    u_path = output_dir / "turnover_events.parquet"
    write_contract_parquet(
        _cast("ground_duel_candidates", ground),
        g_path,
        get_contract("ground_duel_candidates", 1),
        overwrite=False,
    )
    write_contract_parquet(
        _cast("tackle_events", tackles), t_path, get_contract("tackle_events", 1), overwrite=False
    )
    write_contract_parquet(
        _cast("recovery_events", recoveries),
        r_path,
        get_contract("recovery_events", 1),
        overwrite=False,
    )
    write_contract_parquet(
        _cast("turnover_events", turnovers),
        u_path,
        get_contract("turnover_events", 1),
        overwrite=False,
    )
    coverage = {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 8_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_500_000,
        "opponent_context_us": 8_000_000,
        "not_observed_us": 500_000,
        "joint_coverage_ratio": 0.8,
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
        ground_duels=ground,
        aerial_duels=[],
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
        clearances=[],
        coverage_summary=coverage,
        status="succeeded",
    )
    quality = build_synthetic_quality(
        run_id=rid, video_id=vid, coverage=coverage, duels_policy_fingerprint=policy_fp
    )
    evaluation = evaluate_duels(ground_duels=ground).to_dict(
        run_id=rid, video_id=vid, config_fingerprint=config_fp
    )
    summary = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "stage": "12C",
        "ground_duel_candidate_count": len(ground),
        "tackle_event_count": len(tackles),
        "recovery_event_count": len(recoveries),
        "turnover_event_count": len(turnovers),
        "evaluation_status": NOT_EVALUATED_DUELS,
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "contract_fingerprints": {
            "ground_duel_candidates": contract_fingerprint(
                get_contract("ground_duel_candidates", 1)
            ),
            "tackle_events": contract_fingerprint(get_contract("tackle_events", 1)),
            "recovery_events": contract_fingerprint(get_contract("recovery_events", 1)),
            "turnover_events": contract_fingerprint(get_contract("turnover_events", 1)),
        },
        "artifact_hashes": {
            "ground_duel_candidates": sha256_file(g_path),
            "tackle_events": sha256_file(t_path),
            "recovery_events": sha256_file(r_path),
            "turnover_events": sha256_file(u_path),
        },
        "created_at_utc": _utc_now(),
    }
    summary_path = output_dir / "ground_summary.json"
    receipt_path = output_dir / "ground_receipt.json"
    quality_path = output_dir / "ground_quality.json"
    evaluation_path = output_dir / "ground_evaluation.json"
    write_json_record(summary_path, summary, overwrite=True)
    write_json_record(receipt_path, receipt, overwrite=True)
    write_json_record(quality_path, quality, overwrite=True)
    write_json_record(evaluation_path, evaluation, overwrite=True)
    return GroundServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        ground_parquet=str(g_path),
        tackle_parquet=str(t_path),
        recovery_parquet=str(r_path),
        turnover_parquet=str(u_path),
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(evaluation_path),
        summary=summary,
        ground_duels=ground,
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
    )


__all__ = [
    "GroundServiceError",
    "GroundServiceResult",
    "build_ground_family_from_contexts",
    "compute_ground_family",
]
