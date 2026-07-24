"""Stage 12B synthetic take-on computation."""

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
from football_analytics.duels.policy import load_duels_policy, policy_fingerprint
from football_analytics.duels.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
)
from football_analytics.duels.semantics import (
    cut_replay_gap_allows_event,
    nearby_opponent_alone_is_take_on,
    neutral_zone_from_x,
)
from football_analytics.duels.take_on_config import (
    TakeOnConfigError,
    load_take_on_config,
    take_on_config_fingerprint,
)
from football_analytics.duels.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN
from football_analytics.duels.validation import validate_duels_bundle


class TakeOnServiceError(RuntimeError):
    """Take-on baseline service failure."""


@dataclass
class TakeOnServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    take_on_parquet: str | None
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    take_ons: Sequence[Mapping[str, Any]]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "take_on_parquet": self.take_on_parquet,
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


def build_take_ons_from_contexts(
    *,
    contexts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
) -> list[dict[str, Any]]:
    thr = dict(config.get("thresholds") or {})
    min_carry = float(thr.get("min_carry_distance_m", 1.0))
    target = dict(config.get("target") or {})
    target_track = int(target.get("track_id", 1))
    target_rel = str(target.get("relationship", "confirmed_target"))
    rows: list[dict[str, Any]] = []
    for i, ctx in enumerate(contexts):
        idx = i + 1
        cut = bool(ctx.get("cut_or_replay"))
        gap = bool(ctx.get("hard_gap"))
        nearby_alone = bool(ctx.get("nearby_opponent_alone", False))
        direction_alone = bool(ctx.get("direction_change_alone", False))
        carry = ctx.get("carry_distance_m")
        has_ev = bool(ctx.get("evidence_refs")) or bool(ctx.get("contact_candidate_ids"))
        has_poss = bool(ctx.get("has_possession_or_contact"))
        allowed = cut_replay_gap_allows_event(cut_or_replay=cut, hard_gap=gap)
        reasons: list[str] = []
        if nearby_alone or nearby_opponent_alone_is_take_on(nearby_opponent_alone=nearby_alone):
            state = "not_evaluable"
            implies = False
            outcome = "not_evaluable"
            reasons.append("NEARBY_OPPONENT_ALONE_NOT_TAKE_ON")
        elif direction_alone and not has_ev:
            state = "not_evaluable"
            implies = False
            outcome = "not_evaluable"
            reasons.append("direction_change_alone")
        elif not allowed:
            state = "rejected"
            implies = False
            outcome = "rejected"
            reasons.append("CUT_REPLAY_GAP_NO_EVENT")
        elif carry is not None and float(carry) < min_carry:
            state = "rejected"
            implies = False
            outcome = "rejected"
            reasons.append("below_min_carry_distance")
        elif not has_poss:
            state = "not_evaluable"
            implies = False
            outcome = "not_evaluable"
            reasons.append("MISSING_CONTACT_OR_POSSESSION")
        else:
            state = "provisional"
            implies = True
            outcome = "beaten" if ctx.get("beaten_opponent") else "attempted"
        x_m = ctx.get("x_m")
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "take_on_attempt_id": f"take_{idx:02d}",
                "target_human_track_id": int(ctx.get("target_track_id", target_track)),
                "opponent_human_track_id": (
                    int(ctx["opponent_track_id"])
                    if ctx.get("opponent_track_id") is not None
                    else None
                ),
                "start_time_us": int(ctx.get("start_time_us", idx * 1_000_000)),
                "end_time_us": int(ctx.get("end_time_us", idx * 1_000_000 + 500_000)),
                "x_m": float(x_m) if x_m is not None else None,
                "y_m": float(ctx["y_m"]) if ctx.get("y_m") is not None else None,
                "zone_neutral": str(ctx.get("zone_neutral") or neutral_zone_from_x(x_m=x_m)),
                "event_state": state,
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
                "uncertainty": float(ctx.get("uncertainty", 0.4)),
                "reason_codes": reasons,
                "quality_flags": [],
                "metric_origin": METRIC_ORIGIN,
                "definition_style": DEFINITION_STYLE,
                "policy_fingerprint": policy_fp,
                "provenance_json": None,
                "contract_version": CONTRACT_VERSION,
                "nearby_opponent_alone": nearby_alone,
                "direction_change_alone": direction_alone,
                "implies_take_on": implies,
                "implies_take_on_success": bool(ctx.get("beaten_opponent")) and implies,
                "outcome": outcome,
            }
        )
    return rows


def compute_take_ons(
    *,
    output_dir: Path,
    contexts: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
) -> TakeOnServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_take_on_config(config_path, project_root=root)
    except TakeOnConfigError as exc:
        raise TakeOnServiceError(str(exc)) from exc
    config_fp = take_on_config_fingerprint(config)
    policy = load_duels_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)
    rid = run_id or generate_run_id()
    vid = video_id or "video_synth_01"
    if not contexts:
        err_path = output_dir / "failure_receipt.json"
        write_json_record(
            err_path,
            {
                "schema_version": 1,
                "status": "failed",
                "error_code": "EMPTY_INPUT",
                "config_fingerprint": config_fp,
                "created_at_utc": _utc_now(),
            },
            overwrite=True,
        )
        return TakeOnServiceResult(
            accepted=False,
            exit_code=1,
            error_code="EMPTY_INPUT",
            config_fingerprint=config_fp,
            take_on_parquet=None,
            summary_json=None,
            receipt_json=str(err_path),
            quality_json=None,
            evaluation_json=None,
            summary={"context_count": 0},
            take_ons=[],
        )

    take_ons = build_take_ons_from_contexts(
        contexts=contexts,
        config=config,
        policy_fp=policy_fp,
        run_id=rid,
        video_id=vid,
    )
    vr = validate_duels_bundle(
        take_ons=take_ons,
        policy=policy,
        expected_run_id=rid,
        expected_video_id=vid,
    )
    if vr.status != "PASS":
        err_path = output_dir / "failure_receipt.json"
        write_json_record(
            err_path,
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
        return TakeOnServiceResult(
            accepted=False,
            exit_code=1,
            error_code="VALIDATION_FAILED",
            config_fingerprint=config_fp,
            take_on_parquet=None,
            summary_json=None,
            receipt_json=str(err_path),
            quality_json=None,
            evaluation_json=None,
            summary={"errors": vr.errors},
            take_ons=take_ons,
        )

    tbl = _cast("take_on_attempts", take_ons)
    parquet_path = output_dir / "take_on_attempts.parquet"
    write_contract_parquet(tbl, parquet_path, get_contract("take_on_attempts", 1), overwrite=False)
    coverage = {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 8_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_500_000,
        "opponent_context_us": 7_000_000,
        "not_observed_us": 500_000,
        "joint_coverage_ratio": 0.8,
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
        ground_duels=[],
        aerial_duels=[],
        tackles=[],
        recoveries=[],
        turnovers=[],
        clearances=[],
        coverage_summary=coverage,
        status="succeeded",
    )
    quality = build_synthetic_quality(
        run_id=rid,
        video_id=vid,
        coverage=coverage,
        duels_policy_fingerprint=policy_fp,
    )
    evaluation = evaluate_duels(take_ons=take_ons).to_dict(
        run_id=rid, video_id=vid, config_fingerprint=config_fp
    )
    summary = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "stage": "12B",
        "take_on_attempt_count": len(take_ons),
        "evaluation_status": NOT_EVALUATED_DUELS,
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "contract_fingerprints": {
            "take_on_attempts": contract_fingerprint(get_contract("take_on_attempts", 1))
        },
        "artifact_hashes": {"take_on_attempts": sha256_file(parquet_path)},
        "created_at_utc": _utc_now(),
    }
    summary_path = output_dir / "take_on_summary.json"
    receipt_path = output_dir / "take_on_receipt.json"
    quality_path = output_dir / "take_on_quality.json"
    evaluation_path = output_dir / "take_on_evaluation.json"
    write_json_record(summary_path, summary, overwrite=True)
    write_json_record(receipt_path, receipt, overwrite=True)
    write_json_record(quality_path, quality, overwrite=True)
    write_json_record(evaluation_path, evaluation, overwrite=True)
    write_json_record(output_dir / "take_on_request.json", request, overwrite=True)
    return TakeOnServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        take_on_parquet=str(parquet_path),
        summary_json=str(summary_path),
        receipt_json=str(receipt_path),
        quality_json=str(quality_path),
        evaluation_json=str(evaluation_path),
        summary=summary,
        take_ons=take_ons,
    )


__all__ = [
    "TakeOnServiceError",
    "TakeOnServiceResult",
    "build_take_ons_from_contexts",
    "compute_take_ons",
]
