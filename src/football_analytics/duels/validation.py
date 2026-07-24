"""Duels bundle validation (Stage 12A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.duels.semantics import (
    assert_finite_optional,
    assert_half_open_interval,
    assert_no_duplicate_pk,
    assert_result_level,
    assert_scope_match,
    automatic_confirmed_allowed,
    cut_replay_gap_allows_event,
    long_ball_alone_is_clearance,
    monocular_aerial_allows_exact_height,
    nearby_opponent_alone_is_take_on,
    nearest_switch_alone_is_duel_outcome,
)
from football_analytics.duels.types import DuelsContractError


@dataclass
class DuelsValidationResult:
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        self.status = "FAIL"

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _rows(table: Any | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_pylist"):
        return list(table.to_pylist())
    if isinstance(table, list):
        return [dict(r) for r in table]
    raise TypeError("expected pyarrow.Table or list of mappings")


def _common_row_checks(result: DuelsValidationResult, row: Mapping[str, Any]) -> None:
    try:
        assert_result_level(str(row["event_state"]))
        assert_half_open_interval(int(row["start_time_us"]), int(row["end_time_us"]))
        assert_finite_optional(row.get("uncertainty"), label="uncertainty")
        assert_finite_optional(row.get("x_m"), label="x_m")
        assert_finite_optional(row.get("y_m"), label="y_m")
    except DuelsContractError as exc:
        result.err(str(exc))
    if str(row.get("automatic_ceiling")) not in {"candidate", "provisional"}:
        result.err("automatic_ceiling must be candidate|provisional")
    if (
        str(row.get("event_state")) == "confirmed"
        and not automatic_confirmed_allowed()
        and str(row.get("review_status")) != "reviewed"
    ):
        result.err("AUTOMATIC_CONFIRMED_FORBIDDEN")
    if row.get("metric_origin") != "project_generated":
        result.err("metric_origin must be project_generated")
    if row.get("definition_style") != "opta_style_metric_definition":
        result.err("definition_style must be opta_style_metric_definition")
    if not cut_replay_gap_allows_event(
        cut_or_replay=bool(row.get("cut_or_replay")), hard_gap=bool(row.get("hard_gap"))
    ) and str(row.get("event_state")) not in {"rejected", "not_evaluable"}:
        result.err("CUT_REPLAY_GAP_NO_EVENT")


def validate_duels_bundle(
    *,
    take_ons: Any | None = None,
    ground_duels: Any | None = None,
    aerial_duels: Any | None = None,
    tackles: Any | None = None,
    recoveries: Any | None = None,
    turnovers: Any | None = None,
    clearances: Any | None = None,
    policy: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
) -> DuelsValidationResult:
    result = DuelsValidationResult()
    take_rows = _rows(take_ons)
    ground_rows = _rows(ground_duels)
    aerial_rows = _rows(aerial_duels)
    tackle_rows = _rows(tackles)
    recovery_rows = _rows(recoveries)
    turnover_rows = _rows(turnovers)
    clearance_rows = _rows(clearances)

    try:
        assert_no_duplicate_pk(take_rows, ["run_id", "video_id", "take_on_attempt_id"])
        assert_no_duplicate_pk(ground_rows, ["run_id", "video_id", "ground_duel_candidate_id"])
        assert_no_duplicate_pk(aerial_rows, ["run_id", "video_id", "aerial_duel_candidate_id"])
        assert_no_duplicate_pk(tackle_rows, ["run_id", "video_id", "tackle_event_id"])
        assert_no_duplicate_pk(recovery_rows, ["run_id", "video_id", "recovery_event_id"])
        assert_no_duplicate_pk(turnover_rows, ["run_id", "video_id", "turnover_event_id"])
        assert_no_duplicate_pk(clearance_rows, ["run_id", "video_id", "clearance_event_id"])
    except DuelsContractError as exc:
        result.err(str(exc))

    if expected_run_id and expected_video_id:
        try:
            for rows in (
                take_rows,
                ground_rows,
                aerial_rows,
                tackle_rows,
                recovery_rows,
                turnover_rows,
                clearance_rows,
            ):
                assert_scope_match(rows, run_id=expected_run_id, video_id=expected_video_id)
        except DuelsContractError as exc:
            result.err(str(exc))

    ground_ids = {(r["run_id"], r["video_id"], r["ground_duel_candidate_id"]) for r in ground_rows}
    recovery_ids = {(r["run_id"], r["video_id"], r["recovery_event_id"]) for r in recovery_rows}
    turnover_ids = {(r["run_id"], r["video_id"], r["turnover_event_id"]) for r in turnover_rows}

    for row in take_rows:
        _common_row_checks(result, row)
        if row.get("implies_take_on") is True and nearby_opponent_alone_is_take_on(
            nearby_opponent_alone=bool(row.get("nearby_opponent_alone"))
        ):
            result.err("NEARBY_OPPONENT_ALONE_NOT_TAKE_ON")
        if row.get("nearby_opponent_alone") is True and row.get("implies_take_on") is True:
            result.err("NEARBY_OPPONENT_ALONE_NOT_TAKE_ON:implies")
        if row.get("nearby_opponent_alone") is True and row.get("implies_take_on_success") is True:
            result.err("NEARBY_OPPONENT_ALONE_NOT_TAKE_ON:success")

    for row in ground_rows:
        _common_row_checks(result, row)
        if row.get("implies_duel_outcome") is True and nearest_switch_alone_is_duel_outcome(
            nearest_switch_alone=bool(row.get("nearest_switch_alone"))
        ):
            result.err("NEAREST_SWITCH_ALONE_NOT_DUEL_OUTCOME")
        if row.get("nearest_switch_alone") is True and row.get("implies_duel_outcome") is True:
            result.err("NEAREST_SWITCH_ALONE_NOT_DUEL_OUTCOME:implies")

    for row in aerial_rows:
        _common_row_checks(result, row)
        if row.get("exact_3d_height_claimed") is True:
            result.err("MONOCULAR_AERIAL_NO_EXACT_HEIGHT:claimed")
        if row.get("exact_3d_height_m") is not None and monocular_aerial_allows_exact_height(
            monocular_only=bool(row.get("monocular_only", True))
        ):
            result.err("MONOCULAR_AERIAL_NO_EXACT_HEIGHT:value")
        if row.get("monocular_only") is True and row.get("exact_3d_height_m") is not None:
            result.err("MONOCULAR_AERIAL_NO_EXACT_HEIGHT:monocular_value")
        if row.get("monocular_only") is True and str(row.get("aerial_evaluability")) not in {
            "candidate",
            "unknown",
            "not_evaluable",
        }:
            result.err("MONOCULAR_AERIAL_NO_EXACT_HEIGHT:evaluability")

    for row in tackle_rows:
        _common_row_checks(result, row)
        rel = row.get("related_ground_duel_candidate_id")
        if rel and (row["run_id"], row["video_id"], rel) not in ground_ids:
            result.err(f"DANGLING_FK:related_ground_duel_candidate_id={rel}")

    for row in recovery_rows:
        _common_row_checks(result, row)
        rel = row.get("related_turnover_event_id")
        if rel and (row["run_id"], row["video_id"], rel) not in turnover_ids:
            result.err(f"DANGLING_FK:related_turnover_event_id={rel}")

    for row in turnover_rows:
        _common_row_checks(result, row)
        rel = row.get("related_recovery_event_id")
        if rel and (row["run_id"], row["video_id"], rel) not in recovery_ids:
            result.err(f"DANGLING_FK:related_recovery_event_id={rel}")

    for row in clearance_rows:
        _common_row_checks(result, row)
        assert_finite_optional(row.get("ball_distance_m"), label="ball_distance_m")
        if row.get("implies_clearance") is True and long_ball_alone_is_clearance(
            long_ball_alone=bool(row.get("long_ball_alone"))
        ):
            result.err("LONG_BALL_ALONE_NOT_CLEARANCE")
        if row.get("long_ball_alone") is True and row.get("implies_clearance") is True:
            result.err("LONG_BALL_ALONE_NOT_CLEARANCE:implies")

    if policy is not None:
        if policy.get("no_real_duels_inference") is not True:
            result.err("policy must declare no_real_duels_inference")
        auto = policy.get("automatic_baseline") or {}
        if auto.get("nearby_opponent_alone_is_not_take_on") is not True:
            result.err("policy nearby_opponent_alone_is_not_take_on")

    return result


def count_duels_rows(
    *,
    take_ons: Sequence[Mapping[str, Any]],
    ground_duels: Sequence[Mapping[str, Any]],
    aerial_duels: Sequence[Mapping[str, Any]],
    tackles: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
    turnovers: Sequence[Mapping[str, Any]],
    clearances: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "take_on_attempt_count": len(take_ons),
        "ground_duel_candidate_count": len(ground_duels),
        "aerial_duel_candidate_count": len(aerial_duels),
        "tackle_event_count": len(tackles),
        "recovery_event_count": len(recoveries),
        "turnover_event_count": len(turnovers),
        "clearance_event_count": len(clearances),
    }


def recount_duels_counts(
    *,
    take_ons: Sequence[Mapping[str, Any]],
    ground_duels: Sequence[Mapping[str, Any]],
    aerial_duels: Sequence[Mapping[str, Any]],
    tackles: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
    turnovers: Sequence[Mapping[str, Any]],
    clearances: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, int] | list[str]:
    counts = count_duels_rows(
        take_ons=take_ons,
        ground_duels=ground_duels,
        aerial_duels=aerial_duels,
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
        clearances=clearances,
    )
    if receipt is None:
        return counts
    errors: list[str] = []
    for key, value in counts.items():
        if int(receipt.get(key, -1)) != value:
            errors.append(f"{key} mismatch")
    return errors


__all__ = [
    "DuelsValidationResult",
    "validate_duels_bundle",
    "count_duels_rows",
    "recount_duels_counts",
]
