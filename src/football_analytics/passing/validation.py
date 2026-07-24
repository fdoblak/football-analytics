"""Passing bundle validation (Stage 11A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.passing.semantics import (
    assert_finite_optional,
    assert_half_open_interval,
    assert_no_duplicate_pk,
    assert_result_level,
    assert_scope_match,
    automatic_confirmed_allowed,
    box_touch_eligible,
    cut_replay_gap_allows_pass,
    owner_change_alone_is_completed_pass,
    penalty_presence_is_box_touch,
)
from football_analytics.passing.types import PassingContractError


@dataclass
class PassingValidationResult:
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


def validate_passing_bundle(
    *,
    passes: Any | None = None,
    receptions: Any | None = None,
    outcomes: Any | None = None,
    progression: Any | None = None,
    touches: Any | None = None,
    policy: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
) -> PassingValidationResult:
    result = PassingValidationResult()
    pass_rows = _rows(passes)
    recv_rows = _rows(receptions)
    out_rows = _rows(outcomes)
    prog_rows = _rows(progression)
    touch_rows = _rows(touches)

    try:
        assert_no_duplicate_pk(pass_rows, ["run_id", "video_id", "pass_candidate_id"])
        assert_no_duplicate_pk(recv_rows, ["run_id", "video_id", "reception_candidate_id"])
        assert_no_duplicate_pk(out_rows, ["run_id", "video_id", "outcome_id"])
        assert_no_duplicate_pk(prog_rows, ["run_id", "video_id", "segment_id"])
        assert_no_duplicate_pk(touch_rows, ["run_id", "video_id", "touch_id"])
    except PassingContractError as exc:
        result.err(str(exc))

    if expected_run_id and expected_video_id:
        try:
            for rows in (pass_rows, recv_rows, out_rows, prog_rows, touch_rows):
                assert_scope_match(rows, run_id=expected_run_id, video_id=expected_video_id)
        except PassingContractError as exc:
            result.err(str(exc))

    pass_ids = {(r["run_id"], r["video_id"], r["pass_candidate_id"]) for r in pass_rows}
    recv_ids = {(r["run_id"], r["video_id"], r["reception_candidate_id"]) for r in recv_rows}

    for p in pass_rows:
        try:
            assert_result_level(str(p["candidate_state"]))
            assert_half_open_interval(int(p["start_time_us"]), int(p["end_time_us"]))
            assert_finite_optional(p.get("pass_distance_m"), label="pass_distance_m")
            assert_finite_optional(p.get("uncertainty"), label="uncertainty")
        except PassingContractError as exc:
            result.err(str(exc))
        if p.get("implies_completed_pass") is True and owner_change_alone_is_completed_pass(
            owner_changed=bool(p.get("owner_change_alone"))
        ):
            result.err("OWNER_CHANGE_ALONE_NOT_PASS")
        if p.get("implies_completed_pass") is True and bool(p.get("owner_change_alone")):
            result.err("OWNER_CHANGE_ALONE_NOT_PASS:implies_completed")
        if not cut_replay_gap_allows_pass(
            cut_or_replay=bool(p.get("cut_or_replay")), hard_gap=bool(p.get("hard_gap"))
        ) and str(p.get("candidate_state")) not in {"rejected", "not_evaluable"}:
            result.err("CUT_REPLAY_GAP_NO_PASS")
        if str(p.get("automatic_ceiling")) not in {"candidate", "provisional"}:
            result.err("automatic_ceiling must be candidate|provisional")
        if (
            str(p.get("candidate_state")) == "confirmed"
            and not automatic_confirmed_allowed()
            and str(p.get("review_status")) != "reviewed"
        ):
            result.err("AUTOMATIC_CONFIRMED_FORBIDDEN")
        if p.get("metric_origin") != "project_generated":
            result.err("metric_origin must be project_generated")
        if p.get("definition_style") != "opta_style_metric_definition":
            result.err("definition_style must be opta_style_metric_definition")

    for r in recv_rows:
        try:
            assert_result_level(str(r["candidate_state"]))
            assert_half_open_interval(int(r["start_time_us"]), int(r["end_time_us"]))
        except PassingContractError as exc:
            result.err(str(exc))
        src = r.get("source_pass_candidate_id")
        if src and (r["run_id"], r["video_id"], src) not in pass_ids:
            result.err(f"DANGLING_FK:source_pass_candidate_id={src}")
        if r.get("implies_completed_pass") is True and not r.get("contact_candidate_ids"):
            result.warn("reception implies completed without contact refs")

    for o in out_rows:
        try:
            assert_result_level(str(o["outcome_state"]))
        except PassingContractError as exc:
            result.err(str(exc))
        pk = (o["run_id"], o["video_id"], o["pass_candidate_id"])
        if pk not in pass_ids:
            result.err(f"DANGLING_FK:pass_candidate_id={o.get('pass_candidate_id')}")
        rid = o.get("reception_candidate_id")
        if rid and (o["run_id"], o["video_id"], rid) not in recv_ids:
            result.err(f"DANGLING_FK:reception_candidate_id={rid}")
        if o.get("owner_change_alone") is True and str(o.get("outcome")) == "completed":
            result.err("OWNER_CHANGE_ALONE_NOT_PASS:outcome")
        if not o.get("attack_relative_evaluable"):
            if str(o.get("progression_1_to_2")) != "not_evaluable":
                result.err("DIRECTIONAL_METRICS_NOT_EVALUABLE:1_to_2")
            if str(o.get("progression_2_to_3")) != "not_evaluable":
                result.err("DIRECTIONAL_METRICS_NOT_EVALUABLE:2_to_3")
        if str(o.get("outcome_state")) == "confirmed" and str(o.get("review_status")) != "reviewed":
            result.err("AUTOMATIC_CONFIRMED_FORBIDDEN")

    for g in prog_rows:
        try:
            assert_result_level(str(g["segment_state"]))
            assert_half_open_interval(int(g["start_time_us"]), int(g["end_time_us"]))
        except PassingContractError as exc:
            result.err(str(exc))
        if str(g.get("attack_direction")) == "unknown":
            if str(g.get("progression_1_to_2")) != "not_evaluable":
                result.err("ATTACK_DIRECTION_UNKNOWN:1_to_2")
            if str(g.get("progression_2_to_3")) != "not_evaluable":
                result.err("ATTACK_DIRECTION_UNKNOWN:2_to_3")

    for t in touch_rows:
        try:
            assert_result_level(str(t["touch_state"]))
            assert_finite_optional(t.get("uncertainty"), label="uncertainty")
        except PassingContractError as exc:
            result.err(str(exc))
        if (
            t.get("penalty_presence_alone") is True
            and t.get("is_box_touch_candidate") is True
            and penalty_presence_is_box_touch(in_penalty=True)
        ):
            result.err("PENALTY_PRESENCE_NOT_BOX_TOUCH")
        if (
            t.get("is_box_touch_candidate") is True
            and not t.get("has_possession_or_contact")
            and t.get("penalty_presence_alone") is True
        ):
            result.err("PENALTY_PRESENCE_NOT_BOX_TOUCH:missing_contact")
        if t.get("is_box_touch_candidate") is True and not box_touch_eligible(
            in_penalty=bool(t.get("in_penalty_area")),
            has_possession_or_contact=bool(t.get("has_possession_or_contact")),
            has_pitch_mapping=bool(t.get("has_pitch_mapping")),
            playability_status=str(t.get("playability_status")),
        ):
            result.err("PENALTY_PRESENCE_NOT_BOX_TOUCH:ineligible")

    if policy is not None:
        if policy.get("no_real_passing_inference") is not True:
            result.err("policy must declare no_real_passing_inference")
        auto = policy.get("automatic_baseline") or {}
        if auto.get("owner_change_alone_is_not_completed_pass") is not True:
            result.err("policy owner_change_alone_is_not_completed_pass")

    return result


def recount_passing_counts(
    *,
    passes: Sequence[Mapping[str, Any]],
    receptions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    progression: Sequence[Mapping[str, Any]],
    touches: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if int(receipt["pass_candidate_count"]) != len(passes):
        errors.append("pass_candidate_count mismatch")
    if int(receipt["reception_candidate_count"]) != len(receptions):
        errors.append("reception_candidate_count mismatch")
    if int(receipt["pass_outcome_count"]) != len(outcomes):
        errors.append("pass_outcome_count mismatch")
    if int(receipt["progression_segment_count"]) != len(progression):
        errors.append("progression_segment_count mismatch")
    if int(receipt["target_ball_touch_count"]) != len(touches):
        errors.append("target_ball_touch_count mismatch")
    return errors


__all__ = [
    "PassingValidationResult",
    "validate_passing_bundle",
    "recount_passing_counts",
]
