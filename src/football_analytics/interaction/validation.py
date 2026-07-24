"""Human-ball interaction bundle validation (Stage 10A — contracts only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.interaction.eligibility import pitch_distance_usable
from football_analytics.interaction.semantics import (
    assert_finite_optional,
    assert_half_open_interval,
    assert_no_duplicate_pk,
    assert_result_level,
    assert_scope_match,
    event_metrics_forbidden,
    intervals_overlap,
    nearest_player_is_possession,
    owner_transition_requires_evidence,
)
from football_analytics.interaction.types import InteractionContractError


@dataclass
class InteractionValidationResult:
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


def validate_interaction_bundle(
    *,
    proximity: Any | None = None,
    contacts: Any | None = None,
    possessions: Any | None = None,
    policy: Mapping[str, Any] | None = None,
    event_metrics: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
) -> InteractionValidationResult:
    result = InteractionValidationResult()
    prox_rows = _rows(proximity)
    contact_rows = _rows(contacts)
    poss_rows = _rows(possessions)

    try:
        assert_no_duplicate_pk(prox_rows, ["run_id", "video_id", "proximity_id"])
        assert_no_duplicate_pk(contact_rows, ["run_id", "video_id", "candidate_id"])
        assert_no_duplicate_pk(poss_rows, ["run_id", "video_id", "hypothesis_id"])
    except InteractionContractError as exc:
        result.err(str(exc))

    if expected_run_id and expected_video_id:
        try:
            assert_scope_match(prox_rows, run_id=expected_run_id, video_id=expected_video_id)
            assert_scope_match(contact_rows, run_id=expected_run_id, video_id=expected_video_id)
            assert_scope_match(poss_rows, run_id=expected_run_id, video_id=expected_video_id)
        except InteractionContractError as exc:
            result.err(str(exc))

    prox_ids = {(r["run_id"], r["video_id"], r["proximity_id"]) for r in prox_rows}

    for p in prox_rows:
        try:
            assert_result_level(str(p["evidence_level"]))
            assert_finite_optional(p.get("image_distance_px"), label="image_distance_px")
            assert_finite_optional(p.get("pitch_distance_m"), label="pitch_distance_m")
            assert_finite_optional(p.get("uncertainty"), label="uncertainty")
        except InteractionContractError as exc:
            result.err(str(exc))
        if p.get("nearest_implies_possession") is True:
            result.err("NEAREST_PLAYER_NOT_POSSESSION:nearest_implies_possession must be false")
        if p.get("is_nearest_human") and nearest_player_is_possession(is_nearest=True):
            result.err("NEAREST_PLAYER_NOT_POSSESSION")
        usable, reasons = pitch_distance_usable(p)
        if p.get("pitch_distance_usable") is True and not usable:
            result.err(f"pitch_distance_usable inconsistent: {reasons}")
        if (
            str(p.get("ball_air_state")) == "unknown"
            and p.get("pitch_distance_m") is not None
            and p.get("pitch_distance_usable") is True
        ):
            result.err("AIRBORNE_UNKNOWN_BLOCKS_PITCH")

    contact_ids = {(r["run_id"], r["video_id"], r["candidate_id"]) for r in contact_rows}
    for c in contact_rows:
        try:
            assert_result_level(str(c["contact_state"]))
            assert_half_open_interval(int(c["start_time_us"]), int(c["end_time_us"]))
            assert_finite_optional(c.get("confidence"), label="confidence")
            peak_ok = int(c["start_time_us"]) <= int(c["peak_time_us"]) <= int(c["end_time_us"])
            if not peak_ok:
                result.err(f"peak outside interval: {c.get('candidate_id')}")
        except InteractionContractError as exc:
            result.err(str(exc))
        if c.get("implies_controlled_possession") is True:
            result.err("CONTACT_NOT_POSSESSION")
        if c.get("implies_pass_or_event") is True:
            result.err("CONTACT_NOT_EVENT")
        if c.get("implies_box_touch") is True:
            result.err("CONTACT_NOT_EVENT:box_touch")
        if str(c.get("contact_state")) == "confirmed" and str(c.get("review_status")) != "reviewed":
            result.err("AUTOMATIC_CONFIRMED_FORBIDDEN")
        for pid in list(c.get("proximity_ids") or []):
            key = (c["run_id"], c["video_id"], pid)
            if key not in prox_ids and pid:
                result.err(f"DANGLING_FK:proximity_id={pid}")
        soft = (
            c.get("proximity_support")
            and not c.get("multi_frame_support")
            and str(c.get("contact_state")) not in {"rejected", "not_evaluable", "unknown"}
            and "SINGLE_FRAME_PROXIMITY" not in list(c.get("reason_codes") or [])
        )
        if soft:
            result.warn("proximity_support without multi_frame_support")

    for h in poss_rows:
        try:
            assert_result_level(str(h["possession_state"]))
            assert_half_open_interval(int(h["start_time_us"]), int(h["end_time_us"]))
            assert_finite_optional(h.get("uncertainty"), label="uncertainty")
            assert_finite_optional(
                h.get("observed_coverage_ratio"), label="observed_coverage_ratio"
            )
        except InteractionContractError as exc:
            result.err(str(exc))
        if str(h.get("automatic_ceiling")) not in {"candidate", "provisional"}:
            result.err("automatic_ceiling must be candidate|provisional")
        if (
            str(h.get("possession_state")) == "confirmed"
            and str(h.get("review_status")) != "reviewed"
        ):
            result.err("AUTOMATIC_CONFIRMED_FORBIDDEN")
        for flag in (
            "implies_completed_pass",
            "implies_dribble_or_take_on",
            "implies_duel_or_aerial",
            "implies_box_touch",
            "implies_turnover",
        ):
            if h.get(flag) is True:
                result.err(f"POSSESSION_NOT_EVENT:{flag}")
        for cid in list(h.get("contact_candidate_ids") or []):
            key = (h["run_id"], h["video_id"], cid)
            if key not in contact_ids and cid:
                result.err(f"DANGLING_FK:contact_candidate_id={cid}")
        for pid in list(h.get("proximity_ids") or []):
            key = (h["run_id"], h["video_id"], pid)
            if key not in prox_ids and pid:
                result.err(f"DANGLING_FK:proximity_id={pid}")

    # Overlapping non-contested ownership for same owner → conflict
    active = [
        h
        for h in poss_rows
        if str(h.get("possession_state")) in {"candidate", "provisional", "confirmed"}
        and h.get("owner_human_track_id") is not None
    ]
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            if (
                a["run_id"] == b["run_id"]
                and a["video_id"] == b["video_id"]
                and a["owner_human_track_id"] == b["owner_human_track_id"]
                and intervals_overlap(
                    int(a["start_time_us"]),
                    int(a["end_time_us"]),
                    int(b["start_time_us"]),
                    int(b["end_time_us"]),
                )
            ):
                result.err(
                    "OVERLAPPING_POSSESSION: " f"{a['hypothesis_id']} vs {b['hypothesis_id']}"
                )

    # Contested: multiple owners overlapping without contested state
    owners_active = [
        h
        for h in poss_rows
        if str(h.get("possession_state")) in {"candidate", "provisional", "confirmed"}
    ]
    for i, a in enumerate(owners_active):
        for b in owners_active[i + 1 :]:
            if (
                a["run_id"] == b["run_id"]
                and a["video_id"] == b["video_id"]
                and a.get("owner_human_track_id") != b.get("owner_human_track_id")
                and intervals_overlap(
                    int(a["start_time_us"]),
                    int(a["end_time_us"]),
                    int(b["start_time_us"]),
                    int(b["end_time_us"]),
                )
            ):
                result.err("OVERLAPPING_POSSESSION:multi_owner_without_contested")

    # Owner transitions need evidence
    by_owner_chain = sorted(poss_rows, key=lambda r: int(r["start_time_us"]))
    prev: dict[str, Any] | None = None
    for h in by_owner_chain:
        owner_changed = prev is not None and prev.get("owner_human_track_id") != h.get(
            "owner_human_track_id"
        )
        if owner_changed and not owner_transition_requires_evidence(
            previous_owner=prev.get("owner_human_track_id") if prev else None,
            new_owner=h.get("owner_human_track_id"),
            evidence_refs=list(h.get("evidence_refs") or []),
        ):
            result.err("OWNER_TRANSITION_WITHOUT_EVIDENCE")
        prev = h

    if event_metrics is not None:
        for e in event_metrics_forbidden(event_metrics):
            result.err(e)

    if policy is not None:
        if policy.get("no_real_interaction_inference") is not True:
            result.err("policy must declare no_real_interaction_inference")
        auto = policy.get("automatic_baseline") or {}
        if auto.get("nearest_player_is_not_possession") is not True:
            result.err("policy nearest_player_is_not_possession")

    return result


def recount_interaction_counts(
    *,
    proximity: Sequence[Mapping[str, Any]],
    contacts: Sequence[Mapping[str, Any]],
    possessions: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if int(receipt["proximity_row_count"]) != len(proximity):
        errors.append("proximity_row_count mismatch")
    if int(receipt["contact_candidate_count"]) != len(contacts):
        errors.append("contact_candidate_count mismatch")
    if int(receipt["possession_hypothesis_count"]) != len(possessions):
        errors.append("possession_hypothesis_count mismatch")
    return errors


__all__ = [
    "InteractionValidationResult",
    "validate_interaction_bundle",
    "recount_interaction_counts",
]
