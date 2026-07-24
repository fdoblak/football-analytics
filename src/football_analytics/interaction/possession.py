"""Stage 10C finite-state possession hypothesis builder.

Maps conceptual states to contract enums:
  provisional_control → provisional
  loose_ball → unknown + LOOSE_BALL
  not_observed → not_evaluable
  terminated → end interval with termination_reason (or rejected stub)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.interaction.types import CONTRACT_VERSION


def _contacts_covering(
    contacts: Sequence[Mapping[str, Any]], *, time_us: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in contacts:
        if str(c.get("contact_state")) in {"rejected", "not_evaluable"}:
            continue
        if int(c["start_time_us"]) <= time_us < int(c["end_time_us"]):
            out.append(dict(c))
    return out


def _classify_instant(
    *,
    time_us: int,
    prox_at_t: Sequence[Mapping[str, Any]],
    contacts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one timestamp into a possession snapshot (not yet an interval)."""
    poss_cfg = config.get("possession") or {}
    require_contact = bool(poss_cfg.get("require_contact_for_provisional", True))
    nearest_alone = bool(poss_cfg.get("nearest_alone_is_not_owner", True))
    min_prov_us = int(poss_cfg.get("min_contact_duration_us_for_provisional", 80_000))

    if not prox_at_t:
        return {
            "kind": "not_evaluable",
            "state": "not_evaluable",
            "owner": None,
            "contested": [],
            "reasons": ["NOT_OBSERVED"],
            "termination": "none",
            "target_relationship": "unknown",
            "prox_ids": [],
            "contact_ids": [],
            "evidence_refs": [],
            "coverage": None,
        }

    sample = prox_at_t[0]
    play = str(sample.get("playability_status", "playable"))
    if play in {"replay", "non_playable"}:
        return {
            "kind": "terminate",
            "state": "not_evaluable",
            "owner": None,
            "contested": [],
            "reasons": ["REPLAY_OR_CUT_TERMINATES"],
            "termination": "replay" if play == "replay" else "non_playable",
            "target_relationship": str(sample.get("target_relationship", "unknown")),
            "prox_ids": [str(p["proximity_id"]) for p in prox_at_t],
            "contact_ids": [],
            "evidence_refs": [str(p["proximity_id"]) for p in prox_at_t],
            "coverage": 0.0,
        }

    ball_states = {str(p.get("ball_observation_state")) for p in prox_at_t}
    if "missing" in ball_states and ball_states <= {"missing"}:
        return {
            "kind": "missing_ball",
            "state": "not_evaluable",
            "owner": None,
            "contested": [],
            "reasons": ["MISSING_BALL_NOT_NO_POSSESSION"],
            "termination": "none",
            "target_relationship": str(sample.get("target_relationship", "unknown")),
            "prox_ids": [str(p["proximity_id"]) for p in prox_at_t],
            "contact_ids": [],
            "evidence_refs": [str(p["proximity_id"]) for p in prox_at_t],
            "coverage": 0.0,
        }

    if any(
        str(p.get("ball_observation_state")) in {"predicted", "interpolated"} for p in prox_at_t
    ):
        return {
            "kind": "predicted",
            "state": "rejected",
            "owner": None,
            "contested": [],
            "reasons": ["PREDICTED_SOLE_EVIDENCE"],
            "termination": "none",
            "target_relationship": str(sample.get("target_relationship", "unknown")),
            "prox_ids": [str(p["proximity_id"]) for p in prox_at_t],
            "contact_ids": [],
            "evidence_refs": [str(p["proximity_id"]) for p in prox_at_t],
            "coverage": None,
        }

    eligible = [
        p
        for p in prox_at_t
        if str(p.get("eligibility_status")) == "eligible"
        and str(p.get("evidence_level")) in {"candidate", "provisional"}
    ]
    covering = _contacts_covering(contacts, time_us=time_us)
    contact_humans = sorted({int(c["human_track_id"]) for c in covering})
    eligible_humans = sorted({int(p["human_track_id"]) for p in eligible})

    prox_ids = [str(p["proximity_id"]) for p in prox_at_t]
    contact_ids = [str(c["candidate_id"]) for c in covering]
    target_rel = str(sample.get("target_relationship", "confirmed_target"))

    if not eligible_humans and not contact_humans:
        # Observed ball, no eligible proximity → loose_ball → unknown + LOOSE_BALL
        return {
            "kind": "loose_ball",
            "state": "unknown",
            "owner": None,
            "contested": [],
            "reasons": ["LOOSE_BALL"],
            "termination": "none",
            "target_relationship": target_rel,
            "prox_ids": prox_ids,
            "contact_ids": contact_ids,
            "evidence_refs": prox_ids + contact_ids,
            "coverage": 0.0,
        }

    if len(contact_humans) >= 2 or (
        len(eligible_humans) >= 2 and bool(poss_cfg.get("multi_player_overlap_is_contested", True))
    ):
        parts = sorted(set(contact_humans) | set(eligible_humans))
        return {
            "kind": "contested",
            "state": "contested",
            "owner": None,
            "contested": parts,
            "reasons": ["MULTI_PLAYER_AMBIGUITY"],
            "termination": "none",
            "target_relationship": target_rel,
            "prox_ids": prox_ids,
            "contact_ids": contact_ids,
            "evidence_refs": prox_ids + contact_ids,
            "coverage": 0.5,
        }

    # Single-player path
    owner = contact_humans[0] if contact_humans else eligible_humans[0]
    owner_contacts = [c for c in covering if int(c["human_track_id"]) == owner]
    owner_prox = [p for p in eligible if int(p["human_track_id"]) == owner]
    nearest_only = (
        nearest_alone
        and not owner_contacts
        and all(bool(p.get("is_nearest_human")) for p in owner_prox)
        and len(owner_prox) >= 1
    )
    if nearest_only and not owner_contacts:
        return {
            "kind": "nearest_not_owner",
            "state": "unknown",
            "owner": None,
            "contested": [],
            "reasons": ["NEAREST_PLAYER_NOT_POSSESSION"],
            "termination": "none",
            "target_relationship": target_rel,
            "prox_ids": prox_ids,
            "contact_ids": contact_ids,
            "evidence_refs": prox_ids,
            "coverage": 0.2,
        }

    state = "candidate"
    reasons: list[str] = ["TEMPORAL_PROXIMITY_SUPPORT"]
    if owner_contacts:
        reasons.append("CONTACT_CANDIDATE_SUPPORT")
        multi = any(bool(c.get("multi_frame_support")) for c in owner_contacts)
        traj = any(bool(c.get("trajectory_change_support")) for c in owner_contacts)
        dur = max(
            (int(c["end_time_us"]) - int(c["start_time_us"]) for c in owner_contacts),
            default=0,
        )
        if multi and (not require_contact or owner_contacts):
            if traj and dur >= min_prov_us:
                state = "provisional"  # conceptual provisional_control
                reasons.append("CO_MOTION_SUPPORT")
            elif multi:
                state = "candidate"
    max_state = str(poss_cfg.get("max_state", "provisional"))
    if state == "provisional" and max_state == "candidate":
        state = "candidate"
    if state == "confirmed":
        state = "provisional"

    return {
        "kind": "owned",
        "state": state,
        "owner": owner,
        "contested": [],
        "reasons": reasons,
        "termination": "none",
        "target_relationship": target_rel,
        "prox_ids": prox_ids,
        "contact_ids": contact_ids,
        "evidence_refs": prox_ids + contact_ids,
        "coverage": 0.7 if state == "provisional" else 0.45,
    }


def _snap_key(snap: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        snap["state"],
        snap.get("owner"),
        tuple(snap.get("contested") or []),
        snap.get("termination"),
        tuple(snap.get("reasons") or []),
    )


def _emit_hypothesis(
    *,
    run_id: str,
    video_id: str,
    hypothesis_id: str,
    start_us: int,
    end_us: int,
    snap: Mapping[str, Any],
    policy_fingerprint: str,
    termination_reason: str,
    transition_from: str | None,
) -> dict[str, Any]:
    owner = snap.get("owner")
    contested = list(snap.get("contested") or [])
    state = str(snap["state"])
    if state == "confirmed":
        state = "provisional"
    ownership_kind = (
        "human_track"
        if owner is not None
        else (
            "unknown" if state in {"unknown", "not_evaluable", "rejected", "contested"} else "none"
        )
    )
    return {
        "run_id": run_id,
        "video_id": video_id,
        "hypothesis_id": hypothesis_id,
        "owner_human_track_id": owner,
        "owner_team_id": None,
        "ownership_kind": ownership_kind,
        "target_relationship": str(snap.get("target_relationship") or "unknown"),
        "start_time_us": int(start_us),
        "end_time_us": int(end_us),
        "possession_state": state,
        "contested_participant_track_ids": contested,
        "evidence_refs": list(snap.get("evidence_refs") or []),
        "contact_candidate_ids": list(snap.get("contact_ids") or []),
        "proximity_ids": list(snap.get("prox_ids") or []),
        "observed_coverage_ratio": snap.get("coverage"),
        "derived_coverage_ratio": None,
        "uncertainty": 0.55 if state in {"unknown", "contested", "not_evaluable"} else 0.35,
        "termination_reason": termination_reason,
        "transition_from_hypothesis_id": transition_from,
        "automatic_ceiling": "provisional",
        "implies_completed_pass": False,
        "implies_dribble_or_take_on": False,
        "implies_duel_or_aerial": False,
        "implies_box_touch": False,
        "implies_turnover": False,
        "penalty_area_presence_only": False,
        "manual_review_required": state in {"contested", "unknown"},
        "review_status": "unreviewed",
        "decision_log_json": None,
        "reason_codes": list(snap.get("reasons") or []),
        "quality_flags": [],
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def build_possession_hypotheses(
    proximity_rows: Sequence[Mapping[str, Any]],
    contact_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy_fingerprint: str,
) -> list[dict[str, Any]]:
    """Build half-open possession hypotheses from proximity + contact candidates."""
    if not proximity_rows:
        return []

    life = config.get("lifecycle") or {}
    hard_gap_us = int(life.get("hard_gap_us", 500_000))
    run_id = str(proximity_rows[0]["run_id"])
    video_id = str(proximity_rows[0]["video_id"])

    by_time: dict[int, list[Mapping[str, Any]]] = {}
    for p in proximity_rows:
        by_time.setdefault(int(p["video_time_us"]), []).append(p)
    times = sorted(by_time.keys())

    hypotheses: list[dict[str, Any]] = []
    seq = 0
    active_start: int | None = None
    active_snap: dict[str, Any] | None = None
    last_time: int | None = None
    prev_hyp_id: str | None = None

    def _close(end_us: int, termination: str) -> None:
        nonlocal seq, active_start, active_snap, prev_hyp_id
        if active_start is None or active_snap is None:
            return
        if end_us <= active_start:
            end_us = active_start + 1
        seq += 1
        hid = f"poss_{seq:04d}"
        term = (
            termination if termination != "none" else str(active_snap.get("termination") or "none")
        )
        # Owner change / contested close may set owner_transition
        hyp = _emit_hypothesis(
            run_id=run_id,
            video_id=video_id,
            hypothesis_id=hid,
            start_us=active_start,
            end_us=end_us,
            snap=active_snap,
            policy_fingerprint=policy_fingerprint,
            termination_reason=term,
            transition_from=prev_hyp_id,
        )
        hypotheses.append(hyp)
        prev_hyp_id = hid
        active_start = None
        active_snap = None

    for t in times:
        snap = _classify_instant(
            time_us=t, prox_at_t=by_time[t], contacts=contact_rows, config=config
        )
        if last_time is not None and (t - last_time) > hard_gap_us:
            _close(last_time + 1, "hard_gap")
        if active_snap is None:
            active_start = t
            active_snap = snap
        elif _snap_key(snap) != _snap_key(active_snap):
            # Owner transition needs evidence on the new hypothesis
            term = "none"
            if active_snap.get("owner") is not None and snap.get("owner") is not None:
                if active_snap.get("owner") != snap.get("owner"):
                    term = "owner_transition"
            elif str(snap.get("state")) == "contested":
                term = "contested"
            if str(active_snap.get("termination")) not in {"none", None}:
                term = str(active_snap["termination"])
            _close(t, term)
            active_start = t
            active_snap = snap
            # Attach transition evidence if owner changed
            if term == "owner_transition" and not snap.get("evidence_refs"):
                snap = dict(snap)
                snap["evidence_refs"] = list(snap.get("prox_ids") or []) + list(
                    snap.get("contact_ids") or []
                )
                active_snap = snap
        else:
            # Merge evidence ids across the open interval
            for key in ("prox_ids", "contact_ids", "evidence_refs"):
                merged = list(
                    dict.fromkeys(list(active_snap.get(key) or []) + list(snap.get(key) or []))
                )
                active_snap[key] = merged
        last_time = t

    if last_time is not None:
        term = str((active_snap or {}).get("termination") or "none")
        _close(last_time + 1, term)

    # Ensure owner transitions carry evidence_refs (validator)
    for i, h in enumerate(hypotheses):
        if i == 0:
            continue
        prev = hypotheses[i - 1]
        if prev.get("owner_human_track_id") != h.get("owner_human_track_id"):
            refs = list(h.get("evidence_refs") or [])
            if not refs:
                refs = list(h.get("proximity_ids") or []) + list(
                    h.get("contact_candidate_ids") or []
                )
            if not refs:
                refs = [f"transition_evidence_{h['hypothesis_id']}"]
            h["evidence_refs"] = refs
            if (
                str(h.get("termination_reason")) == "none"
                and prev.get("owner_human_track_id") is not None
            ):
                # previous closed with owner_transition typically
                pass
    return hypotheses


__all__ = ["build_possession_hypotheses"]
