"""Stage 10B contact-candidate extraction from multi-frame proximity."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.interaction.types import CONTRACT_VERSION


def _speed_mps(dx: float, dy: float, dt_us: int) -> float | None:
    if dt_us <= 0:
        return None
    return math.hypot(dx, dy) / (dt_us / 1_000_000.0)


def extract_contact_candidates(
    proximity_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    policy_fingerprint: str,
) -> list[dict[str, Any]]:
    """Cluster eligible proximity into contact candidates. Single-frame never becomes contact."""
    contact_cfg = config.get("contact") or {}
    life = config.get("lifecycle") or {}
    min_frames = int(contact_cfg.get("min_frames", 2))
    min_duration_us = int(contact_cfg.get("min_duration_us", 40_000))
    max_gap_us = int(contact_cfg.get("max_internal_gap_us", 120_000))
    hard_gap_us = int(life.get("hard_gap_us", 500_000))
    max_state = str(contact_cfg.get("max_state", "provisional"))
    if max_state == "confirmed":
        max_state = "provisional"

    eligible = [
        r
        for r in proximity_rows
        if str(r.get("eligibility_status")) == "eligible"
        and str(r.get("evidence_level")) in {"candidate", "provisional"}
    ]
    by_pair: dict[tuple[str, str, int, Any], list[dict[str, Any]]] = {}
    for r in eligible:
        key = (
            str(r["run_id"]),
            str(r["video_id"]),
            int(r["human_track_id"]),
            r.get("ball_track_id"),
        )
        by_pair.setdefault(key, []).append(dict(r))

    out: list[dict[str, Any]] = []
    cand_seq = 0
    for key, rows in sorted(by_pair.items(), key=lambda kv: kv[0]):
        rows_sorted = sorted(rows, key=lambda r: int(r["video_time_us"]))
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for r in rows_sorted:
            if not current:
                current = [r]
                continue
            gap = int(r["video_time_us"]) - int(current[-1]["video_time_us"])
            if gap > hard_gap_us or gap > max_gap_us:
                clusters.append(current)
                current = [r]
            else:
                current.append(r)
        if current:
            clusters.append(current)

        for cluster in clusters:
            cand_seq += 1
            n = len(cluster)
            start_us = int(cluster[0]["video_time_us"])
            end_us = int(cluster[-1]["video_time_us"]) + 1
            duration = end_us - start_us
            multi = n >= min_frames and duration >= min_duration_us
            reasons: list[str] = []
            evidence_types = ["proximity_support"]
            traj_support = False
            if n >= 2:
                # relative motion / trajectory change support from pitch distances when usable
                d0 = cluster[0].get("pitch_distance_m")
                d1 = cluster[-1].get("pitch_distance_m")
                if d0 is not None and d1 is not None and abs(float(d1) - float(d0)) >= 0.3:
                    traj_support = True
                    evidence_types.append("trajectory_change_support")
            if multi:
                evidence_types.append("multi_frame_support")
                state = "provisional" if traj_support else "candidate"
                if state == "provisional" and max_state == "candidate":
                    state = "candidate"
            else:
                state = "rejected"
                reasons.append("SINGLE_FRAME_PROXIMITY")

            peak = cluster[n // 2]
            conf = None
            if multi:
                conf = 0.55 if traj_support else 0.4

            out.append(
                {
                    "run_id": key[0],
                    "video_id": key[1],
                    "candidate_id": f"contact_{cand_seq:04d}",
                    "human_track_id": key[2],
                    "ball_track_id": key[3],
                    "start_time_us": start_us,
                    "peak_time_us": int(peak["video_time_us"]),
                    "end_time_us": end_us,
                    "contact_state": state,
                    "evidence_types": evidence_types,
                    "proximity_support": True,
                    "trajectory_change_support": traj_support,
                    "multi_frame_support": multi,
                    "confidence": conf,
                    "review_status": "unreviewed",
                    "rejection_reason_codes": list(reasons) if state == "rejected" else [],
                    "implies_controlled_possession": False,
                    "implies_pass_or_event": False,
                    "implies_box_touch": False,
                    "ball_air_state": str(peak.get("ball_air_state", "unknown")),
                    "ball_candidate_status": str(peak.get("ball_candidate_status", "primary")),
                    "target_relationship": str(peak.get("target_relationship", "confirmed_target")),
                    "proximity_ids": [str(r["proximity_id"]) for r in cluster],
                    "evidence_refs": [str(r["proximity_id"]) for r in cluster],
                    "manual_review_required": False,
                    "reason_codes": reasons,
                    "quality_flags": [],
                    "policy_fingerprint": policy_fingerprint,
                    "provenance_json": None,
                    "contract_version": CONTRACT_VERSION,
                }
            )
    return out


__all__ = ["extract_contact_candidates", "_speed_mps"]
