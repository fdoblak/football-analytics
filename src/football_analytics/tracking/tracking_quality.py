"""Stage 6D operational tracking quality gates and review sampling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.tracking.evaluation import NOT_EVALUATED_TRACKING
from football_analytics.tracking.types import LifecycleState, TrackEntityType

# Keep human/ball-specific codes available in provenance.
NOT_EVALUATED_HUMAN_TRACKING = "NOT_EVALUATED_NO_REVIEWED_" "HUMAN_TRACKING_GROUND_TRUTH"
NOT_EVALUATED_BALL_TRACKING = "NOT_EVALUATED_NO_REVIEWED_" "BALL_TRACKING_GROUND_TRUTH"


@dataclass(frozen=True)
class TrackingQualityReport:
    status: str
    ground_truth_evaluation_status: str
    metrics: dict[str, Any]
    findings: tuple[str, ...]
    created_at_utc: str

    def to_dict(self, *, run_id: str, video_id: str, config_fingerprint: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "video_id": video_id,
            "config_fingerprint": config_fingerprint,
            "status": self.status,
            "ground_truth_evaluation_status": self.ground_truth_evaluation_status,
            "metrics": dict(self.metrics),
            "findings": list(self.findings),
            "created_at_utc": self.created_at_utc,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fragmentation_score(lifecycle: Sequence[Mapping[str, Any]]) -> float:
    """Approximate fragmentation: terminated tracks / unique tracks."""
    by_track: dict[tuple[Any, Any, int], list[str]] = defaultdict(list)
    for r in lifecycle:
        by_track[(r["run_id"], r["video_id"], int(r["track_id"]))].append(str(r["lifecycle_state"]))
    if not by_track:
        return 0.0
    terminated = sum(1 for states in by_track.values() if LifecycleState.TERMINATED.value in states)
    return terminated / max(len(by_track), 1)


def _gap_stats(
    observations: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]] | None
) -> tuple[float, float]:
    time_by_frame: dict[tuple[Any, Any, int], int] = {}
    for f in frames or []:
        time_by_frame[(f["run_id"], f["video_id"], int(f["frame_index"]))] = int(f["video_time_us"])
    by_track: dict[tuple[Any, Any, int], list[int]] = defaultdict(list)
    for o in observations:
        fi = int(o["frame_index"])
        t = time_by_frame.get((o["run_id"], o["video_id"], fi), fi * 40000)
        by_track[(o["run_id"], o["video_id"], int(o["track_id"]))].append(t)
    gaps: list[int] = []
    for times in by_track.values():
        ordered = sorted(times)
        for i in range(1, len(ordered)):
            g = ordered[i] - ordered[i - 1]
            if g > 0:
                gaps.append(g)
    if not gaps:
        return 0.0, 0.0
    return float(max(gaps)), float(sum(gaps) / len(gaps))


def evaluate_tracking_quality(
    *,
    observations: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    lifecycle: Sequence[Mapping[str, Any]],
    detection_attributes: Sequence[Mapping[str, Any]],
    primary_sidecar: Sequence[Mapping[str, Any]] | None,
    frames: Sequence[Mapping[str, Any]] | None,
    analysis_windows: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    receipt_counts: Mapping[str, Any] | None = None,
    dangling_fk_count: int = 0,
    duplicate_count: int = 0,
    invalid_bbox_count: int = 0,
    cross_cut_violation_count: int = 0,
    terminated_reopen_count: int = 0,
    receipt_mismatch_count: int = 0,
    has_reviewed_ground_truth: bool = False,
) -> TrackingQualityReport:
    """Operational quality only — never claims football tracking accuracy."""
    thr = config["quality_thresholds"]
    findings: list[str] = []
    hard_fail = False
    _ = summaries

    windows = list(analysis_windows or [])
    eligible_frames: set[int] = set()
    for w in windows:
        if str(w.get("tracking_eligibility")) in {"eligible", "conditionally_eligible"}:
            for fi in range(int(w["start_frame_index"]), int(w["end_frame_index_exclusive"])):
                eligible_frames.add(fi)
    if not eligible_frames and frames:
        eligible_frames = {int(f["frame_index"]) for f in frames}

    tracked_frames = {int(o["frame_index"]) for o in observations}
    coverage = (
        len(tracked_frames & eligible_frames) / max(len(eligible_frames), 1)
        if eligible_frames
        else (1.0 if not observations else 0.0)
    )

    observed = sum(1 for o in observations if o["observation_state"] == "observed")
    predicted = sum(1 for o in observations if o["observation_state"] == "predicted")
    interpolated = sum(1 for o in observations if o["observation_state"] == "interpolated")
    total_obs = max(len(observations), 1)
    predicted_ratio = (predicted + interpolated) / total_obs

    assigned = {
        (o["frame_index"], o["detection_id"])
        for o in observations
        if o["observation_state"] == "observed" and o.get("detection_id") is not None
    }
    human_dets = sum(
        1 for a in detection_attributes if a.get("entity_type") == TrackEntityType.HUMAN.value
    )
    ball_dets = sum(
        1 for a in detection_attributes if a.get("entity_type") == TrackEntityType.BALL.value
    )
    assignment_coverage = len(assigned) / max(human_dets + ball_dets, 1)

    frag = _fragmentation_score(lifecycle)
    max_gap, mean_gap = _gap_stats(observations, frames)

    human_attrs = [
        a for a in detection_attributes if a.get("entity_type") == TrackEntityType.HUMAN.value
    ]
    abstain = sum(1 for a in human_attrs if str(a.get("role_label") or "unknown") == "unknown")
    abstain_rate = abstain / max(len(human_attrs), 1)

    primary = list(primary_sidecar or [])
    amb = sum(1 for f in primary if str(f.get("status") or "") == "ambiguous")
    no_cand = sum(
        1
        for f in primary
        if str(f.get("status") or "") in {"no_candidate", "none", "empty"}
        or (
            f.get("primary_track_id") is None
            and str(f.get("status") or "") not in {"primary", "ambiguous"}
        )
    )
    primary_n = sum(1 for f in primary if str(f.get("status") or "") == "primary")
    ball_denom = max(primary_n + amb + no_cand, 1)
    amb_rate = amb / ball_denom

    if coverage < float(thr["min_eligible_tracking_coverage"]):
        hard_fail = True
        findings.append("eligible_tracking_coverage_below_threshold")
    if dangling_fk_count > int(thr["max_dangling_fk"]):
        hard_fail = True
        findings.append("dangling_fk_detected")
    if duplicate_count > int(thr["max_duplicate_keys"]):
        hard_fail = True
        findings.append("duplicate_track_keys")
    if invalid_bbox_count > int(thr["max_invalid_bbox"]):
        hard_fail = True
        findings.append("invalid_bbox_detected")
    if cross_cut_violation_count > int(thr["max_cross_cut_violations"]):
        hard_fail = True
        findings.append("cross_cut_continuation")
    if terminated_reopen_count > int(thr["max_terminated_reopen"]):
        hard_fail = True
        findings.append("terminated_reopen")
    if receipt_mismatch_count > int(thr["max_receipt_mismatch"]):
        hard_fail = True
        findings.append("receipt_count_mismatch")
    if predicted_ratio > float(thr["max_predicted_ratio"]) and observations:
        hard_fail = True
        findings.append("predicted_ratio_exceeded")

    if frag > float(thr["fragmentation_finding_rate"]) and lifecycle:
        findings.append("high_track_fragmentation")
    if abstain_rate > float(thr["role_abstention_finding_rate"]) and human_attrs:
        findings.append("high_role_abstention_rate")
    if amb_rate > float(thr["ball_ambiguity_finding_rate"]) and primary:
        findings.append("high_ball_ambiguity_rate")

    if receipt_counts is not None:
        expected_keys = {
            "observed_count": observed,
            "predicted_count": predicted,
            "interpolated_count": interpolated,
        }
        for k, v in expected_keys.items():
            if receipt_counts.get(k) is not None and receipt_counts.get(k) != v:
                hard_fail = True
                if "receipt_count_mismatch" not in findings:
                    findings.append("receipt_count_mismatch")
                break

    gt_status = "EVALUATED" if has_reviewed_ground_truth else NOT_EVALUATED_TRACKING
    if not has_reviewed_ground_truth:
        findings.append(NOT_EVALUATED_TRACKING)

    if hard_fail:
        status = "fail"
    elif findings:
        status = "pass_with_findings"
    else:
        status = "pass"

    metrics = {
        "eligible_frame_count": len(eligible_frames),
        "tracked_frame_count": len(tracked_frames),
        "eligible_tracking_coverage": round(coverage, 6),
        "detection_assignment_coverage": round(assignment_coverage, 6),
        "observed_count": observed,
        "predicted_count": predicted,
        "interpolated_count": interpolated,
        "predicted_ratio": round(predicted_ratio, 6),
        "track_fragmentation": round(frag, 6),
        "max_gap_us": max_gap,
        "mean_gap_us": round(mean_gap, 3),
        "dangling_fk_count": dangling_fk_count,
        "duplicate_count": duplicate_count,
        "invalid_bbox_count": invalid_bbox_count,
        "cross_cut_violation_count": cross_cut_violation_count,
        "terminated_reopen_count": terminated_reopen_count,
        "receipt_mismatch_count": receipt_mismatch_count,
        "role_abstention_count": abstain,
        "role_abstention_rate": round(abstain_rate, 6),
        "primary_ball_frames": primary_n,
        "ambiguous_ball_frames": amb,
        "no_candidate_ball_frames": no_cand,
        "ball_ambiguity_rate": round(amb_rate, 6),
        "human_detection_count": human_dets,
        "ball_detection_count": ball_dets,
    }
    return TrackingQualityReport(
        status=status,
        ground_truth_evaluation_status=gt_status,
        metrics=metrics,
        findings=tuple(findings),
        created_at_utc=_utc_now(),
    )


def build_tracking_review_queue(
    *,
    observations: Sequence[Mapping[str, Any]],
    lifecycle: Sequence[Mapping[str, Any]],
    detection_attributes: Sequence[Mapping[str, Any]],
    primary_sidecar: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    quality: TrackingQualityReport,
    run_id: str,
    video_id: str,
    policy_version: str = "1",
) -> dict[str, Any]:
    """Build sampled review queue — does not spam every unknown/empty frame."""
    review = config["review_policy"]
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item_id: str, reasons: list[str], priority: str, evidence: Mapping[str, Any]) -> None:
        if item_id in seen:
            return
        seen.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "reason_codes": reasons,
                "priority": priority,
                "evidence": dict(evidence),
                "status": "pending",
            }
        )

    if review["review_on_receipt_mismatch"] and "receipt_count_mismatch" in quality.findings:
        _add(
            "receipt_mismatch",
            ["RECEIPT_COUNT_MISMATCH"],
            "high",
            {"run_id": run_id, "video_id": video_id},
        )
    if review["review_on_fk_duplicate"] and (
        "dangling_fk_detected" in quality.findings or "duplicate_track_keys" in quality.findings
    ):
        _add(
            "fk_or_duplicate",
            ["FK_OR_DUPLICATE"],
            "high",
            {"run_id": run_id, "video_id": video_id},
        )
    if review["review_on_cross_cut"] and "cross_cut_continuation" in quality.findings:
        _add(
            "cross_cut",
            ["CROSS_CUT_CONTINUATION"],
            "high",
            {"run_id": run_id, "video_id": video_id},
        )

    if review["sample_role_abstention"]:
        unknowns = [
            a
            for a in detection_attributes
            if a.get("entity_type") == TrackEntityType.HUMAN.value
            and str(a.get("role_label") or "unknown") == "unknown"
        ]
        unknowns = sorted(unknowns, key=lambda a: (int(a["frame_index"]), int(a["detection_id"])))
        stride = int(review["role_sample_stride"])
        max_items = int(review["max_role_review_items"])
        for a in unknowns[::stride][:max_items]:
            _add(
                f"unknown_role_f{a['frame_index']}_d{a['detection_id']}",
                ["ROLE_UNKNOWN_ABSTENTION"],
                "low",
                {
                    "frame_index": a["frame_index"],
                    "detection_id": a["detection_id"],
                    "run_id": run_id,
                    "video_id": video_id,
                },
            )

    if review["sample_fragmentation"] and "high_track_fragmentation" in quality.findings:
        term_tracks = []
        by_track: dict[int, list[str]] = defaultdict(list)
        for r in lifecycle:
            by_track[int(r["track_id"])].append(str(r["lifecycle_state"]))
        for tid, states in sorted(by_track.items()):
            if LifecycleState.TERMINATED.value in states:
                term_tracks.append(tid)
        for tid in term_tracks[: int(review["max_fragmentation_review_items"])]:
            _add(
                f"fragment_track_{tid}",
                ["TRACK_FRAGMENTATION"],
                "medium",
                {"track_id": tid, "run_id": run_id, "video_id": video_id},
            )

    if review["sample_ball_ambiguity"]:
        amb_frames = [
            f for f in (primary_sidecar or []) if str(f.get("status") or "") == "ambiguous"
        ]
        amb_frames = sorted(amb_frames, key=lambda f: int(f.get("frame_index", 0)))
        for f in amb_frames[: int(review["max_ambiguity_review_items"])]:
            _add(
                f"ball_amb_f{f.get('frame_index')}",
                ["BALL_PRIMARY_AMBIGUOUS"],
                "medium",
                {
                    "frame_index": f.get("frame_index"),
                    "run_id": run_id,
                    "video_id": video_id,
                    "note": "ambiguous primary is not upgraded to true ball",
                },
            )

    if review["sample_long_no_ball"] and primary_sidecar:
        streak = 0
        streak_start = None
        limit = int(review["long_no_ball_frames"])
        for f in sorted(primary_sidecar, key=lambda x: int(x.get("frame_index", 0))):
            status = str(f.get("status") or "")
            empty = status in {"no_candidate", "none", "empty"} or (
                f.get("primary_track_id") is None and status != "ambiguous"
            )
            if empty:
                if streak == 0:
                    streak_start = int(f.get("frame_index", 0))
                streak += 1
                if streak >= limit:
                    _add(
                        f"long_no_ball_{streak_start}",
                        ["LONG_NO_BALL"],
                        "medium",
                        {
                            "start_frame_index": streak_start,
                            "length": streak,
                            "run_id": run_id,
                            "video_id": video_id,
                        },
                    )
                    if review["do_not_spam_empty_frames"]:
                        break
            else:
                streak = 0
                streak_start = None

    if review["sample_high_predicted"] and "predicted_ratio_exceeded" in quality.findings:
        _add(
            "high_predicted_ratio",
            ["HIGH_PREDICTED_RATIO"],
            "medium",
            {
                "predicted_ratio": quality.metrics.get("predicted_ratio"),
                "run_id": run_id,
                "video_id": video_id,
            },
        )

    # Silence unused obs when do_not_spam — still referenced for API symmetry.
    _ = observations

    return {
        "schema_version": 1,
        "policy_version": policy_version,
        "run_id": run_id,
        "video_id": video_id,
        "items": items,
    }


__all__ = [
    "NOT_EVALUATED_TRACKING",
    "NOT_EVALUATED_HUMAN_TRACKING",
    "NOT_EVALUATED_BALL_TRACKING",
    "TrackingQualityReport",
    "evaluate_tracking_quality",
    "build_tracking_review_queue",
]
