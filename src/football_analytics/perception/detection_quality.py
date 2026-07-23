"""Stage 5E operational detection quality gates and review sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.perception.types import EntityType, ProcessingStatus, RoleLabel

NOT_EVALUATED_DETECTION = "NOT_EVALUATED_NO_REVIEWED_DETECTION_GROUND_TRUTH"

PROCESSED_SET = {
    ProcessingStatus.PROCESSED.value,
    ProcessingStatus.PROCESSED_NO_DETECTIONS.value,
}


@dataclass(frozen=True)
class QualityReport:
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


def compute_role_counts(attributes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {r.value: 0 for r in RoleLabel}
    abstain = 0
    for a in attributes:
        if a.get("entity_type") != EntityType.HUMAN.value:
            continue
        role = str(a.get("role_label") or RoleLabel.UNKNOWN.value)
        if role not in counts:
            counts[role] = 0
        counts[role] += 1
        if role == RoleLabel.UNKNOWN.value:
            abstain += 1
    counts["__abstention__"] = abstain
    return counts


def evaluate_detection_quality(
    *,
    detections: Sequence[Mapping[str, Any]],
    frame_status: Sequence[Mapping[str, Any]],
    attributes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    receipt_counts: Mapping[str, Any] | None = None,
    invalid_bbox_count: int = 0,
    dangling_fk_count: int = 0,
    duplicate_count: int = 0,
    receipt_mismatch_count: int = 0,
    has_reviewed_ground_truth: bool = False,
) -> QualityReport:
    """Operational quality only — never claims football accuracy."""
    thr = config["quality_thresholds"]
    findings: list[str] = []
    hard_fail = False

    n_status = len(frame_status)
    eligible = sum(
        1
        for r in frame_status
        if r.get("eligibility") in {"eligible", "conditionally_eligible"}
        or r["processing_status"] in PROCESSED_SET
        or r["processing_status"] == ProcessingStatus.SKIPPED.value
    )
    processed = sum(1 for r in frame_status if r["processing_status"] in PROCESSED_SET)
    skipped = sum(
        1 for r in frame_status if r["processing_status"] == ProcessingStatus.SKIPPED.value
    )
    failed = sum(1 for r in frame_status if r["processing_status"] == ProcessingStatus.FAILED.value)
    no_det = sum(
        1
        for r in frame_status
        if r["processing_status"] == ProcessingStatus.PROCESSED_NO_DETECTIONS.value
    )
    not_eligible = sum(
        1 for r in frame_status if r["processing_status"] == ProcessingStatus.NOT_ELIGIBLE.value
    )

    denom = max(eligible, 1)
    coverage = processed / denom if eligible else (1.0 if processed == 0 else 0.0)
    failed_rate = failed / max(n_status, 1)
    skipped_rate = skipped / max(n_status, 1)
    no_det_rate = no_det / max(processed, 1) if processed else 0.0

    role_counts = compute_role_counts(attributes)
    human_n = sum(1 for a in attributes if a.get("entity_type") == EntityType.HUMAN.value)
    abstain = int(role_counts.get("__abstention__", 0))
    abstain_rate = abstain / max(human_n, 1)

    if coverage < float(thr["min_eligible_processing_coverage"]):
        hard_fail = True
        findings.append("eligible_processing_coverage_below_threshold")
    if failed_rate > float(thr["max_failed_frame_rate"]):
        hard_fail = True
        findings.append("failed_frame_rate_exceeded")
    if dangling_fk_count > int(thr["max_dangling_fk"]):
        hard_fail = True
        findings.append("dangling_fk_detected")
    if duplicate_count > int(thr["max_duplicate_detection_keys"]):
        hard_fail = True
        findings.append("duplicate_detection_keys")
    if invalid_bbox_count > int(thr["max_invalid_bbox"]):
        hard_fail = True
        findings.append("invalid_bbox_detected")
    if receipt_mismatch_count > int(thr["max_receipt_mismatch"]):
        hard_fail = True
        findings.append("receipt_count_mismatch")

    if abstain_rate > float(thr["role_abstention_finding_rate"]) and human_n > 0:
        findings.append("high_role_abstention_rate")

    # Receipt consistency when provided
    if receipt_counts is not None:
        expected = {
            "processed_frame_count": processed,
            "skipped_frame_count": skipped,
            "failed_frame_count": failed,
            "processed_no_detection_count": no_det,
            "total_detection_count": len(detections),
            "human_detection_count": sum(
                1 for a in attributes if a.get("entity_type") == EntityType.HUMAN.value
            ),
            "ball_detection_count": sum(
                1 for a in attributes if a.get("entity_type") == EntityType.BALL.value
            ),
        }
        for k, v in expected.items():
            if receipt_counts.get(k) != v:
                hard_fail = True
                if "receipt_count_mismatch" not in findings:
                    findings.append("receipt_count_mismatch")
                break

    gt_status = "EVALUATED" if has_reviewed_ground_truth else NOT_EVALUATED_DETECTION
    if not has_reviewed_ground_truth:
        findings.append(NOT_EVALUATED_DETECTION)

    if hard_fail:
        status = "fail"
    elif findings:
        status = "pass_with_findings"
    else:
        status = "pass"

    metrics = {
        "frame_status_rows": n_status,
        "eligible_frame_count": eligible,
        "processed_frame_count": processed,
        "skipped_frame_count": skipped,
        "failed_frame_count": failed,
        "processed_no_detection_count": no_det,
        "not_eligible_frame_count": not_eligible,
        "eligible_processing_coverage": round(coverage, 6),
        "failed_frame_rate": round(failed_rate, 6),
        "skipped_frame_rate": round(skipped_rate, 6),
        "no_detection_frame_rate": round(no_det_rate, 6),
        "total_detection_count": len(detections),
        "human_detection_count": human_n,
        "ball_detection_count": sum(
            1 for a in attributes if a.get("entity_type") == EntityType.BALL.value
        ),
        "invalid_bbox_count": invalid_bbox_count,
        "dangling_fk_count": dangling_fk_count,
        "duplicate_detection_count": duplicate_count,
        "receipt_mismatch_count": receipt_mismatch_count,
        "role_counts": {k: v for k, v in role_counts.items() if not k.startswith("__")},
        "role_abstention_count": abstain,
        "role_abstention_rate": round(abstain_rate, 6),
    }
    return QualityReport(
        status=status,
        ground_truth_evaluation_status=gt_status,
        metrics=metrics,
        findings=tuple(findings),
        created_at_utc=_utc_now(),
    )


def build_detection_review_queue(
    *,
    attributes: Sequence[Mapping[str, Any]],
    frame_status: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    quality: QualityReport,
    run_id: str,
    video_id: str,
    policy_version: str = "1",
) -> dict[str, Any]:
    """Build sampled review queue — does not spam every unknown role."""
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
    if review["review_on_invalid_bbox"] and "invalid_bbox_detected" in quality.findings:
        _add(
            "invalid_bbox",
            ["INVALID_BBOX"],
            "high",
            {"run_id": run_id, "video_id": video_id},
        )
    if review["review_on_duplicate"] and "duplicate_detection_keys" in quality.findings:
        _add(
            "duplicate_detections",
            ["DUPLICATE_DETECTION"],
            "high",
            {"run_id": run_id, "video_id": video_id},
        )

    # Long no-detection stretches inside processed windows.
    if review["review_on_long_no_detection"]:
        streak = 0
        streak_start = None
        limit = int(review["long_no_detection_frames"])
        for r in sorted(frame_status, key=lambda x: int(x["frame_index"])):
            if r["processing_status"] == ProcessingStatus.PROCESSED_NO_DETECTIONS.value:
                if streak == 0:
                    streak_start = int(r["frame_index"])
                streak += 1
                if streak >= limit:
                    _add(
                        f"long_no_det_{streak_start}",
                        ["LONG_NO_DETECTION"],
                        "medium",
                        {
                            "start_frame_index": streak_start,
                            "length": streak,
                            "run_id": run_id,
                            "video_id": video_id,
                        },
                    )
                    streak = 0
                    streak_start = None
            else:
                streak = 0
                streak_start = None

    # Sample unknown human roles (no spam every unknown).
    if review["sample_unknown_roles"]:
        unknowns = [
            a
            for a in attributes
            if a.get("entity_type") == EntityType.HUMAN.value
            and a.get("role_label") == RoleLabel.UNKNOWN.value
        ]
        unknowns = sorted(
            unknowns,
            key=lambda a: (int(a["frame_index"]), int(a["detection_id"])),
        )
        stride = int(review["unknown_sample_stride"])
        max_items = int(review["max_unknown_review_items"])
        sampled = unknowns[::stride][:max_items]
        if review["do_not_spam_every_unknown"] and len(unknowns) > max_items:
            # Cap already applied; ensure we never enqueue all unknowns.
            pass
        for a in sampled:
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

    return {
        "schema_version": 1,
        "policy_version": policy_version,
        "run_id": run_id,
        "video_id": video_id,
        "items": items,
    }


__all__ = [
    "NOT_EVALUATED_DETECTION",
    "QualityReport",
    "compute_role_counts",
    "evaluate_detection_quality",
    "build_detection_review_queue",
]
