"""Identity bundle validation: conflicts, leakage, FK, uniqueness (Stage 7A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.identity.assignments import validate_assignment_rows
from football_analytics.identity.evidence import (
    assert_no_face_biometric_evidence,
    validate_evidence_rows,
)
from football_analytics.identity.reid_links import validate_reid_links
from football_analytics.identity.types import AssignmentStatus, LeakageClass


@dataclass
class IdentityValidationResult:
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


def validate_identity_bundle(
    *,
    identity_evidence: Any | None = None,
    reid_candidate_links: Any | None = None,
    track_identity_assignments: Any | None = None,
    track_summaries: Any | None = None,
    policy: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
    allow_cross_video_manual: bool = False,
) -> IdentityValidationResult:
    """Validate identity contract bundle (synthetic or future production)."""
    result = IdentityValidationResult()
    _ = policy

    try:
        evidence = validate_evidence_rows(_rows(identity_evidence))
        assert_no_face_biometric_evidence(evidence)
    except Exception as exc:  # noqa: BLE001
        result.err(f"evidence: {exc}")
        evidence = _rows(identity_evidence)

    try:
        links = validate_reid_links(
            _rows(reid_candidate_links), allow_cross_video_manual=allow_cross_video_manual
        )
    except Exception as exc:  # noqa: BLE001
        result.err(f"reid_links: {exc}")
        links = _rows(reid_candidate_links)

    try:
        assignments = validate_assignment_rows(_rows(track_identity_assignments))
    except Exception as exc:  # noqa: BLE001
        result.err(f"assignments: {exc}")
        assignments = _rows(track_identity_assignments)

    evidence_ids = {e["evidence_id"] for e in evidence}
    for a in assignments:
        for eid in a.get("evidence_ids") or []:
            if eid not in evidence_ids:
                result.err(f"assignment {a.get('assignment_id')} dangling evidence FK: {eid}")
    for link in links:
        for eid in link.get("evidence_ids") or []:
            if eid not in evidence_ids:
                result.err(f"link {link.get('link_id')} dangling evidence FK: {eid}")

    # Track FK soft-check against summaries when provided
    summary_keys: set[tuple[str, str, int]] = set()
    for s in _rows(track_summaries):
        summary_keys.add((str(s["run_id"]), str(s["video_id"]), int(s["track_id"])))
    if summary_keys:
        for a in assignments:
            key = (str(a["run_id"]), str(a["video_id"]), int(a["track_id"]))
            if key not in summary_keys:
                result.err(f"assignment dangling track FK: {a.get('assignment_id')}")

    # Duplicate confirmed identity for same target overlapping intervals
    confirmed = [
        a for a in assignments if a.get("assignment_status") == AssignmentStatus.CONFIRMED.value
    ]
    by_target: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for a in confirmed:
        tkey = (str(a["run_id"]), str(a["video_id"]), str(a["target_player_id"]))
        by_target.setdefault(tkey, []).append(a)
    for _tkey, rows in by_target.items():
        if len(rows) < 2:
            continue
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                if int(a["track_id"]) == int(b["track_id"]):
                    continue
                # Overlap check
                if int(a["start_frame_index"]) <= int(b["end_frame_index"]) and int(
                    b["start_frame_index"]
                ) <= int(a["end_frame_index"]):
                    result.err(
                        "DUPLICATE_CONFIRMED_IDENTITY: "
                        f"{a['assignment_id']} vs {b['assignment_id']}"
                    )

    # One track → multiple non-revoked identities
    by_track: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for a in assignments:
        if a.get("assignment_status") in {
            AssignmentStatus.REVOKED.value,
            AssignmentStatus.REJECTED.value,
        }:
            continue
        trkey = (str(a["run_id"]), str(a["video_id"]), int(a["track_id"]))
        by_track.setdefault(trkey, []).append(a)
    for trkey, rows in by_track.items():
        targets = {str(r["target_player_id"]) for r in rows}
        if len(targets) > 1:
            result.err(f"TRACK_MULTI_IDENTITY: track={trkey[2]} targets={sorted(targets)}")

    # Leakage: evaluation must not mix into production decision features
    for e in evidence:
        lc = str(e.get("leakage_class"))
        if lc == LeakageClass.EVALUATION.value:
            # Evaluation-labeled evidence cannot be used for confirmed production
            for a in confirmed:
                if e["evidence_id"] in (a.get("evidence_ids") or []):
                    result.err(
                        f"LEAKAGE_SEPARATION_VIOLATION: evaluation evidence {e['evidence_id']}"
                    )
        flags = [str(x) for x in (e.get("quality_flags") or [])]
        if "tuning_and_independent_eval" in flags:
            result.err("LEAKAGE_SEPARATION_VIOLATION: same track/frame tuning+eval")

    # Confirmed from alone-insufficient single evidence
    for a in confirmed:
        eids = list(a.get("evidence_ids") or [])
        if len(eids) == 1:
            ev = next((e for e in evidence if e["evidence_id"] == eids[0]), None)
            if ev is not None and str(ev.get("evidence_type")) in {
                "jersey_number",
                "team_assignment",
                "role_consistency",
                "appearance_similarity",
            }:
                result.err(
                    f"SINGLE_WEAK_CANNOT_CONFIRM: assignment {a['assignment_id']} "
                    f"via {ev['evidence_type']}"
                )

    if receipt is not None:
        ac = receipt.get("assignment_counts") or {}
        if int(ac.get("total", -1)) != len(assignments):
            result.err("RECEIPT_COUNT_MISMATCH: assignment total")

    return result


def assert_no_evaluation_leakage(
    evidence: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> None:
    """Raise if evaluation-labeled evidence feeds confirmed production decisions."""
    result = validate_identity_bundle(
        identity_evidence=list(evidence),
        track_identity_assignments=list(assignments),
    )
    leaks = [e for e in result.errors if "LEAKAGE" in e]
    if leaks:
        raise ValueError("; ".join(leaks))


__all__ = [
    "IdentityValidationResult",
    "validate_identity_bundle",
    "assert_no_evaluation_leakage",
]
