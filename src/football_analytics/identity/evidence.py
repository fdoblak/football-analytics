"""Identity evidence record helpers (Stage 7A — contract validation only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.identity.types import (
    CONTRACT_VERSION,
    EvidencePolarity,
    EvidenceType,
    IdentityContractError,
    LeakageClass,
    ReliabilityTier,
    ReviewStatus,
)

REQUIRED_EVIDENCE_KEYS = frozenset(
    {
        "run_id",
        "video_id",
        "evidence_id",
        "evidence_type",
        "reliability_tier",
        "polarity",
        "review_status",
        "producer",
        "producer_version",
        "reason_codes",
        "quality_flags",
        "leakage_class",
        "contract_version",
    }
)


def validate_evidence_record(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_EVIDENCE_KEYS - set(row)
    if missing:
        raise IdentityContractError(f"evidence missing keys: {sorted(missing)}")
    et = str(row["evidence_type"])
    if et not in {e.value for e in EvidenceType}:
        raise IdentityContractError(f"unknown evidence_type: {et}")
    if et == "face" or "biometric" in et or "face_recognition" in et:
        raise IdentityContractError("FACE_BIOMETRIC_FORBIDDEN")
    if str(row["reliability_tier"]) not in {t.value for t in ReliabilityTier}:
        raise IdentityContractError("invalid reliability_tier")
    if str(row["polarity"]) not in {p.value for p in EvidencePolarity}:
        raise IdentityContractError("invalid polarity")
    if str(row["review_status"]) not in {r.value for r in ReviewStatus}:
        raise IdentityContractError("invalid review_status")
    if str(row["leakage_class"]) not in {c.value for c in LeakageClass}:
        raise IdentityContractError("invalid leakage_class")
    if int(row["contract_version"]) != CONTRACT_VERSION:
        raise IdentityContractError("contract_version mismatch")
    start_f = row.get("start_frame_index")
    end_f = row.get("end_frame_index")
    if start_f is not None and end_f is not None and int(end_f) < int(start_f):
        raise IdentityContractError("evidence interval invalid")
    return dict(row)


def validate_evidence_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = [validate_evidence_record(r) for r in rows]
    ids = [r["evidence_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise IdentityContractError("duplicate evidence_id")
    return out


def assert_no_face_biometric_evidence(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        et = str(r.get("evidence_type", "")).lower()
        flags = [str(x).lower() for x in (r.get("quality_flags") or [])]
        reasons = [str(x).lower() for x in (r.get("reason_codes") or [])]
        blob = " ".join([et, *flags, *reasons])
        if "face" in blob or "biometric" in blob:
            raise IdentityContractError("FACE_BIOMETRIC_FORBIDDEN")


__all__ = [
    "REQUIRED_EVIDENCE_KEYS",
    "validate_evidence_record",
    "validate_evidence_rows",
    "assert_no_face_biometric_evidence",
]
