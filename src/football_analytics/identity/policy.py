"""Strict identity evidence policy loader + decision matrix (Stage 7A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.identity.types import (
    ALONE_INSUFFICIENT_TYPES,
    AssignmentStatus,
    EvidencePolarity,
    EvidenceType,
    PolicyError,
    ReliabilityTier,
)

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "policy_version",
        "config_version",
        "id_namespace",
        "safety",
        "reliability_tiers",
        "evidence_types",
        "assignment_statuses",
        "target_scopes",
        "metric_eligibility",
        "decision_matrix",
        "confidence_policy",
        "metric_eligibility_rules",
        "review_triggers",
        "leakage_separation",
        "reference_contracts",
        "notes",
    }
)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{label} must be a mapping")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{label} must be a bool")
    return value


def _require_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PolicyError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise PolicyError(f"{label} entries must be non-empty strings")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def default_identity_policy_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "identity" / "identity_evidence_policy.yaml"


def load_identity_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_identity_policy_path(project_root=project_root)
    if p.is_symlink():
        raise PolicyError(f"symlink rejected: {p}")
    raw_bytes = p.read_bytes()
    if len(raw_bytes) > MAX_CONFIG_BYTES:
        raise PolicyError("policy exceeds max bytes")
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise PolicyError("policy root must be a mapping")
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise PolicyError(f"policy missing keys: {sorted(missing)}")
    if int(data["config_version"]) != CONFIG_VERSION:
        raise PolicyError("unsupported config_version")

    safety = _require_mapping(data["safety"], label="safety")
    for key in (
        "face_recognition_forbidden",
        "biometric_identity_forbidden",
        "cross_video_auto_link_forbidden",
        "physical_track_merge_forbidden",
        "single_weak_evidence_cannot_confirm",
        "jersey_number_alone_insufficient",
        "team_or_kit_color_alone_insufficient",
        "role_alone_insufficient",
        "appearance_score_alone_cannot_confirm",
        "auto_target_selection_forbidden",
        "revoked_not_metric_eligible",
        "append_only_manual_audit",
    ):
        if key not in safety:
            raise PolicyError(f"safety missing {key}")
        if _require_bool(safety[key], label=f"safety.{key}") is not True:
            raise PolicyError(f"safety.{key} must be true")

    tiers = set(_require_str_list(data["reliability_tiers"], label="reliability_tiers"))
    expected_tiers = {t.value for t in ReliabilityTier}
    if tiers != expected_tiers:
        raise PolicyError(f"reliability_tiers mismatch: {sorted(tiers)}")

    etypes = set(_require_str_list(data["evidence_types"], label="evidence_types"))
    expected_types = {t.value for t in EvidenceType}
    if etypes != expected_types:
        raise PolicyError(f"evidence_types mismatch: {sorted(etypes)}")

    statuses = set(_require_str_list(data["assignment_statuses"], label="assignment_statuses"))
    expected_status = {s.value for s in AssignmentStatus}
    if statuses != expected_status:
        raise PolicyError(f"assignment_statuses mismatch: {sorted(statuses)}")

    matrix = _require_mapping(data["decision_matrix"], label="decision_matrix")
    if matrix.get("auto_confirmed_forbidden") is not True:
        raise PolicyError("auto_confirmed_forbidden must be true")
    min_prov = matrix.get("min_independent_supporting_for_provisional")
    if not isinstance(min_prov, int) or min_prov < 2:
        raise PolicyError("min_independent_supporting_for_provisional must be >= 2")

    leakage = _require_mapping(data["leakage_separation"], label="leakage_separation")
    for key in (
        "require_leakage_class_on_evidence",
        "evaluation_label_must_not_enter_features",
        "synthetic_fixtures_not_accuracy_claims",
    ):
        if leakage.get(key) is not True:
            raise PolicyError(f"leakage_separation.{key} must be true")

    return _deep_freeze(data)


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(policy))


def decide_assignment_status(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    within_manual_anchor_scope: bool = False,
) -> tuple[str, list[str]]:
    """Apply decision matrix to an evidence set (contract-level; no inference).

    Returns (assignment_status, reason_codes).
    Never auto-confirms without a scoped manual_verified anchor.
    """
    _ = policy
    reasons: list[str] = []
    if not evidence_rows:
        return AssignmentStatus.UNKNOWN.value, ["INSUFFICIENT_COVERAGE"]

    supporting = [
        e
        for e in evidence_rows
        if str(e.get("polarity")) == EvidencePolarity.SUPPORTS.value
        and str(e.get("reliability_tier"))
        not in {ReliabilityTier.CONFLICTING.value, ReliabilityTier.UNAVAILABLE.value}
    ]
    conflicting = [
        e
        for e in evidence_rows
        if str(e.get("polarity")) == EvidencePolarity.CONFLICTS.value
        or str(e.get("reliability_tier")) == ReliabilityTier.CONFLICTING.value
    ]
    if conflicting:
        reasons.append("HARD_EVIDENCE_CONFLICT")
        return AssignmentStatus.REJECTED.value, reasons

    manual = [
        e
        for e in supporting
        if str(e.get("evidence_type")) == EvidenceType.MANUAL_TRACK_ANCHOR.value
        and str(e.get("reliability_tier")) == ReliabilityTier.MANUAL_VERIFIED.value
    ]
    if manual and within_manual_anchor_scope:
        reasons.append("MANUAL_VERIFIED_IN_SCOPE")
        return AssignmentStatus.CONFIRMED.value, reasons
    if manual and not within_manual_anchor_scope:
        reasons.append("MANUAL_VERIFIED_OUT_OF_SCOPE")
        return AssignmentStatus.CANDIDATE.value, reasons

    # Alone-insufficient / single weak
    if len(supporting) == 1:
        sole = supporting[0]
        et = str(sole.get("evidence_type"))
        tier = str(sole.get("reliability_tier"))
        if et in ALONE_INSUFFICIENT_TYPES or tier == ReliabilityTier.WEAK.value:
            if et == EvidenceType.JERSEY_NUMBER.value:
                reasons.append("JERSEY_ALONE_INSUFFICIENT")
            elif et == EvidenceType.TEAM_ASSIGNMENT.value:
                reasons.append("TEAM_ALONE_INSUFFICIENT")
            elif et == EvidenceType.ROLE_CONSISTENCY.value:
                reasons.append("ROLE_ALONE_INSUFFICIENT")
            elif et == EvidenceType.APPEARANCE_SIMILARITY.value:
                reasons.append("APPEARANCE_ALONE_INSUFFICIENT")
            else:
                reasons.append("SINGLE_WEAK_CANNOT_CONFIRM")
            return AssignmentStatus.CANDIDATE.value, reasons

    # Distinct evidence types count as independent supporting cues
    independent_types = {str(e.get("evidence_type")) for e in supporting}
    independent_types.discard(EvidenceType.UNKNOWN.value)
    min_prov = int(policy["decision_matrix"]["min_independent_supporting_for_provisional"])
    if len(independent_types) >= min_prov:
        reasons.append("MULTI_SUPPORTING_PROVISIONAL")
        return AssignmentStatus.PROVISIONAL.value, reasons

    reasons.append("INSUFFICIENT_INDEPENDENT_EVIDENCE")
    return AssignmentStatus.CANDIDATE.value, reasons


__all__ = [
    "CONFIG_VERSION",
    "PolicyError",
    "default_identity_policy_path",
    "load_identity_policy",
    "policy_fingerprint",
    "decide_assignment_status",
]
