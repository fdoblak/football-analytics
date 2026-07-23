"""Stage 7A ReID / identity / target-player contracts (no inference)."""

from __future__ import annotations

from football_analytics.identity.assignments import (
    build_revocation,
    build_supersede,
    validate_assignment_record,
    validate_assignment_rows,
)
from football_analytics.identity.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_JERSEY_OBSERVATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    EXPECTED_TEAM_ASSIGNMENTS_FP,
    EXPECTED_TRACK_LIFECYCLE_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    IDENTITY_ARROW_CONTRACTS,
    assert_frozen_upstream_fingerprints,
    assert_identity_contracts_registered,
    compile_identity_schemas,
    identity_schema_fingerprints,
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.evaluation import (
    NOT_EVALUATED_IDENTITY,
    NULL_METRICS,
    IdentityEvaluationReport,
    evaluate_identity,
)
from football_analytics.identity.evidence import (
    assert_no_face_biometric_evidence,
    validate_evidence_record,
    validate_evidence_rows,
)
from football_analytics.identity.metric_eligibility import (
    customer_metric_allowed,
    resolve_metric_eligibility,
)
from football_analytics.identity.policy import (
    decide_assignment_status,
    load_identity_policy,
    policy_fingerprint,
)
from football_analytics.identity.receipt import (
    build_synthetic_receipt,
    build_synthetic_target_request,
    validate_receipt_payload,
    validate_request_payload,
)
from football_analytics.identity.reid_links import validate_reid_link, validate_reid_links
from football_analytics.identity.review_audit import (
    append_audit_log,
    sample_review_items,
    should_enqueue_review,
    validate_audit_entry,
)
from football_analytics.identity.target_profile import validate_target_player_request
from football_analytics.identity.types import (
    ALONE_INSUFFICIENT_TYPES,
    AssignmentStatus,
    EvidenceType,
    IdentityContractError,
    LeakageClass,
    MetricEligibility,
    ReliabilityTier,
    TargetScope,
)
from football_analytics.identity.validation import (
    IdentityValidationResult,
    assert_no_evaluation_leakage,
    validate_identity_bundle,
)

__all__ = [
    "ALONE_INSUFFICIENT_TYPES",
    "AssignmentStatus",
    "EvidenceType",
    "EXPECTED_DETECTIONS_FP",
    "EXPECTED_JERSEY_OBSERVATIONS_FP",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "EXPECTED_TEAM_ASSIGNMENTS_FP",
    "EXPECTED_TRACK_LIFECYCLE_FP",
    "EXPECTED_TRACK_OBSERVATIONS_FP",
    "EXPECTED_TRACK_SUMMARIES_FP",
    "IDENTITY_ARROW_CONTRACTS",
    "IdentityContractError",
    "IdentityEvaluationReport",
    "IdentityValidationResult",
    "LeakageClass",
    "MetricEligibility",
    "NOT_EVALUATED_IDENTITY",
    "NULL_METRICS",
    "ReliabilityTier",
    "TargetScope",
    "append_audit_log",
    "assert_frozen_upstream_fingerprints",
    "assert_identity_contracts_registered",
    "assert_no_evaluation_leakage",
    "assert_no_face_biometric_evidence",
    "build_revocation",
    "build_supersede",
    "build_synthetic_receipt",
    "build_synthetic_target_request",
    "compile_identity_schemas",
    "customer_metric_allowed",
    "decide_assignment_status",
    "evaluate_identity",
    "identity_schema_fingerprints",
    "load_identity_json_schema",
    "load_identity_policy",
    "policy_fingerprint",
    "resolve_metric_eligibility",
    "sample_review_items",
    "should_enqueue_review",
    "validate_against_json_schema",
    "validate_assignment_record",
    "validate_assignment_rows",
    "validate_audit_entry",
    "validate_evidence_record",
    "validate_evidence_rows",
    "validate_identity_bundle",
    "validate_receipt_payload",
    "validate_reid_link",
    "validate_reid_links",
    "validate_request_payload",
    "validate_target_player_request",
]
