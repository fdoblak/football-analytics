"""Stage 6 multi-object tracking: contracts (6A), human MOT (6B), ball MOT (6C)."""

from football_analytics.tracking.ball_tracking_evaluation import (
    NOT_EVALUATED_BALL_TRACKING,
    evaluate_ball_tracking,
)
from football_analytics.tracking.bbox_rules import validate_track_bbox
from football_analytics.tracking.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    JSON_SCHEMA_NAMES,
    TRACK_CONTRACT_NAMES,
    TRACK_LIFECYCLE_CONTRACT,
    assert_track_contracts_registered,
    assert_v1_track_fingerprints_unchanged,
    compile_tracking_schemas,
    load_all_tracking_contracts,
    load_tracking_contract,
    load_tracking_json_schema,
    tracking_schema_fingerprints,
    validate_against_json_schema,
)
from football_analytics.tracking.evaluation import (
    NOT_EVALUATED_TRACKING,
    evaluate_tracking,
)
from football_analytics.tracking.human_tracking_evaluation import (
    NOT_EVALUATED_HUMAN_TRACKING,
    evaluate_human_tracking,
)
from football_analytics.tracking.lifecycle import (
    assert_transition_allowed,
    validate_lifecycle_sequence,
)
from football_analytics.tracking.policy import (
    default_tracking_policy_path,
    load_tracking_policy,
    policy_fingerprint,
)
from football_analytics.tracking.receipt import (
    build_synthetic_receipt,
    build_synthetic_request,
    recount_receipt_from_tables,
    validate_receipt_payload,
    validate_request_payload,
)
from football_analytics.tracking.time_rules import (
    gap_us,
    require_monotonic_times,
    resolve_video_time_us,
)
from football_analytics.tracking.track_ids import TrackIdAllocator, allocate_hash_track_id
from football_analytics.tracking.types import (
    CONTRACT_VERSION,
    ERROR_CODES,
    OBSERVATION_STATE_MAP,
    GapReason,
    LifecycleState,
    ObservationSource,
    ReceiptStatus,
    TrackEntityType,
    TrackingContractError,
    TrackingError,
    TransitionError,
    observation_state_for_source,
)
from football_analytics.tracking.validation import validate_track_bundle

__all__ = [
    "CONTRACT_VERSION",
    "ERROR_CODES",
    "EXPECTED_DETECTIONS_FP",
    "EXPECTED_TRACK_OBSERVATIONS_FP",
    "EXPECTED_TRACK_SUMMARIES_FP",
    "JSON_SCHEMA_NAMES",
    "NOT_EVALUATED_BALL_TRACKING",
    "NOT_EVALUATED_HUMAN_TRACKING",
    "NOT_EVALUATED_TRACKING",
    "OBSERVATION_STATE_MAP",
    "TRACK_CONTRACT_NAMES",
    "TRACK_LIFECYCLE_CONTRACT",
    "GapReason",
    "LifecycleState",
    "ObservationSource",
    "ReceiptStatus",
    "TrackEntityType",
    "TrackIdAllocator",
    "TrackingContractError",
    "TrackingError",
    "TransitionError",
    "allocate_hash_track_id",
    "assert_track_contracts_registered",
    "assert_transition_allowed",
    "assert_v1_track_fingerprints_unchanged",
    "build_synthetic_receipt",
    "build_synthetic_request",
    "compile_tracking_schemas",
    "default_tracking_policy_path",
    "evaluate_ball_tracking",
    "evaluate_human_tracking",
    "evaluate_tracking",
    "gap_us",
    "load_all_tracking_contracts",
    "load_tracking_contract",
    "load_tracking_json_schema",
    "load_tracking_policy",
    "observation_state_for_source",
    "policy_fingerprint",
    "recount_receipt_from_tables",
    "require_monotonic_times",
    "resolve_video_time_us",
    "tracking_schema_fingerprints",
    "validate_against_json_schema",
    "validate_lifecycle_sequence",
    "validate_receipt_payload",
    "validate_request_payload",
    "validate_track_bbox",
    "validate_track_bundle",
]
