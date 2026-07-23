"""Public Stage 5A/5B/5C perception detection API (adapters are lazy)."""

# Stage 5B/5C pure modules (no Ultralytics side effects).
from football_analytics.perception.ball_detector_config import (  # noqa: E402
    ball_detector_config_fingerprint,
    default_ball_detector_config_path,
    load_ball_detector_config,
)
from football_analytics.perception.ball_evaluation import (  # noqa: E402
    evaluate_ball_from_rows,
)
from football_analytics.perception.contracts import (
    CONTRACT_NAMES,
    DETECTIONS_CONTRACT,
    DETECTIONS_VERSION,
    JSON_SCHEMA_NAMES,
    assert_detection_contracts_registered,
    compile_detection_schemas,
    detection_schema_fingerprints,
    load_all_detection_contracts,
    load_detection_contract,
    load_perception_json_schema,
    validate_against_json_schema,
)
from football_analytics.perception.detection_evaluation import (  # noqa: E402
    evaluate_from_rows as evaluate_human_detections_from_rows,
)
from football_analytics.perception.human_detector_config import (  # noqa: E402
    default_human_detector_config_path,
    human_detector_config_fingerprint,
    load_human_detector_config,
)
from football_analytics.perception.policy import (
    load_detection_policy,
    policy_fingerprint,
    resolve_frame_routing,
)
from football_analytics.perception.taxonomy import (
    ClassMappingResult,
    load_detection_taxonomy,
    map_model_class,
    taxonomy_fingerprint,
)
from football_analytics.perception.transforms import (
    build_preprocessing_transform,
    clip_bbox_xyxy,
    forward_bbox,
    inverse_bbox,
    roundtrip_bbox,
    validate_bbox_xyxy,
)
from football_analytics.perception.types import (
    CONTRACT_VERSION,
    ERROR_CODES,
    PROCESSED_STATUSES,
    UNPROCESSED_STATUSES,
    ChannelOrder,
    ColorSpace,
    DetectionAttributes,
    DetectionFrameStatus,
    DetectionRunReceipt,
    Eligibility,
    EntityType,
    Orientation,
    PerceptionContractError,
    PerceptionError,
    PreprocessingTransform,
    ProcessingStatus,
    ReceiptStatus,
    ResizeMode,
    ReviewStatus,
    RoleLabel,
    RoleSource,
)
from football_analytics.perception.validation import validate_detection_bundle

__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_NAMES",
    "DETECTIONS_CONTRACT",
    "DETECTIONS_VERSION",
    "JSON_SCHEMA_NAMES",
    "ERROR_CODES",
    "PROCESSED_STATUSES",
    "UNPROCESSED_STATUSES",
    "PerceptionError",
    "PerceptionContractError",
    "EntityType",
    "RoleLabel",
    "RoleSource",
    "ProcessingStatus",
    "Eligibility",
    "ReviewStatus",
    "ResizeMode",
    "ReceiptStatus",
    "ColorSpace",
    "ChannelOrder",
    "Orientation",
    "DetectionFrameStatus",
    "DetectionAttributes",
    "PreprocessingTransform",
    "DetectionRunReceipt",
    "ClassMappingResult",
    "load_detection_contract",
    "load_all_detection_contracts",
    "detection_schema_fingerprints",
    "compile_detection_schemas",
    "assert_detection_contracts_registered",
    "load_perception_json_schema",
    "validate_against_json_schema",
    "load_detection_taxonomy",
    "taxonomy_fingerprint",
    "map_model_class",
    "load_detection_policy",
    "policy_fingerprint",
    "resolve_frame_routing",
    "validate_bbox_xyxy",
    "clip_bbox_xyxy",
    "build_preprocessing_transform",
    "forward_bbox",
    "inverse_bbox",
    "roundtrip_bbox",
    "validate_detection_bundle",
    "load_human_detector_config",
    "human_detector_config_fingerprint",
    "default_human_detector_config_path",
    "evaluate_human_detections_from_rows",
    "load_ball_detector_config",
    "ball_detector_config_fingerprint",
    "default_ball_detector_config_path",
    "evaluate_ball_from_rows",
    "run_human_detection",
    "run_ball_detection",
]


def __getattr__(name: str):
    """Lazy export for service entrypoints (avoids pulling adapters at import)."""
    if name == "run_human_detection":
        from football_analytics.perception.detection_service import run_human_detection

        return run_human_detection
    if name == "run_ball_detection":
        from football_analytics.perception.ball_service import run_ball_detection

        return run_ball_detection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
