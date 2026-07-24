"""Schema loading helpers for Stage 10A human-ball interaction contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.interaction.types import InteractionContractError

HUMAN_BALL_PROXIMITY_CONTRACT = "human_ball_proximity"
BALL_CONTACT_CANDIDATES_CONTRACT = "ball_contact_candidates"
POSSESSION_HYPOTHESES_CONTRACT = "possession_hypotheses"

# Frozen upstream — must not change in Stage 10A.
DETECTIONS_CONTRACT = "detections"
TRACK_OBSERVATIONS_CONTRACT = "track_observations"
TRACK_SUMMARIES_CONTRACT = "track_summaries"
TRACK_LIFECYCLE_CONTRACT = "track_lifecycle"
TRACK_IDENTITY_ASSIGNMENTS_CONTRACT = "track_identity_assignments"
PROJECTED_POSITIONS_CONTRACT = "projected_positions"
CALIBRATIONS_CONTRACT = "calibrations"
CALIBRATION_SEGMENTS_CONTRACT = "calibration_segments"
PHYSICAL_METRIC_RESULTS_CONTRACT = "physical_metric_results"

EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"
EXPECTED_TRACK_OBSERVATIONS_FP = "9ca2f7af56e69b47ec8db8d644164c84aa7fe3a62da40e247ed6db4f2c4c5f01"
EXPECTED_TRACK_SUMMARIES_FP = "7b04e31d641c49e66ad06baec53e1075e2bc286b9f08f1497aa0571bf7c1c168"
EXPECTED_TRACK_LIFECYCLE_FP = "613cd81e4359e780c19ba6d709999963a9f528bed25245ea2236ca84027fa40d"
EXPECTED_TRACK_IDENTITY_ASSIGNMENTS_FP = (
    "235e7888e13b3e1435eddc8b2c7aa0fcdcbbd2505b3e3c8b00bd51fdf09e74c0"
)
EXPECTED_PROJECTED_POSITIONS_FP = "1860638e13c101a2c4b52ecb86e31c264f1b09ee8f255b3813fb9da8325055ba"
EXPECTED_CALIBRATIONS_FP = "41360b19ae034f361949a75d8e773c265f0792b2603b69f612ab5863662ac871"
EXPECTED_CALIBRATION_SEGMENTS_FP = (
    "9ce13ae0d771d92a66b72ec5818943f29e96ff4ac71ebf5603fb7fc9c8ab5037"
)
EXPECTED_PHYSICAL_METRIC_RESULTS_FP = (
    "aa705bb819e10e49acb71be4d54b6bf65345b9795611b909676d2f5b596dc55b"
)

INTERACTION_ARROW_CONTRACTS: tuple[str, ...] = (
    HUMAN_BALL_PROXIMITY_CONTRACT,
    BALL_CONTACT_CANDIDATES_CONTRACT,
    POSSESSION_HYPOTHESES_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "human_ball_interaction_request",
    "human_ball_interaction_run_receipt",
    "human_ball_interaction_evaluation",
    "human_ball_interaction_quality",
    "human_ball_interaction_manual_review_queue",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 35


def load_interaction_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(INTERACTION_ARROW_CONTRACTS) | {
        DETECTIONS_CONTRACT,
        TRACK_OBSERVATIONS_CONTRACT,
        TRACK_SUMMARIES_CONTRACT,
        TRACK_LIFECYCLE_CONTRACT,
        TRACK_IDENTITY_ASSIGNMENTS_CONTRACT,
        PROJECTED_POSITIONS_CONTRACT,
        CALIBRATIONS_CONTRACT,
        CALIBRATION_SEGMENTS_CONTRACT,
        PHYSICAL_METRIC_RESULTS_CONTRACT,
        "frames",
        "videos",
        "analysis_windows",
        "team_assignments",
    }
    if name not in allowed:
        raise InteractionContractError(f"unknown interaction-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_interaction_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_interaction_contract(name, 1, registry=registry)
        for name in INTERACTION_ARROW_CONTRACTS
    }


def interaction_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_interaction_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    for name in (
        DETECTIONS_CONTRACT,
        TRACK_OBSERVATIONS_CONTRACT,
        TRACK_SUMMARIES_CONTRACT,
        TRACK_LIFECYCLE_CONTRACT,
        TRACK_IDENTITY_ASSIGNMENTS_CONTRACT,
        PROJECTED_POSITIONS_CONTRACT,
        CALIBRATIONS_CONTRACT,
        CALIBRATION_SEGMENTS_CONTRACT,
        PHYSICAL_METRIC_RESULTS_CONTRACT,
    ):
        out[name] = contract_fingerprint(load_interaction_contract(name, 1, registry=registry))
    return out


def compile_interaction_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_interaction_contracts(registry=registry).items()
    }


def assert_interaction_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in INTERACTION_ARROW_CONTRACTS if n not in names]
    if missing:
        raise InteractionContractError(f"interaction contracts missing from registry: {missing}")


def assert_frozen_upstream_fingerprints(*, registry: Any = None) -> None:
    fps = interaction_schema_fingerprints(registry=registry)
    checks = {
        DETECTIONS_CONTRACT: EXPECTED_DETECTIONS_FP,
        TRACK_OBSERVATIONS_CONTRACT: EXPECTED_TRACK_OBSERVATIONS_FP,
        TRACK_SUMMARIES_CONTRACT: EXPECTED_TRACK_SUMMARIES_FP,
        TRACK_LIFECYCLE_CONTRACT: EXPECTED_TRACK_LIFECYCLE_FP,
        TRACK_IDENTITY_ASSIGNMENTS_CONTRACT: EXPECTED_TRACK_IDENTITY_ASSIGNMENTS_FP,
        PROJECTED_POSITIONS_CONTRACT: EXPECTED_PROJECTED_POSITIONS_FP,
        CALIBRATIONS_CONTRACT: EXPECTED_CALIBRATIONS_FP,
        CALIBRATION_SEGMENTS_CONTRACT: EXPECTED_CALIBRATION_SEGMENTS_FP,
        PHYSICAL_METRIC_RESULTS_CONTRACT: EXPECTED_PHYSICAL_METRIC_RESULTS_FP,
    }
    for name, expected in checks.items():
        if fps[name] != expected:
            raise InteractionContractError(f"{name} v1 fingerprint changed")


def interaction_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "interaction"


def load_interaction_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise InteractionContractError(f"unknown interaction json schema: {name}")
    path = interaction_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise InteractionContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise InteractionContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "HUMAN_BALL_PROXIMITY_CONTRACT",
    "BALL_CONTACT_CANDIDATES_CONTRACT",
    "POSSESSION_HYPOTHESES_CONTRACT",
    "DETECTIONS_CONTRACT",
    "TRACK_OBSERVATIONS_CONTRACT",
    "TRACK_SUMMARIES_CONTRACT",
    "TRACK_LIFECYCLE_CONTRACT",
    "TRACK_IDENTITY_ASSIGNMENTS_CONTRACT",
    "PROJECTED_POSITIONS_CONTRACT",
    "CALIBRATIONS_CONTRACT",
    "CALIBRATION_SEGMENTS_CONTRACT",
    "PHYSICAL_METRIC_RESULTS_CONTRACT",
    "EXPECTED_DETECTIONS_FP",
    "EXPECTED_TRACK_OBSERVATIONS_FP",
    "EXPECTED_TRACK_SUMMARIES_FP",
    "EXPECTED_TRACK_LIFECYCLE_FP",
    "EXPECTED_TRACK_IDENTITY_ASSIGNMENTS_FP",
    "EXPECTED_PROJECTED_POSITIONS_FP",
    "EXPECTED_CALIBRATIONS_FP",
    "EXPECTED_CALIBRATION_SEGMENTS_FP",
    "EXPECTED_PHYSICAL_METRIC_RESULTS_FP",
    "INTERACTION_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_interaction_contract",
    "load_all_interaction_contracts",
    "interaction_schema_fingerprints",
    "compile_interaction_schemas",
    "assert_interaction_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "interaction_schema_dir",
    "load_interaction_json_schema",
    "validate_against_json_schema",
]
