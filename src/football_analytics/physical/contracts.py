"""Schema loading helpers for Stage 9A physical / trajectory contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.physical.types import PhysicalContractError

TARGET_TRAJECTORY_SAMPLES_CONTRACT = "target_trajectory_samples"
TARGET_TRAJECTORY_SEGMENTS_CONTRACT = "target_trajectory_segments"
TRAJECTORY_GAPS_CONTRACT = "trajectory_gaps"
PHYSICAL_METRIC_RESULTS_CONTRACT = "physical_metric_results"

# Frozen upstream — must not change in Stage 9A.
PROJECTED_POSITIONS_CONTRACT = "projected_positions"
TRACK_IDENTITY_ASSIGNMENTS_CONTRACT = "track_identity_assignments"
CALIBRATIONS_CONTRACT = "calibrations"
CALIBRATION_SEGMENTS_CONTRACT = "calibration_segments"

EXPECTED_PROJECTED_POSITIONS_FP = "1860638e13c101a2c4b52ecb86e31c264f1b09ee8f255b3813fb9da8325055ba"
EXPECTED_TRACK_IDENTITY_ASSIGNMENTS_FP = (
    "235e7888e13b3e1435eddc8b2c7aa0fcdcbbd2505b3e3c8b00bd51fdf09e74c0"
)
EXPECTED_CALIBRATIONS_FP = "41360b19ae034f361949a75d8e773c265f0792b2603b69f612ab5863662ac871"
EXPECTED_CALIBRATION_SEGMENTS_FP = (
    "9ce13ae0d771d92a66b72ec5818943f29e96ff4ac71ebf5603fb7fc9c8ab5037"
)

PHYSICAL_ARROW_CONTRACTS: tuple[str, ...] = (
    TARGET_TRAJECTORY_SAMPLES_CONTRACT,
    TARGET_TRAJECTORY_SEGMENTS_CONTRACT,
    TRAJECTORY_GAPS_CONTRACT,
    PHYSICAL_METRIC_RESULTS_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "physical_metric_request",
    "physical_metric_run_receipt",
    "physical_metric_evaluation",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 30


def load_physical_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(PHYSICAL_ARROW_CONTRACTS) | {
        PROJECTED_POSITIONS_CONTRACT,
        TRACK_IDENTITY_ASSIGNMENTS_CONTRACT,
        CALIBRATIONS_CONTRACT,
        CALIBRATION_SEGMENTS_CONTRACT,
        "frames",
        "videos",
        "track_observations",
        "analysis_windows",
    }
    if name not in allowed:
        raise PhysicalContractError(f"unknown physical-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_physical_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_physical_contract(name, 1, registry=registry)
        for name in PHYSICAL_ARROW_CONTRACTS
    }


def physical_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_physical_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    for name in (
        PROJECTED_POSITIONS_CONTRACT,
        TRACK_IDENTITY_ASSIGNMENTS_CONTRACT,
        CALIBRATIONS_CONTRACT,
        CALIBRATION_SEGMENTS_CONTRACT,
    ):
        out[name] = contract_fingerprint(load_physical_contract(name, 1, registry=registry))
    return out


def compile_physical_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_physical_contracts(registry=registry).items()
    }


def assert_physical_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in PHYSICAL_ARROW_CONTRACTS if n not in names]
    if missing:
        raise PhysicalContractError(f"physical contracts missing from registry: {missing}")


def assert_frozen_upstream_fingerprints(*, registry: Any = None) -> None:
    fps = physical_schema_fingerprints(registry=registry)
    checks = {
        PROJECTED_POSITIONS_CONTRACT: EXPECTED_PROJECTED_POSITIONS_FP,
        TRACK_IDENTITY_ASSIGNMENTS_CONTRACT: EXPECTED_TRACK_IDENTITY_ASSIGNMENTS_FP,
        CALIBRATIONS_CONTRACT: EXPECTED_CALIBRATIONS_FP,
        CALIBRATION_SEGMENTS_CONTRACT: EXPECTED_CALIBRATION_SEGMENTS_FP,
    }
    for name, expected in checks.items():
        if fps[name] != expected:
            raise PhysicalContractError(f"{name} v1 fingerprint changed")


def physical_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "physical"


def load_physical_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise PhysicalContractError(f"unknown physical json schema: {name}")
    path = physical_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise PhysicalContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PhysicalContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "TARGET_TRAJECTORY_SAMPLES_CONTRACT",
    "TARGET_TRAJECTORY_SEGMENTS_CONTRACT",
    "TRAJECTORY_GAPS_CONTRACT",
    "PHYSICAL_METRIC_RESULTS_CONTRACT",
    "PROJECTED_POSITIONS_CONTRACT",
    "TRACK_IDENTITY_ASSIGNMENTS_CONTRACT",
    "CALIBRATIONS_CONTRACT",
    "CALIBRATION_SEGMENTS_CONTRACT",
    "EXPECTED_PROJECTED_POSITIONS_FP",
    "EXPECTED_TRACK_IDENTITY_ASSIGNMENTS_FP",
    "EXPECTED_CALIBRATIONS_FP",
    "EXPECTED_CALIBRATION_SEGMENTS_FP",
    "PHYSICAL_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_physical_contract",
    "load_all_physical_contracts",
    "physical_schema_fingerprints",
    "compile_physical_schemas",
    "assert_physical_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "physical_schema_dir",
    "load_physical_json_schema",
    "validate_against_json_schema",
]
