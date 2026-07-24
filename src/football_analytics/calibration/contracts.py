"""Schema loading helpers for Stage 8A calibration contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.calibration.types import CalibrationContractError
from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec

CALIBRATIONS_CONTRACT = "calibrations"
CALIBRATION_FEATURES_CONTRACT = "calibration_features"
CALIBRATION_SEGMENTS_CONTRACT = "calibration_segments"
PROJECTED_POSITIONS_CONTRACT = "projected_positions"

# Frozen — must not change in Stage 8A.
EXPECTED_CALIBRATIONS_FP = "41360b19ae034f361949a75d8e773c265f0792b2603b69f612ab5863662ac871"

CALIBRATION_ARROW_CONTRACTS: tuple[str, ...] = (
    CALIBRATION_FEATURES_CONTRACT,
    CALIBRATION_SEGMENTS_CONTRACT,
    PROJECTED_POSITIONS_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "calibration_request",
    "calibration_run_receipt",
    "calibration_evaluation",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 27


def load_calibration_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(CALIBRATION_ARROW_CONTRACTS) | {
        CALIBRATIONS_CONTRACT,
        "frames",
        "videos",
        "track_observations",
        "analysis_windows",
    }
    if name not in allowed:
        raise CalibrationContractError(f"unknown calibration-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_calibration_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_calibration_contract(name, 1, registry=registry)
        for name in CALIBRATION_ARROW_CONTRACTS
    }


def calibration_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_calibration_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    out[CALIBRATIONS_CONTRACT] = contract_fingerprint(
        load_calibration_contract(CALIBRATIONS_CONTRACT, 1, registry=registry)
    )
    return out


def compile_calibration_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_calibration_contracts(registry=registry).items()
    }


def assert_calibration_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in CALIBRATION_ARROW_CONTRACTS if n not in names]
    if missing:
        raise CalibrationContractError(f"calibration contracts missing from registry: {missing}")
    if CALIBRATIONS_CONTRACT not in names:
        raise CalibrationContractError("calibrations missing from registry")


def assert_calibrations_fingerprint_frozen(*, registry: Any = None) -> None:
    fp = contract_fingerprint(
        load_calibration_contract(CALIBRATIONS_CONTRACT, 1, registry=registry)
    )
    if fp != EXPECTED_CALIBRATIONS_FP:
        raise CalibrationContractError("calibrations v1 fingerprint changed")


def calibration_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "calibration"


def load_calibration_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise CalibrationContractError(f"unknown calibration json schema: {name}")
    path = calibration_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise CalibrationContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CalibrationContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "CALIBRATIONS_CONTRACT",
    "CALIBRATION_FEATURES_CONTRACT",
    "CALIBRATION_SEGMENTS_CONTRACT",
    "PROJECTED_POSITIONS_CONTRACT",
    "EXPECTED_CALIBRATIONS_FP",
    "CALIBRATION_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_calibration_contract",
    "load_all_calibration_contracts",
    "calibration_schema_fingerprints",
    "compile_calibration_schemas",
    "assert_calibration_contracts_registered",
    "assert_calibrations_fingerprint_frozen",
    "calibration_schema_dir",
    "load_calibration_json_schema",
    "validate_against_json_schema",
]
