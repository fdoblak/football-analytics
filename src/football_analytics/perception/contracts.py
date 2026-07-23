"""Schema loading helpers for Stage 5A detection contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec

CONTRACT_NAMES: tuple[str, ...] = (
    "detection_frame_status",
    "detection_attributes",
)

# detections v1 remains canonical and unchanged; referenced but not owned here.
DETECTIONS_CONTRACT = "detections"
DETECTIONS_VERSION = 1

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "detection_run_receipt",
    "preprocessing_transform",
    "detection_pipeline_receipt",
    "detection_quality_report",
)


def load_detection_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    if name not in CONTRACT_NAMES and name != DETECTIONS_CONTRACT:
        raise ValueError(f"unknown detection contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_detection_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    out = {name: load_detection_contract(name, 1, registry=registry) for name in CONTRACT_NAMES}
    out[DETECTIONS_CONTRACT] = load_detection_contract(
        DETECTIONS_CONTRACT, DETECTIONS_VERSION, registry=registry
    )
    return out


def detection_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_detection_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    return out


def compile_detection_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_detection_contracts(registry=registry).items()
    }


def assert_detection_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in CONTRACT_NAMES if n not in names]
    if missing:
        raise ValueError(f"detection contracts missing from registry: {missing}")
    if DETECTIONS_CONTRACT not in names:
        raise ValueError("detections contract missing from registry")


def perception_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "perception"


def load_perception_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise ValueError(f"unknown perception json schema: {name}")
    path = perception_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "CONTRACT_NAMES",
    "DETECTIONS_CONTRACT",
    "DETECTIONS_VERSION",
    "JSON_SCHEMA_NAMES",
    "load_detection_contract",
    "load_all_detection_contracts",
    "detection_schema_fingerprints",
    "compile_detection_schemas",
    "assert_detection_contracts_registered",
    "perception_schema_dir",
    "load_perception_json_schema",
    "validate_against_json_schema",
]
