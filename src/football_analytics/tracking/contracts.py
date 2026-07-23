"""Schema loading helpers for Stage 6A tracking contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.tracking.types import TrackingContractError

# Existing v1 — must not change.
TRACK_OBSERVATIONS_CONTRACT = "track_observations"
TRACK_SUMMARIES_CONTRACT = "track_summaries"
DETECTIONS_CONTRACT = "detections"
TRACK_LIFECYCLE_CONTRACT = "track_lifecycle"

EXPECTED_TRACK_OBSERVATIONS_FP = "9ca2f7af56e69b47ec8db8d644164c84aa7fe3a62da40e247ed6db4f2c4c5f01"
EXPECTED_TRACK_SUMMARIES_FP = "7b04e31d641c49e66ad06baec53e1075e2bc286b9f08f1497aa0571bf7c1c168"
EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"

TRACK_CONTRACT_NAMES: tuple[str, ...] = (
    TRACK_OBSERVATIONS_CONTRACT,
    TRACK_SUMMARIES_CONTRACT,
    TRACK_LIFECYCLE_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "tracking_request",
    "tracking_run_receipt",
    "tracking_evaluation",
    "tracking_pipeline_receipt",
    "tracking_quality_report",
    "tracking_bundle_manifest",
)


def load_tracking_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(TRACK_CONTRACT_NAMES) | {DETECTIONS_CONTRACT, "frames", "videos"}
    if name not in allowed:
        raise TrackingContractError(f"unknown tracking contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_tracking_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_tracking_contract(name, 1, registry=registry) for name in TRACK_CONTRACT_NAMES
    }


def tracking_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_tracking_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    out[DETECTIONS_CONTRACT] = contract_fingerprint(
        load_tracking_contract(DETECTIONS_CONTRACT, 1, registry=registry)
    )
    return out


def compile_tracking_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_tracking_contracts(registry=registry).items()
    }


def assert_track_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in TRACK_CONTRACT_NAMES if n not in names]
    if missing:
        raise TrackingContractError(f"tracking contracts missing from registry: {missing}")


def assert_v1_track_fingerprints_unchanged(*, registry: Any = None) -> None:
    fps = tracking_schema_fingerprints(registry=registry)
    if fps[TRACK_OBSERVATIONS_CONTRACT] != EXPECTED_TRACK_OBSERVATIONS_FP:
        raise TrackingContractError("track_observations v1 fingerprint changed")
    if fps[TRACK_SUMMARIES_CONTRACT] != EXPECTED_TRACK_SUMMARIES_FP:
        raise TrackingContractError("track_summaries v1 fingerprint changed")
    if fps[DETECTIONS_CONTRACT] != EXPECTED_DETECTIONS_FP:
        raise TrackingContractError("detections v1 fingerprint changed")


def tracking_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "tracking"


def load_tracking_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise TrackingContractError(f"unknown tracking json schema: {name}")
    path = tracking_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise TrackingContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TrackingContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "TRACK_OBSERVATIONS_CONTRACT",
    "TRACK_SUMMARIES_CONTRACT",
    "DETECTIONS_CONTRACT",
    "TRACK_LIFECYCLE_CONTRACT",
    "EXPECTED_TRACK_OBSERVATIONS_FP",
    "EXPECTED_TRACK_SUMMARIES_FP",
    "EXPECTED_DETECTIONS_FP",
    "TRACK_CONTRACT_NAMES",
    "JSON_SCHEMA_NAMES",
    "load_tracking_contract",
    "load_all_tracking_contracts",
    "tracking_schema_fingerprints",
    "compile_tracking_schemas",
    "assert_track_contracts_registered",
    "assert_v1_track_fingerprints_unchanged",
    "tracking_schema_dir",
    "load_tracking_json_schema",
    "validate_against_json_schema",
]
