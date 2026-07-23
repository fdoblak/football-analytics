"""Schema loading helpers for Stage 7A identity contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.identity.types import IdentityContractError

IDENTITY_EVIDENCE_CONTRACT = "identity_evidence"
REID_CANDIDATE_LINKS_CONTRACT = "reid_candidate_links"
TRACK_IDENTITY_ASSIGNMENTS_CONTRACT = "track_identity_assignments"

# Frozen upstream contracts — must not change.
TEAM_ASSIGNMENTS_CONTRACT = "team_assignments"
JERSEY_OBSERVATIONS_CONTRACT = "jersey_observations"
DETECTIONS_CONTRACT = "detections"
TRACK_OBSERVATIONS_CONTRACT = "track_observations"
TRACK_SUMMARIES_CONTRACT = "track_summaries"
TRACK_LIFECYCLE_CONTRACT = "track_lifecycle"

EXPECTED_TEAM_ASSIGNMENTS_FP = "759aa9b77de6faa8c00d32ccc71846b164bd5cb66ca4155051b0587be07458d0"
EXPECTED_JERSEY_OBSERVATIONS_FP = "aabc7642d9c7a31b99393bd2704470d2095c61d84ca1281d1a5e63cd3dfd9365"
EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"
EXPECTED_TRACK_OBSERVATIONS_FP = "9ca2f7af56e69b47ec8db8d644164c84aa7fe3a62da40e247ed6db4f2c4c5f01"
EXPECTED_TRACK_SUMMARIES_FP = "7b04e31d641c49e66ad06baec53e1075e2bc286b9f08f1497aa0571bf7c1c168"
EXPECTED_TRACK_LIFECYCLE_FP = "613cd81e4359e780c19ba6d709999963a9f528bed25245ea2236ca84027fa40d"

IDENTITY_ARROW_CONTRACTS: tuple[str, ...] = (
    IDENTITY_EVIDENCE_CONTRACT,
    REID_CANDIDATE_LINKS_CONTRACT,
    TRACK_IDENTITY_ASSIGNMENTS_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "target_player_request",
    "identity_manual_audit",
    "identity_run_receipt",
    "identity_evaluation",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 19


def load_identity_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(IDENTITY_ARROW_CONTRACTS) | {
        TEAM_ASSIGNMENTS_CONTRACT,
        JERSEY_OBSERVATIONS_CONTRACT,
        DETECTIONS_CONTRACT,
        TRACK_OBSERVATIONS_CONTRACT,
        TRACK_SUMMARIES_CONTRACT,
        TRACK_LIFECYCLE_CONTRACT,
        "frames",
        "videos",
    }
    if name not in allowed:
        raise IdentityContractError(f"unknown identity-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_identity_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_identity_contract(name, 1, registry=registry)
        for name in IDENTITY_ARROW_CONTRACTS
    }


def identity_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_identity_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    for name in (
        TEAM_ASSIGNMENTS_CONTRACT,
        JERSEY_OBSERVATIONS_CONTRACT,
        DETECTIONS_CONTRACT,
        TRACK_OBSERVATIONS_CONTRACT,
        TRACK_SUMMARIES_CONTRACT,
        TRACK_LIFECYCLE_CONTRACT,
    ):
        out[name] = contract_fingerprint(load_identity_contract(name, 1, registry=registry))
    return out


def compile_identity_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_identity_contracts(registry=registry).items()
    }


def assert_identity_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in IDENTITY_ARROW_CONTRACTS if n not in names]
    if missing:
        raise IdentityContractError(f"identity contracts missing from registry: {missing}")


def assert_frozen_upstream_fingerprints(*, registry: Any = None) -> None:
    fps = identity_schema_fingerprints(registry=registry)
    checks = {
        TEAM_ASSIGNMENTS_CONTRACT: EXPECTED_TEAM_ASSIGNMENTS_FP,
        JERSEY_OBSERVATIONS_CONTRACT: EXPECTED_JERSEY_OBSERVATIONS_FP,
        DETECTIONS_CONTRACT: EXPECTED_DETECTIONS_FP,
        TRACK_OBSERVATIONS_CONTRACT: EXPECTED_TRACK_OBSERVATIONS_FP,
        TRACK_SUMMARIES_CONTRACT: EXPECTED_TRACK_SUMMARIES_FP,
        TRACK_LIFECYCLE_CONTRACT: EXPECTED_TRACK_LIFECYCLE_FP,
    }
    for name, expected in checks.items():
        if fps[name] != expected:
            raise IdentityContractError(f"{name} v1 fingerprint changed")


def identity_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "identity"


def load_identity_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise IdentityContractError(f"unknown identity json schema: {name}")
    path = identity_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise IdentityContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise IdentityContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "IDENTITY_EVIDENCE_CONTRACT",
    "REID_CANDIDATE_LINKS_CONTRACT",
    "TRACK_IDENTITY_ASSIGNMENTS_CONTRACT",
    "TEAM_ASSIGNMENTS_CONTRACT",
    "JERSEY_OBSERVATIONS_CONTRACT",
    "DETECTIONS_CONTRACT",
    "TRACK_OBSERVATIONS_CONTRACT",
    "TRACK_SUMMARIES_CONTRACT",
    "TRACK_LIFECYCLE_CONTRACT",
    "EXPECTED_TEAM_ASSIGNMENTS_FP",
    "EXPECTED_JERSEY_OBSERVATIONS_FP",
    "EXPECTED_DETECTIONS_FP",
    "EXPECTED_TRACK_OBSERVATIONS_FP",
    "EXPECTED_TRACK_SUMMARIES_FP",
    "EXPECTED_TRACK_LIFECYCLE_FP",
    "IDENTITY_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_identity_contract",
    "load_all_identity_contracts",
    "identity_schema_fingerprints",
    "compile_identity_schemas",
    "assert_identity_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "identity_schema_dir",
    "load_identity_json_schema",
    "validate_against_json_schema",
]
