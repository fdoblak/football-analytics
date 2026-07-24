"""Schema loading helpers for Stage 12A duels / competitive-events contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.duels.types import DuelsContractError

TAKE_ON_ATTEMPTS_CONTRACT = "take_on_attempts"
GROUND_DUEL_CANDIDATES_CONTRACT = "ground_duel_candidates"
AERIAL_DUEL_CANDIDATES_CONTRACT = "aerial_duel_candidates"
TACKLE_EVENTS_CONTRACT = "tackle_events"
RECOVERY_EVENTS_CONTRACT = "recovery_events"
TURNOVER_EVENTS_CONTRACT = "turnover_events"
CLEARANCE_EVENTS_CONTRACT = "clearance_events"

# Frozen upstream — must not change in Stage 12A.
PASS_CANDIDATES_CONTRACT = "pass_candidates"
RECEPTION_CANDIDATES_CONTRACT = "reception_candidates"
PASS_OUTCOMES_CONTRACT = "pass_outcomes"
POSSESSION_HYPOTHESES_CONTRACT = "possession_hypotheses"
BALL_CONTACT_CANDIDATES_CONTRACT = "ball_contact_candidates"

EXPECTED_PASS_CANDIDATES_FP = "923f16283c45eca74d32bae4570d00935952628640a295334ad09600e8d55053"
EXPECTED_RECEPTION_CANDIDATES_FP = (
    "ceb755b8cc2de05e87f48bbc645e70f2fed46b772b33ed628bb3983eef03336f"
)
EXPECTED_PASS_OUTCOMES_FP = "bc65b9e3079f388854aa2b962c04b03900ec030313e89844f54012ab789584b8"
EXPECTED_POSSESSION_HYPOTHESES_FP = (
    "ab6f816a93b188841d42fe45531463ae7dd97b7842dbbdd599ee34d2a8e6f927"
)
EXPECTED_BALL_CONTACT_CANDIDATES_FP = (
    "962e1566de124f1e34f72df09c91b2768cafaebffcfc7669bb73ccef153be058"
)

DUELS_ARROW_CONTRACTS: tuple[str, ...] = (
    TAKE_ON_ATTEMPTS_CONTRACT,
    GROUND_DUEL_CANDIDATES_CONTRACT,
    AERIAL_DUEL_CANDIDATES_CONTRACT,
    TACKLE_EVENTS_CONTRACT,
    RECOVERY_EVENTS_CONTRACT,
    TURNOVER_EVENTS_CONTRACT,
    CLEARANCE_EVENTS_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "duels_request",
    "duels_run_receipt",
    "duels_evaluation",
    "duels_quality",
    "manual_review_queue",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 42


def load_duels_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(DUELS_ARROW_CONTRACTS) | {
        PASS_CANDIDATES_CONTRACT,
        RECEPTION_CANDIDATES_CONTRACT,
        PASS_OUTCOMES_CONTRACT,
        POSSESSION_HYPOTHESES_CONTRACT,
        BALL_CONTACT_CANDIDATES_CONTRACT,
        "frames",
        "videos",
        "analysis_windows",
        "team_assignments",
    }
    if name not in allowed:
        raise DuelsContractError(f"unknown duels-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_duels_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {name: load_duels_contract(name, 1, registry=registry) for name in DUELS_ARROW_CONTRACTS}


def duels_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_duels_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    for name in (
        PASS_CANDIDATES_CONTRACT,
        RECEPTION_CANDIDATES_CONTRACT,
        PASS_OUTCOMES_CONTRACT,
        POSSESSION_HYPOTHESES_CONTRACT,
        BALL_CONTACT_CANDIDATES_CONTRACT,
    ):
        out[name] = contract_fingerprint(load_duels_contract(name, 1, registry=registry))
    return out


def compile_duels_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_duels_contracts(registry=registry).items()
    }


def assert_duels_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in DUELS_ARROW_CONTRACTS if n not in names]
    if missing:
        raise DuelsContractError(f"duels contracts missing from registry: {missing}")


def assert_frozen_upstream_fingerprints(*, registry: Any = None) -> None:
    fps = duels_schema_fingerprints(registry=registry)
    checks = {
        PASS_CANDIDATES_CONTRACT: EXPECTED_PASS_CANDIDATES_FP,
        RECEPTION_CANDIDATES_CONTRACT: EXPECTED_RECEPTION_CANDIDATES_FP,
        PASS_OUTCOMES_CONTRACT: EXPECTED_PASS_OUTCOMES_FP,
        POSSESSION_HYPOTHESES_CONTRACT: EXPECTED_POSSESSION_HYPOTHESES_FP,
        BALL_CONTACT_CANDIDATES_CONTRACT: EXPECTED_BALL_CONTACT_CANDIDATES_FP,
    }
    for name, expected in checks.items():
        if fps[name] != expected:
            raise DuelsContractError(f"{name} v1 fingerprint changed")


def duels_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "duels"


def load_duels_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise DuelsContractError(f"unknown duels json schema: {name}")
    path = duels_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise DuelsContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DuelsContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "TAKE_ON_ATTEMPTS_CONTRACT",
    "GROUND_DUEL_CANDIDATES_CONTRACT",
    "AERIAL_DUEL_CANDIDATES_CONTRACT",
    "TACKLE_EVENTS_CONTRACT",
    "RECOVERY_EVENTS_CONTRACT",
    "TURNOVER_EVENTS_CONTRACT",
    "CLEARANCE_EVENTS_CONTRACT",
    "PASS_CANDIDATES_CONTRACT",
    "RECEPTION_CANDIDATES_CONTRACT",
    "PASS_OUTCOMES_CONTRACT",
    "POSSESSION_HYPOTHESES_CONTRACT",
    "BALL_CONTACT_CANDIDATES_CONTRACT",
    "EXPECTED_PASS_CANDIDATES_FP",
    "EXPECTED_RECEPTION_CANDIDATES_FP",
    "EXPECTED_PASS_OUTCOMES_FP",
    "EXPECTED_POSSESSION_HYPOTHESES_FP",
    "EXPECTED_BALL_CONTACT_CANDIDATES_FP",
    "DUELS_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_duels_contract",
    "load_all_duels_contracts",
    "duels_schema_fingerprints",
    "compile_duels_schemas",
    "assert_duels_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "duels_schema_dir",
    "load_duels_json_schema",
    "validate_against_json_schema",
]
