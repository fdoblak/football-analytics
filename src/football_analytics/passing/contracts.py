"""Schema loading helpers for Stage 11A passing / reception / progression contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.passing.types import PassingContractError

PASS_CANDIDATES_CONTRACT = "pass_candidates"
RECEPTION_CANDIDATES_CONTRACT = "reception_candidates"
PASS_OUTCOMES_CONTRACT = "pass_outcomes"
BALL_PROGRESSION_SEGMENTS_CONTRACT = "ball_progression_segments"
TARGET_BALL_TOUCHES_CONTRACT = "target_ball_touches"

# Frozen upstream — must not change in Stage 11A.
HUMAN_BALL_PROXIMITY_CONTRACT = "human_ball_proximity"
BALL_CONTACT_CANDIDATES_CONTRACT = "ball_contact_candidates"
POSSESSION_HYPOTHESES_CONTRACT = "possession_hypotheses"
PROJECTED_POSITIONS_CONTRACT = "projected_positions"
PHYSICAL_METRIC_RESULTS_CONTRACT = "physical_metric_results"

EXPECTED_HUMAN_BALL_PROXIMITY_FP = (
    "a8f6bf8a36402bbdb065263c24870c6ba4cc2ed37061fe7575bb472209852ebd"
)
EXPECTED_BALL_CONTACT_CANDIDATES_FP = (
    "962e1566de124f1e34f72df09c91b2768cafaebffcfc7669bb73ccef153be058"
)
EXPECTED_POSSESSION_HYPOTHESES_FP = (
    "ab6f816a93b188841d42fe45531463ae7dd97b7842dbbdd599ee34d2a8e6f927"
)
EXPECTED_PROJECTED_POSITIONS_FP = "1860638e13c101a2c4b52ecb86e31c264f1b09ee8f255b3813fb9da8325055ba"
EXPECTED_PHYSICAL_METRIC_RESULTS_FP = (
    "aa705bb819e10e49acb71be4d54b6bf65345b9795611b909676d2f5b596dc55b"
)

PASSING_ARROW_CONTRACTS: tuple[str, ...] = (
    PASS_CANDIDATES_CONTRACT,
    RECEPTION_CANDIDATES_CONTRACT,
    PASS_OUTCOMES_CONTRACT,
    BALL_PROGRESSION_SEGMENTS_CONTRACT,
    TARGET_BALL_TOUCHES_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "passing_request",
    "passing_run_receipt",
    "passing_evaluation",
    "passing_quality",
    "attack_direction_evidence",
    "manual_review_queue",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 42


def load_passing_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(PASSING_ARROW_CONTRACTS) | {
        HUMAN_BALL_PROXIMITY_CONTRACT,
        BALL_CONTACT_CANDIDATES_CONTRACT,
        POSSESSION_HYPOTHESES_CONTRACT,
        PROJECTED_POSITIONS_CONTRACT,
        PHYSICAL_METRIC_RESULTS_CONTRACT,
        "frames",
        "videos",
        "analysis_windows",
        "team_assignments",
    }
    if name not in allowed:
        raise PassingContractError(f"unknown passing-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_passing_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_passing_contract(name, 1, registry=registry) for name in PASSING_ARROW_CONTRACTS
    }


def passing_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_passing_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    for name in (
        HUMAN_BALL_PROXIMITY_CONTRACT,
        BALL_CONTACT_CANDIDATES_CONTRACT,
        POSSESSION_HYPOTHESES_CONTRACT,
        PROJECTED_POSITIONS_CONTRACT,
        PHYSICAL_METRIC_RESULTS_CONTRACT,
    ):
        out[name] = contract_fingerprint(load_passing_contract(name, 1, registry=registry))
    return out


def compile_passing_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_passing_contracts(registry=registry).items()
    }


def assert_passing_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in PASSING_ARROW_CONTRACTS if n not in names]
    if missing:
        raise PassingContractError(f"passing contracts missing from registry: {missing}")


def assert_frozen_upstream_fingerprints(*, registry: Any = None) -> None:
    fps = passing_schema_fingerprints(registry=registry)
    checks = {
        HUMAN_BALL_PROXIMITY_CONTRACT: EXPECTED_HUMAN_BALL_PROXIMITY_FP,
        BALL_CONTACT_CANDIDATES_CONTRACT: EXPECTED_BALL_CONTACT_CANDIDATES_FP,
        POSSESSION_HYPOTHESES_CONTRACT: EXPECTED_POSSESSION_HYPOTHESES_FP,
        PROJECTED_POSITIONS_CONTRACT: EXPECTED_PROJECTED_POSITIONS_FP,
        PHYSICAL_METRIC_RESULTS_CONTRACT: EXPECTED_PHYSICAL_METRIC_RESULTS_FP,
    }
    for name, expected in checks.items():
        if fps[name] != expected:
            raise PassingContractError(f"{name} v1 fingerprint changed")


def passing_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "passing"


def load_passing_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise PassingContractError(f"unknown passing json schema: {name}")
    path = passing_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise PassingContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PassingContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "PASS_CANDIDATES_CONTRACT",
    "RECEPTION_CANDIDATES_CONTRACT",
    "PASS_OUTCOMES_CONTRACT",
    "BALL_PROGRESSION_SEGMENTS_CONTRACT",
    "TARGET_BALL_TOUCHES_CONTRACT",
    "HUMAN_BALL_PROXIMITY_CONTRACT",
    "BALL_CONTACT_CANDIDATES_CONTRACT",
    "POSSESSION_HYPOTHESES_CONTRACT",
    "PROJECTED_POSITIONS_CONTRACT",
    "PHYSICAL_METRIC_RESULTS_CONTRACT",
    "EXPECTED_HUMAN_BALL_PROXIMITY_FP",
    "EXPECTED_BALL_CONTACT_CANDIDATES_FP",
    "EXPECTED_POSSESSION_HYPOTHESES_FP",
    "EXPECTED_PROJECTED_POSITIONS_FP",
    "EXPECTED_PHYSICAL_METRIC_RESULTS_FP",
    "PASSING_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_passing_contract",
    "load_all_passing_contracts",
    "passing_schema_fingerprints",
    "compile_passing_schemas",
    "assert_passing_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "passing_schema_dir",
    "load_passing_json_schema",
    "validate_against_json_schema",
]
