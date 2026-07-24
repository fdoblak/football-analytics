"""Schema loading helpers for Stage 13 target-events contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import default_project_root
from football_analytics.data.types import ContractSpec
from football_analytics.events.types import EventsContractError

REPLAY_CANDIDATES_CONTRACT = "replay_candidates"
TARGET_EVENT_LEDGER_CONTRACT = "target_event_ledger"
EVENT_REVISIONS_CONTRACT = "event_revisions"

# Frozen upstream Stage 10–12 contracts.
PASS_CANDIDATES_CONTRACT = "pass_candidates"
PASS_OUTCOMES_CONTRACT = "pass_outcomes"
RECEPTION_CANDIDATES_CONTRACT = "reception_candidates"
TAKE_ON_ATTEMPTS_CONTRACT = "take_on_attempts"
GROUND_DUEL_CANDIDATES_CONTRACT = "ground_duel_candidates"
AERIAL_DUEL_CANDIDATES_CONTRACT = "aerial_duel_candidates"
TACKLE_EVENTS_CONTRACT = "tackle_events"
RECOVERY_EVENTS_CONTRACT = "recovery_events"
TURNOVER_EVENTS_CONTRACT = "turnover_events"
CLEARANCE_EVENTS_CONTRACT = "clearance_events"
TARGET_BALL_TOUCHES_CONTRACT = "target_ball_touches"

EXPECTED_PASS_CANDIDATES_FP = "923f16283c45eca74d32bae4570d00935952628640a295334ad09600e8d55053"
EXPECTED_PASS_OUTCOMES_FP = "bc65b9e3079f388854aa2b962c04b03900ec030313e89844f54012ab789584b8"
EXPECTED_RECEPTION_CANDIDATES_FP = (
    "ceb755b8cc2de05e87f48bbc645e70f2fed46b772b33ed628bb3983eef03336f"
)
EXPECTED_TAKE_ON_ATTEMPTS_FP = "b38f29ac54e00f2281906ad7b55107a70196b4312c52a74cbfb425fc23c25412"
EXPECTED_GROUND_DUEL_CANDIDATES_FP = (
    "47c7d1eee2857cd89b11f5625ac94fe8f19036e21ce7ef5030659d8dfc15834c"
)
EXPECTED_AERIAL_DUEL_CANDIDATES_FP = (
    "be840728078fe43da39e6336b1188a4e260ae8d701469201132d9c96b97360de"
)
EXPECTED_TACKLE_EVENTS_FP = "0e6d278d54d8e3c59f2bc3de8ef18dae8112c507e24ea73e241bf27385f966fe"
EXPECTED_RECOVERY_EVENTS_FP = "411eedc7cdd44b825ba990c3b7223df540e204c5082264cd2d06ea115cb14d39"
EXPECTED_TURNOVER_EVENTS_FP = "e3b2e14d9b7cf0e3474a764bd84b2d10c74c3e809f3faf660609b07dd47146ba"
EXPECTED_CLEARANCE_EVENTS_FP = "b29f1e433734c9a8fb3972152334946937b419c02d7eea073cbc1f3ce7b4fe9b"
EXPECTED_TARGET_BALL_TOUCHES_FP = "89c3a9f24b7d01345bc59a4b54c8f2a62603dd66f49bd1f6e3f560fc595a77d3"

EVENTS_ARROW_CONTRACTS: tuple[str, ...] = (
    REPLAY_CANDIDATES_CONTRACT,
    TARGET_EVENT_LEDGER_CONTRACT,
    EVENT_REVISIONS_CONTRACT,
)

JSON_SCHEMA_NAMES: tuple[str, ...] = (
    "events_request",
    "events_run_receipt",
    "events_evaluation",
    "events_quality",
    "manual_review_queue",
    "attack_direction_evidence",
)

EXPECTED_REGISTRY_CONTRACT_COUNT = 45


def load_events_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    allowed = set(EVENTS_ARROW_CONTRACTS) | {
        PASS_CANDIDATES_CONTRACT,
        PASS_OUTCOMES_CONTRACT,
        RECEPTION_CANDIDATES_CONTRACT,
        TAKE_ON_ATTEMPTS_CONTRACT,
        GROUND_DUEL_CANDIDATES_CONTRACT,
        AERIAL_DUEL_CANDIDATES_CONTRACT,
        TACKLE_EVENTS_CONTRACT,
        RECOVERY_EVENTS_CONTRACT,
        TURNOVER_EVENTS_CONTRACT,
        CLEARANCE_EVENTS_CONTRACT,
        TARGET_BALL_TOUCHES_CONTRACT,
        "frames",
        "videos",
        "analysis_windows",
        "camera_view_segments",
    }
    if name not in allowed:
        raise EventsContractError(f"unknown events-related contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_events_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {
        name: load_events_contract(name, 1, registry=registry) for name in EVENTS_ARROW_CONTRACTS
    }


def events_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_events_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    for name in (
        PASS_CANDIDATES_CONTRACT,
        PASS_OUTCOMES_CONTRACT,
        RECEPTION_CANDIDATES_CONTRACT,
        TAKE_ON_ATTEMPTS_CONTRACT,
        GROUND_DUEL_CANDIDATES_CONTRACT,
        AERIAL_DUEL_CANDIDATES_CONTRACT,
        TACKLE_EVENTS_CONTRACT,
        RECOVERY_EVENTS_CONTRACT,
        TURNOVER_EVENTS_CONTRACT,
        CLEARANCE_EVENTS_CONTRACT,
        TARGET_BALL_TOUCHES_CONTRACT,
    ):
        out[name] = contract_fingerprint(load_events_contract(name, 1, registry=registry))
    return out


def compile_events_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_events_contracts(registry=registry).items()
    }


def assert_events_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in EVENTS_ARROW_CONTRACTS if n not in names]
    if missing:
        raise EventsContractError(f"events contracts missing from registry: {missing}")


def assert_frozen_upstream_fingerprints(*, registry: Any = None) -> None:
    fps = events_schema_fingerprints(registry=registry)
    checks = {
        PASS_CANDIDATES_CONTRACT: EXPECTED_PASS_CANDIDATES_FP,
        PASS_OUTCOMES_CONTRACT: EXPECTED_PASS_OUTCOMES_FP,
        RECEPTION_CANDIDATES_CONTRACT: EXPECTED_RECEPTION_CANDIDATES_FP,
        TAKE_ON_ATTEMPTS_CONTRACT: EXPECTED_TAKE_ON_ATTEMPTS_FP,
        GROUND_DUEL_CANDIDATES_CONTRACT: EXPECTED_GROUND_DUEL_CANDIDATES_FP,
        AERIAL_DUEL_CANDIDATES_CONTRACT: EXPECTED_AERIAL_DUEL_CANDIDATES_FP,
        TACKLE_EVENTS_CONTRACT: EXPECTED_TACKLE_EVENTS_FP,
        RECOVERY_EVENTS_CONTRACT: EXPECTED_RECOVERY_EVENTS_FP,
        TURNOVER_EVENTS_CONTRACT: EXPECTED_TURNOVER_EVENTS_FP,
        CLEARANCE_EVENTS_CONTRACT: EXPECTED_CLEARANCE_EVENTS_FP,
        TARGET_BALL_TOUCHES_CONTRACT: EXPECTED_TARGET_BALL_TOUCHES_FP,
    }
    for name, expected in checks.items():
        if fps[name] != expected:
            raise EventsContractError(f"{name} v1 fingerprint changed")


def events_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "events"


def load_events_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in JSON_SCHEMA_NAMES:
        raise EventsContractError(f"unknown events json schema: {name}")
    path = events_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise EventsContractError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EventsContractError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


__all__ = [
    "REPLAY_CANDIDATES_CONTRACT",
    "TARGET_EVENT_LEDGER_CONTRACT",
    "EVENT_REVISIONS_CONTRACT",
    "EVENTS_ARROW_CONTRACTS",
    "JSON_SCHEMA_NAMES",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "load_events_contract",
    "load_all_events_contracts",
    "events_schema_fingerprints",
    "compile_events_schemas",
    "assert_events_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "events_schema_dir",
    "load_events_json_schema",
    "validate_against_json_schema",
]
