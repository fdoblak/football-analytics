"""Schema loading helpers for Stage 4A broadcast contracts."""

from __future__ import annotations

from typing import Any

from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.types import ContractSpec

CONTRACT_NAMES: tuple[str, ...] = (
    "shot_boundaries",
    "shot_segments",
    "camera_view_segments",
)


def load_broadcast_contract(name: str, version: int = 1, *, registry: Any = None) -> ContractSpec:
    if name not in CONTRACT_NAMES:
        raise ValueError(f"unknown broadcast contract: {name}")
    return get_contract(name, version, registry=registry)


def load_all_broadcast_contracts(*, registry: Any = None) -> dict[str, ContractSpec]:
    return {name: load_broadcast_contract(name, 1, registry=registry) for name in CONTRACT_NAMES}


def broadcast_schema_fingerprints(*, registry: Any = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in load_all_broadcast_contracts(registry=registry).items():
        out[name] = contract_fingerprint(spec)
    return out


def compile_broadcast_schemas(*, registry: Any = None) -> dict[str, Any]:
    return {
        name: compile_arrow_schema(spec)
        for name, spec in load_all_broadcast_contracts(registry=registry).items()
    }


def assert_broadcast_contracts_registered(*, registry: Any = None) -> None:
    names = set(list_contracts(registry=registry))
    missing = [n for n in CONTRACT_NAMES if n not in names]
    if missing:
        raise ValueError(f"broadcast contracts missing from registry: {missing}")


__all__ = [
    "CONTRACT_NAMES",
    "load_broadcast_contract",
    "load_all_broadcast_contracts",
    "broadcast_schema_fingerprints",
    "compile_broadcast_schemas",
    "assert_broadcast_contracts_registered",
]
