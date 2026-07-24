"""JSON schema helpers for Stage 14 orchestration (no new Arrow contracts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data.compiler import list_contracts
from football_analytics.data.registry import default_project_root, load_schema_registry
from football_analytics.orchestration.types import OrchestrationError

# Prefer JSON schemas; Arrow registry count remains Stage 13 freeze (45).
EXPECTED_REGISTRY_CONTRACT_COUNT = 45

ORCHESTRATION_JSON_SCHEMAS: tuple[str, ...] = (
    "pipeline_request",
    "pipeline_plan",
    "pipeline_run_status",
    "cancellation_receipt",
)

REVIEW_JSON_SCHEMAS: tuple[str, ...] = (
    "unified_review_package",
    "unified_review_decision",
    "unified_review_audit",
)

REPORT_JSON_SCHEMA = "single_player_report"

STAGE_CHAIN: tuple[str, ...] = (
    "ingest",
    "probe",
    "normalize",
    "timeline",
    "broadcast",
    "detection",
    "tracking",
    "identity",
    "calibration",
    "physical",
    "interaction",
    "passing",
    "events",
    "report",
)

REVIEW_DOMAINS: tuple[str, ...] = (
    "identity",
    "ball_ambiguity",
    "possession_contact",
    "pass",
    "dribble_duel",
    "attack_direction",
    "calibration",
)

RESERVED_FINAL_VISUAL_PATHS: tuple[str, ...] = (
    "/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png",
    "artifacts/final/single_player_analysis_summary.png",
)

GATE_HINT = (
    "PASS_WITH_FINDINGS — SINGLE PLAYER PIPELINE ACTIVE; "
    "STAGE 14 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)


def orchestration_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "orchestration"


def review_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "review"


def metrics_schema_dir(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "schemas" / "metrics"


def load_orchestration_json_schema(
    name: str, *, project_root: Path | None = None
) -> dict[str, Any]:
    if name not in ORCHESTRATION_JSON_SCHEMAS:
        raise OrchestrationError(f"unknown orchestration json schema: {name}")
    path = orchestration_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise OrchestrationError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationError("schema root must be object")
    return data


def load_review_json_schema(name: str, *, project_root: Path | None = None) -> dict[str, Any]:
    if name not in REVIEW_JSON_SCHEMAS:
        raise OrchestrationError(f"unknown review json schema: {name}")
    path = review_schema_dir(project_root=project_root) / f"{name}.schema.json"
    if path.is_symlink():
        raise OrchestrationError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationError("schema root must be object")
    return data


def load_report_json_schema(*, project_root: Path | None = None) -> dict[str, Any]:
    path = metrics_schema_dir(project_root=project_root) / f"{REPORT_JSON_SCHEMA}.schema.json"
    if path.is_symlink():
        raise OrchestrationError(f"symlink rejected: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationError("schema root must be object")
    return data


def validate_against_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


def assert_registry_contract_count_unchanged(*, project_root: Path | None = None) -> int:
    """Stage 14 prefers JSON schemas — do not bump Arrow registry without need."""
    from football_analytics.data.registry import default_registry_path

    _ = project_root  # reserved for future containment; registry path is package-rooted
    registry = load_schema_registry(default_registry_path())
    n = len(list_contracts(registry=registry))
    if n != EXPECTED_REGISTRY_CONTRACT_COUNT:
        raise OrchestrationError(
            f"Arrow registry count {n} != expected {EXPECTED_REGISTRY_CONTRACT_COUNT}"
        )
    return n


__all__ = [
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "ORCHESTRATION_JSON_SCHEMAS",
    "REVIEW_JSON_SCHEMAS",
    "REPORT_JSON_SCHEMA",
    "STAGE_CHAIN",
    "REVIEW_DOMAINS",
    "RESERVED_FINAL_VISUAL_PATHS",
    "GATE_HINT",
    "orchestration_schema_dir",
    "review_schema_dir",
    "metrics_schema_dir",
    "load_orchestration_json_schema",
    "load_review_json_schema",
    "load_report_json_schema",
    "validate_against_json_schema",
    "assert_registry_contract_count_unchanged",
]
