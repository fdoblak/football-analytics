"""Plan a single-player analysis stage chain (Stage 14A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.orchestration.cache_keys import plan_fingerprint, stage_cache_key
from football_analytics.orchestration.config import (
    load_pipeline_config,
    pipeline_config_fingerprint,
)
from football_analytics.orchestration.contracts import (
    STAGE_CHAIN,
    load_orchestration_json_schema,
    validate_against_json_schema,
)
from football_analytics.orchestration.types import OrchestrationError

_FIXTURE_ENTRYPOINTS = frozenset({"physical", "interaction", "passing", "events"})
_REPORT_STAGES = frozenset({"report"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stage_mode(name: str) -> str:
    if name in _REPORT_STAGES:
        return "report_builder"
    if name in _FIXTURE_ENTRYPOINTS:
        return "fixture_entrypoint"
    return "synthetic_stub"


def build_pipeline_request(
    *,
    request_id: str,
    run_id: str,
    video_id: str,
    target_player_id: str,
    output_directory: str,
    config_fingerprint: str,
    mode: str = "synthetic_fixture",
    match_id: str | None = None,
    video_path: str | None = None,
    video_fingerprint: str | None = None,
    resume_from_stage: str | None = None,
    force_restart: bool = False,
    cancel_requested: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "match_id": match_id,
        "mode": mode,
        "video_path": video_path,
        "video_fingerprint": video_fingerprint,
        "config_fingerprint": config_fingerprint,
        "resume_from_stage": resume_from_stage,
        "force_restart": force_restart,
        "cancel_requested": cancel_requested,
        "output_directory": output_directory,
        "provenance": {
            "synthetic_fixture": mode == "synthetic_fixture",
            "never_mutate_user_video": True,
            "real_football_accuracy_validated": False,
            "notes": notes,
        },
    }
    schema = load_orchestration_json_schema("pipeline_request")
    validate_against_json_schema(body, schema)
    return body


def plan_pipeline(
    request: Mapping[str, Any],
    *,
    stages: Sequence[str] | None = None,
    project_root: Any = None,
) -> dict[str, Any]:
    schema = load_orchestration_json_schema("pipeline_request")
    validate_against_json_schema(dict(request), schema)
    cfg = load_pipeline_config(project_root=project_root)
    cfg_fp = pipeline_config_fingerprint(cfg)
    if request["config_fingerprint"] != cfg_fp:
        raise OrchestrationError("request config_fingerprint mismatch vs loaded pipeline config")

    chain = list(stages) if stages is not None else list(cfg.get("stages") or STAGE_CHAIN)
    if chain != list(STAGE_CHAIN) and stages is None:
        # Config may list same chain; validate subset/order against canonical.
        if list(cfg.get("stages") or []) != list(STAGE_CHAIN):
            raise OrchestrationError("pipeline config stages must match STAGE_CHAIN")
        chain = list(STAGE_CHAIN)

    upstream: dict[str, str] = {}
    stage_rows: list[dict[str, Any]] = []
    for i, name in enumerate(chain):
        depends = [chain[i - 1]] if i > 0 else []
        ck = stage_cache_key(
            stage_name=name,
            run_id=str(request["run_id"]),
            video_id=str(request["video_id"]),
            target_player_id=str(request["target_player_id"]),
            config_fingerprint=cfg_fp,
            upstream_fingerprints=upstream,
            mode=str(request["mode"]),
        )
        row = {
            "name": name,
            "order": i,
            "cache_key": ck,
            "depends_on": depends,
            "mode": _stage_mode(name),
        }
        stage_rows.append(row)
        upstream[name] = ck

    plan: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": f"plan_{request['request_id']}",
        "request_id": request["request_id"],
        "run_id": request["run_id"],
        "config_fingerprint": cfg_fp,
        "created_at_utc": _utc_now(),
        "stages": stage_rows,
        "warnings": [],
        "plan_fingerprint": "",
    }
    plan["plan_fingerprint"] = plan_fingerprint(
        stage_rows, request_id=str(request["request_id"]), run_id=str(request["run_id"])
    )
    if request.get("video_path"):
        plan["warnings"].append("user_video_path_recorded_read_only")
    plan_schema = load_orchestration_json_schema("pipeline_plan")
    validate_against_json_schema(plan, plan_schema)
    return plan


__all__ = ["build_pipeline_request", "plan_pipeline"]
