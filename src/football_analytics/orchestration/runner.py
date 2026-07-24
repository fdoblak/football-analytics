"""Run / resume single-player pipeline with isolation + receipts (Stage 14A)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.orchestration.cleanup import (
    cleanup_stage_owned_temp,
    write_cleanup_receipt,
)
from football_analytics.orchestration.contracts import (
    GATE_HINT,
    load_orchestration_json_schema,
    validate_against_json_schema,
)
from football_analytics.orchestration.planner import plan_pipeline
from football_analytics.orchestration.stage_handlers import execute_stage_handler
from football_analytics.orchestration.types import (
    OverwriteForbiddenError,
    StaleArtifactError,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class PipelineRunResult:
    accepted: bool
    overall_status: str
    status_path: str
    plan_path: str
    report_json_path: str | None
    cancellation_receipt_path: str | None
    summary: dict[str, Any]


def _status_fingerprint(status_body: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in status_body.items() if k != "status_fingerprint"}
    return hash_canonical_json(payload)


def _write_status(path: Path, body: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    body = dict(body)
    body["updated_at_utc"] = _utc_now()
    body["status_fingerprint"] = _status_fingerprint(body)
    schema = load_orchestration_json_schema("pipeline_run_status")
    validate_against_json_schema(body, schema)
    write_json_record(path, body, overwrite=overwrite)
    return body


def _write_cancellation(
    path: Path,
    *,
    run_id: str,
    request_id: str,
    cancelled_after_stage: str | None,
    reason: str,
    stages_completed: list[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "request_id": request_id,
        "cancelled_at_utc": _utc_now(),
        "cancelled_after_stage": cancelled_after_stage,
        "reason": reason,
        "stages_completed": stages_completed,
        "user_video_mutated": False,
        "receipt_fingerprint": "",
    }
    body["receipt_fingerprint"] = hash_canonical_json(
        {k: v for k, v in body.items() if k != "receipt_fingerprint"}
    )
    schema = load_orchestration_json_schema("cancellation_receipt")
    validate_against_json_schema(body, schema)
    write_json_record(path, body, overwrite=True)
    return body


def _load_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StaleArtifactError("status root must be object")
    expected = _status_fingerprint(data)
    if data.get("status_fingerprint") != expected:
        raise StaleArtifactError("stale status fingerprint rejection")
    return data


def run_pipeline(
    request: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    light_stubs_only: bool = False,
) -> PipelineRunResult:
    """Execute plan sequentially with resume/no-overwrite/failure isolation."""
    from football_analytics.orchestration.report.builder import build_and_write_report_stage

    plan = plan_pipeline(request, project_root=project_root)
    out_root = Path(str(request["output_directory"]))
    out_root.mkdir(parents=True, exist_ok=True)
    # Never mutate user video — only record path if present.
    video_path = request.get("video_path")
    if video_path:
        vp = Path(str(video_path))
        if vp.is_symlink():
            raise OverwriteForbiddenError("user video symlink rejected")

    plan_path = out_root / "pipeline_plan.json"
    status_path = out_root / "pipeline_run_status.json"
    if plan_path.exists() and not request.get("force_restart"):
        # no-overwrite of plan unless restart
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise StaleArtifactError("existing plan fingerprint mismatch; use force_restart")
    else:
        write_json_record(plan_path, plan, overwrite=bool(request.get("force_restart")))

    prior = None if request.get("force_restart") else _load_status(status_path)
    resume_from = request.get("resume_from_stage")
    if prior and resume_from:
        names = [s["name"] for s in prior.get("stages") or []]
        if resume_from not in names:
            raise StaleArtifactError(f"resume_from_stage unknown: {resume_from}")

    stage_states: list[dict[str, Any]] = []
    for row in plan["stages"]:
        prev_status = "pending"
        if prior:
            for ps in prior.get("stages") or []:
                if ps["name"] == row["name"]:
                    prev_status = str(ps["status"])
                    break
        stage_states.append(
            {
                "name": row["name"],
                "status": prev_status if prev_status == "succeeded" else "pending",
                "cache_key": row["cache_key"],
                "artifact_path": None,
                "error_code": None,
                "warnings": [],
            }
        )

    status_body: dict[str, Any] = {
        "schema_version": 1,
        "run_id": request["run_id"],
        "request_id": request["request_id"],
        "overall_status": "running",
        "config_fingerprint": plan["config_fingerprint"],
        "updated_at_utc": _utc_now(),
        "status_fingerprint": "",
        "stages": stage_states,
        "partial": False,
        "cancellation_receipt_path": None,
        "report_json_path": None,
        "gate_hint": GATE_HINT,
        "warnings": list(plan.get("warnings") or []),
    }
    _write_status(status_path, status_body, overwrite=True)

    cancel_path: str | None = None
    report_path: str | None = None
    completed: list[str] = []
    any_failed = False
    skipped_after_cancel = False

    for i, row in enumerate(plan["stages"]):
        st = stage_states[i]
        if request.get("cancel_requested") or skipped_after_cancel:
            if st["status"] != "succeeded":
                st["status"] = "cancelled"
            skipped_after_cancel = True
            continue

        # Resume: skip already succeeded stages (deterministic cache key match).
        if st["status"] == "succeeded" and not request.get("force_restart"):
            if prior:
                for ps in prior.get("stages") or []:
                    if ps["name"] == row["name"] and ps.get("cache_key") != row["cache_key"]:
                        raise StaleArtifactError(
                            f"stale cache key for stage {row['name']}; reject resume"
                        )
            completed.append(row["name"])
            continue

        if resume_from and row["name"] != resume_from and row["name"] not in completed:
            # Skip stages before resume point only if they were succeeded; else run.
            names_before = [r["name"] for r in plan["stages"][:i]]
            names_from_here = [r["name"] for r in plan["stages"][i:]]
            if (
                resume_from in names_from_here
                and row["name"] in names_before
                and st["status"] != "succeeded"
            ):
                st["status"] = "skipped"
                st["warnings"].append("skipped_before_resume_without_prior_success")
                continue

        stage_dir = out_root / "stages" / row["name"]
        receipt_path = stage_dir / "stage_receipt.json"
        if (
            receipt_path.exists()
            and not request.get("force_restart")
            and st["status"] != "succeeded"
        ):
            # no-overwrite of successful receipts; allow re-run only with restart
            existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing_receipt.get("cache_key") != row["cache_key"]:
                raise StaleArtifactError(f"stale stage receipt for {row['name']}")
            if existing_receipt.get("status") == "succeeded":
                st["status"] = "succeeded"
                st["artifact_path"] = str(receipt_path)
                completed.append(row["name"])
                continue

        st["status"] = "running"
        _write_status(status_path, status_body, overwrite=True)

        mode = row["mode"]
        if light_stubs_only and mode == "fixture_entrypoint":
            mode = "synthetic_stub"

        try:
            receipt = execute_stage_handler(
                stage_name=row["name"],
                stage_dir=stage_dir,
                run_id=str(request["run_id"]),
                cache_key=row["cache_key"],
                mode=mode,
                report_builder=lambda **kw: build_and_write_report_stage(
                    request=request,
                    plan=plan,
                    stage_states=stage_states,
                    out_root=out_root,
                    **kw,
                ),
            )
            st["status"] = "succeeded" if receipt.get("status") == "succeeded" else "failed"
            st["artifact_path"] = str(stage_dir / "stage_receipt.json")
            if row["name"] == "report":
                report_path = str(out_root / "single_player_report.json")
                status_body["report_json_path"] = report_path
            if st["status"] == "succeeded":
                completed.append(row["name"])
            else:
                any_failed = True
                st["error_code"] = "STAGE_FAILED"
        except Exception as exc:  # noqa: BLE001
            # Failure isolation: mark failed, continue remaining as skipped/pending attempt
            any_failed = True
            st["status"] = "failed"
            st["error_code"] = type(exc).__name__
            st["warnings"].append(str(exc)[:200])
            # continue to next stages (partial)

        _write_status(status_path, status_body, overwrite=True)

    if request.get("cancel_requested"):
        cpath = out_root / "cancellation_receipt.json"
        _write_cancellation(
            cpath,
            run_id=str(request["run_id"]),
            request_id=str(request["request_id"]),
            cancelled_after_stage=completed[-1] if completed else None,
            reason="cancel_requested",
            stages_completed=completed,
        )
        cancel_path = str(cpath)
        status_body["cancellation_receipt_path"] = cancel_path
        status_body["overall_status"] = "cancelled"
    elif any_failed and completed:
        status_body["overall_status"] = "partial"
        status_body["partial"] = True
    elif any_failed:
        status_body["overall_status"] = "failed"
    else:
        status_body["overall_status"] = "succeeded"

    _write_status(status_path, status_body, overwrite=True)

    removed = cleanup_stage_owned_temp(out_root)
    write_cleanup_receipt(
        out_root / "cleanup_receipt.json", removed=removed, run_id=str(request["run_id"])
    )

    accepted = status_body["overall_status"] in {"succeeded", "partial"}
    return PipelineRunResult(
        accepted=accepted,
        overall_status=str(status_body["overall_status"]),
        status_path=str(status_path),
        plan_path=str(plan_path),
        report_json_path=report_path or status_body.get("report_json_path"),
        cancellation_receipt_path=cancel_path,
        summary={
            "gate_hint": GATE_HINT,
            "stages_completed": completed,
            "overall_status": status_body["overall_status"],
            "real_football_accuracy_validated": False,
            "user_video_mutated": False,
            "light_stubs_only": light_stubs_only,
        },
    )


def resume_pipeline(
    request: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    light_stubs_only: bool = False,
) -> PipelineRunResult:
    body = dict(request)
    body["mode"] = "resume"
    body["force_restart"] = False
    return run_pipeline(body, project_root=project_root, light_stubs_only=light_stubs_only)


__all__ = ["PipelineRunResult", "run_pipeline", "resume_pipeline"]
