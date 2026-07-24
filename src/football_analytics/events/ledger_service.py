"""Stage 13C append-only canonical target event ledger service."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import default_project_root
from football_analytics.events.dedup import suppress_duplicate_events
from football_analytics.events.evaluation import NOT_EVALUATED, evaluate_events
from football_analytics.events.ledger_config import (
    LedgerConfigError,
    ledger_config_fingerprint,
    load_ledger_config,
)
from football_analytics.events.policy import load_events_policy, policy_fingerprint
from football_analytics.events.semantics import (
    evaluation_leakage_guard,
    event_live_eligible,
    extract_source_event_id,
    family_for_source_contract,
    replay_status_from_source,
)
from football_analytics.events.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN
from football_analytics.events.validation import validate_events_bundle


class LedgerServiceError(RuntimeError):
    """Ledger service failure."""


@dataclass
class LedgerServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    ledger_parquet: str | None
    revisions_parquet: str | None
    summary_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    ledger: Sequence[Mapping[str, Any]]
    revisions: Sequence[Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def _source_batches(
    sources: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    out: list[tuple[str, Mapping[str, Any]]] = []
    for contract, rows in (sources or {}).items():
        for row in rows:
            out.append((str(contract), row))
    return out


def build_ledger_rows(
    *,
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    policy_fp: str,
    run_id: str,
    video_id: str,
    period_id: str = "period_1",
    half_id: str = "first_half",
    anonymous_team_id: str | None = "anon_team_a",
    overlap_us: int = 500_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    for i, (contract, src) in enumerate(_source_batches(sources), start=1):
        leaks = evaluation_leakage_guard(src)
        if leaks:
            raise LedgerServiceError(f"evaluation leakage in source {contract}: {leaks}")
        source_id = extract_source_event_id(contract, src)
        family = family_for_source_contract(contract)
        if "start_time_us" in src:
            start = int(src["start_time_us"])
        elif "touch_time_us" in src:
            start = int(src["touch_time_us"])
        else:
            start = i * 1_000_000
        end = int(src["end_time_us"]) if "end_time_us" in src else start + 400_000
        replay_status = replay_status_from_source(src)
        live_ok = event_live_eligible(src)
        lineage = {
            "source_contract": contract,
            "source_event_id": source_id,
            "preserved": True,
            "destructive_merge": False,
        }
        ledger_id = f"led_{i:04d}_{family}"
        row = {
            "run_id": run_id,
            "video_id": video_id,
            "ledger_event_id": ledger_id,
            "event_family": family,
            "event_type": str(src.get("event_type") or family),
            "source_contract": contract,
            "source_event_id": source_id,
            "source_refs": [f"{contract}:{source_id}"],
            "lineage_json": json.dumps(lineage, sort_keys=True, separators=(",", ":")),
            "start_time_us": start,
            "end_time_us": end,
            "period_id": str(src.get("period_id") or period_id),
            "half_id": str(src.get("half_id") or half_id),
            "anonymous_team_id": anonymous_team_id,
            "target_relationship": str(src.get("target_relationship") or "confirmed_target"),
            "target_human_track_id": src.get("target_human_track_id"),
            "outcome": str(src.get("outcome") or src.get("event_state") or "uncertain"),
            "event_state": str(src.get("event_state") or "provisional"),
            "replay_status": replay_status,
            "live_event_eligible": live_ok,
            "suppressed_duplicate": False,
            "suppressed_by_id": None,
            "overlap_group_id": None,
            "revision_status": "active",
            "current_revision_id": None,
            "review_status": str(src.get("review_status") or "unreviewed"),
            "manual_review_required": bool(src.get("manual_review_required", False)),
            "automatic_ceiling": "provisional",
            "confidence": src.get("confidence") or src.get("uncertainty"),
            "coverage": src.get("coverage"),
            "evidence_refs": list(src.get("evidence_refs") or []),
            "reason_codes": list(src.get("reason_codes") or []),
            "quality_flags": list(src.get("quality_flags") or []),
            "metric_origin": METRIC_ORIGIN,
            "definition_style": DEFINITION_STYLE,
            "policy_fingerprint": policy_fp,
            "attributes_json": src.get("attributes_json")
            or json.dumps(
                {
                    k: src[k]
                    for k in ("zone_neutral", "is_long_pass", "passer_is_target")
                    if k in src
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "provenance_json": src.get("provenance_json"),
            "contract_version": CONTRACT_VERSION,
        }
        if row["event_state"] == "confirmed":
            row["event_state"] = "provisional"
            existing = row.get("reason_codes")
            reasons: list[str] = [str(x) for x in existing] if isinstance(existing, list) else []
            reasons.append("AUTOMATIC_CONFIRMED_FORBIDDEN")
            row["reason_codes"] = reasons
        rows.append(row)

    rows = suppress_duplicate_events(rows, overlap_us=overlap_us)
    for row in rows:
        if row.get("suppressed_duplicate") is True:
            rev_id = f"rev_suppress_{row['ledger_event_id']}"
            row["revision_status"] = "revised"
            row["current_revision_id"] = rev_id
            revisions.append(
                {
                    "run_id": run_id,
                    "video_id": video_id,
                    "revision_id": rev_id,
                    "ledger_event_id": row["ledger_event_id"],
                    "revision_action": "suppress_duplicate",
                    "previous_revision_id": None,
                    "reason_codes": ["DUPLICATE_SUPPRESSED"],
                    "review_status": "not_required",
                    "actor": "pipeline",
                    "created_at_utc": _utc_now(),
                    "payload_json": json.dumps(
                        {"suppressed_by_id": row.get("suppressed_by_id")},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "evidence_refs": list(row.get("evidence_refs") or []),
                    "metric_origin": METRIC_ORIGIN,
                    "definition_style": DEFINITION_STYLE,
                    "policy_fingerprint": policy_fp,
                    "contract_version": CONTRACT_VERSION,
                }
            )
    return rows, revisions


def build_target_event_ledger(
    *,
    output_dir: Path,
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
    period_id: str = "period_1",
    half_id: str = "first_half",
    anonymous_team_id: str | None = "anon_team_a",
) -> LedgerServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_ledger_config(config_path, project_root=root)
    except LedgerConfigError as exc:
        raise LedgerServiceError(str(exc)) from exc
    config_fp = ledger_config_fingerprint(config)
    policy = load_events_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)
    run_id = run_id or generate_run_id()
    video_id = video_id or "synthetic_video_13c"
    dedup = dict(config.get("dedup") or {})
    overlap_us = int(dedup.get("same_family_overlap_us", 500_000))

    if not sources:
        raise LedgerServiceError("EMPTY_INPUT")

    ledger, revisions = build_ledger_rows(
        sources=sources,
        policy_fp=policy_fp,
        run_id=run_id,
        video_id=video_id,
        period_id=period_id,
        half_id=half_id,
        anonymous_team_id=anonymous_team_id,
        overlap_us=overlap_us,
    )
    validate_events_bundle(ledger=ledger, revisions=revisions)

    led_path = output_dir / "target_event_ledger.parquet"
    rev_path = output_dir / "event_revisions.parquet"
    write_contract_parquet(
        _cast("target_event_ledger", ledger),
        led_path,
        get_contract("target_event_ledger", 1),
        overwrite=True,
    )
    write_contract_parquet(
        _cast("event_revisions", revisions),
        rev_path,
        get_contract("event_revisions", 1),
        overwrite=True,
    )
    evaluation = evaluate_events(ledger_rows=ledger)
    eval_path = output_dir / "evaluation.json"
    write_json_record(eval_path, evaluation, overwrite=True)
    summary = {
        "run_id": run_id,
        "video_id": video_id,
        "ledger_count": len(ledger),
        "revision_count": len(revisions),
        "suppressed_count": sum(1 for r in ledger if r.get("suppressed_duplicate")),
        "active_count": sum(1 for r in ledger if not r.get("suppressed_duplicate")),
        "live_eligible_count": sum(1 for r in ledger if r.get("live_event_eligible")),
        "append_only": True,
        "no_destructive_merge": True,
        "source_contracts": sorted(sources.keys()),
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "evaluation_status": evaluation["evaluation_status"],
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "ledger_parquet_sha256": sha256_file(led_path),
        "revisions_parquet_sha256": sha256_file(rev_path),
        "created_at_utc": _utc_now(),
        "NOT_EVALUATED": NOT_EVALUATED,
    }
    summary_path = output_dir / "summary.json"
    write_json_record(summary_path, summary, overwrite=True)
    return LedgerServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        ledger_parquet=str(led_path),
        revisions_parquet=str(rev_path),
        summary_json=str(summary_path),
        evaluation_json=str(eval_path),
        summary=summary,
        ledger=ledger,
        revisions=revisions,
    )


__all__ = [
    "LedgerServiceError",
    "LedgerServiceResult",
    "build_ledger_rows",
    "build_target_event_ledger",
]
