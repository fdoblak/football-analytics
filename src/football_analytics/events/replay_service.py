"""Stage 13A conservative replay candidate baseline service."""

from __future__ import annotations

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
from football_analytics.events.camera_position import resolve_camera_position
from football_analytics.events.eligibility import (
    implies_live,
    live_event_eligible,
    normalize_replay_status,
)
from football_analytics.events.evaluation import NOT_EVALUATED, evaluate_events
from football_analytics.events.policy import load_events_policy, policy_fingerprint
from football_analytics.events.replay_config import (
    ReplayConfigError,
    load_replay_config,
    replay_config_fingerprint,
)
from football_analytics.events.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN
from football_analytics.events.validation import validate_events_bundle


class ReplayServiceError(RuntimeError):
    """Replay candidate service failure."""


@dataclass
class ReplayServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    replay_parquet: str | None
    summary_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    replays: Sequence[Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract("replay_candidates", 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def classify_replay_context(
    ctx: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
    index: int,
) -> dict[str, Any]:
    thr = dict(config.get("thresholds") or {})
    live_min = float(thr.get("live_confidence_min", 0.70))
    replay_min = float(thr.get("replay_confidence_min", 0.55))
    dissolve_min = float(thr.get("dissolve_score_min", 0.55))
    graphics_min = float(thr.get("graphics_overlay_min", 0.45))
    scoreboard_min = float(thr.get("scoreboard_interruption_overlay_min", 0.60))

    dissolve = float(ctx.get("dissolve_score", 0.0) or 0.0)
    overlay = float(ctx.get("overlay_fraction", 0.0) or 0.0)
    hint = str(ctx.get("transition_hint") or "unknown")
    scoreboard = bool(ctx.get("scoreboard_interruption")) or overlay >= scoreboard_min

    candidate = str(ctx.get("replay_status_hint") or "unknown")
    conf = ctx.get("confidence")
    if conf is None:
        if dissolve >= dissolve_min or hint in {"dissolve", "fade"}:
            candidate = "replay_transition"
            conf = dissolve
        elif overlay >= graphics_min or scoreboard:
            candidate = "unknown"
            conf = overlay
        elif ctx.get("explicit_live") is True:
            candidate = "live"
            conf = float(ctx.get("live_confidence", live_min))
        else:
            candidate = "unknown"
            conf = 0.0

    status = normalize_replay_status(
        candidate_status=candidate,
        confidence=float(conf) if conf is not None else None,
        live_confidence_min=live_min,
        replay_confidence_min=replay_min,
    )
    # Hard rule: never invent live when uncertain signals present
    if status == "live" and (
        dissolve >= dissolve_min or scoreboard or hint in {"dissolve", "fade"}
    ):
        status = "unknown"

    view = str(ctx.get("view_family") or "unknown")
    cam = resolve_camera_position(
        view_family=view,
        supported_by_view=dict(config.get("supported_camera_position_by_view") or {}),
        uncertain=bool(ctx.get("camera_uncertain")),
    )
    live_ok = live_event_eligible(replay_status=status)
    reasons: list[str] = []
    if status == "unknown":
        reasons.append("REPLAY_UNCERTAIN_BLOCKS_LIVE")
    if scoreboard:
        reasons.append("SCOREBOARD_INTERRUPTION")
    if cam == "unknown" and view not in dict(config.get("supported_camera_position_by_view") or {}):
        reasons.append("CAMERA_POSITION_UNSUPPORTED")

    start = int(ctx.get("start_time_us", index * 1_000_000))
    end = int(ctx.get("end_time_us", start + 500_000))
    return {
        "run_id": run_id,
        "video_id": video_id,
        "replay_candidate_id": str(ctx.get("replay_candidate_id") or f"replay_{index:03d}"),
        "start_time_us": start,
        "end_time_us": end,
        "start_frame_index": ctx.get("start_frame_index"),
        "end_frame_index_exclusive": ctx.get("end_frame_index_exclusive"),
        "replay_status": status,
        "implies_live": implies_live(status),
        "live_event_eligible": live_ok,
        "camera_position": cam,
        "view_family": view,
        "graphics_status": str(
            ctx.get("graphics_status") or ("full_screen" if scoreboard else "unknown")
        ),
        "scoreboard_interruption": scoreboard,
        "transition_hint": hint if hint else "unknown",
        "confidence": float(conf) if conf is not None else None,
        "evidence_refs": list(ctx.get("evidence_refs") or []),
        "reason_codes": reasons,
        "quality_flags": list(ctx.get("quality_flags") or []),
        "review_status": "unreviewed",
        "manual_review_required": status == "unknown",
        "automatic_ceiling": "provisional",
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fp,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def compute_replay_candidates(
    *,
    output_dir: Path,
    contexts: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
) -> ReplayServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_replay_config(config_path, project_root=root)
    except ReplayConfigError as exc:
        raise ReplayServiceError(str(exc)) from exc
    config_fp = replay_config_fingerprint(config)
    policy = load_events_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)
    run_id = run_id or generate_run_id()
    video_id = video_id or "synthetic_video_13a"

    rows = [
        classify_replay_context(
            ctx, config=config, policy_fp=policy_fp, run_id=run_id, video_id=video_id, index=i
        )
        for i, ctx in enumerate(contexts, start=1)
    ]
    validate_events_bundle(replays=rows)
    table = _cast(rows)
    pq_path = output_dir / "replay_candidates.parquet"
    write_contract_parquet(table, pq_path, get_contract("replay_candidates", 1), overwrite=True)

    evaluation = evaluate_events()
    eval_path = output_dir / "evaluation.json"
    write_json_record(eval_path, evaluation, overwrite=True)
    summary = {
        "run_id": run_id,
        "video_id": video_id,
        "replay_count": len(rows),
        "live_eligible_count": sum(1 for r in rows if r["live_event_eligible"]),
        "unknown_replay_count": sum(1 for r in rows if r["replay_status"] == "unknown"),
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "evaluation_status": evaluation["evaluation_status"],
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "replay_parquet_sha256": sha256_file(pq_path),
        "created_at_utc": _utc_now(),
        "NOT_EVALUATED": NOT_EVALUATED,
    }
    summary_path = output_dir / "summary.json"
    write_json_record(summary_path, summary, overwrite=True)
    return ReplayServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        replay_parquet=str(pq_path),
        summary_json=str(summary_path),
        evaluation_json=str(eval_path),
        summary=summary,
        replays=rows,
    )


__all__ = [
    "ReplayServiceError",
    "ReplayServiceResult",
    "classify_replay_context",
    "compute_replay_candidates",
]
