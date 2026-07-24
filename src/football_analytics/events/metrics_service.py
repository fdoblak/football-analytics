"""Stage 13D coverage-aware target event metrics aggregation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.registry import default_project_root
from football_analytics.events.attack_direction import (
    attack_relative_evaluable,
    resolve_period_attack_direction,
)
from football_analytics.events.evaluation import NOT_EVALUATED, evaluate_events
from football_analytics.events.metrics_config import (
    MetricsConfigError,
    load_metrics_config,
    metrics_config_fingerprint,
)
from football_analytics.events.policy import load_events_policy, policy_fingerprint
from football_analytics.events.types import DEFINITION_STYLE, METRIC_ORIGIN


class MetricsServiceError(RuntimeError):
    """Target event metrics failure."""


@dataclass
class MetricsServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    metrics_json: str | None
    attack_direction_json: str | None
    summary_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _metric(
    *,
    metric_id: str,
    value: Any,
    status: str,
    numerator: Any = None,
    denominator: Any = None,
    coverage: float | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    sources: Sequence[str] | None = None,
    review_status: str = "unreviewed",
    warnings: Sequence[str] | None = None,
    definition_version: int = 1,
    provenance: str = "video_derived",
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value": value,
        "status": status,
        "numerator": numerator,
        "denominator": denominator,
        "definition": metric_id,
        "definition_version": definition_version,
        "source_events": list(sources or []),
        "coverage": coverage,
        "confidence": confidence,
        "provenance": provenance,
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "review_status": review_status,
        "warnings": list(warnings or []),
        "reason_not_evaluable": reason,
    }


def _active(ledger: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        r
        for r in ledger
        if not r.get("suppressed_duplicate")
        and str(r.get("revision_status")) in {"active", "revised"}
        and str(r.get("event_state")) not in {"rejected", "not_evaluable"}
    ]


def _family(rows: Sequence[Mapping[str, Any]], family: str) -> list[Mapping[str, Any]]:
    return [r for r in rows if str(r.get("event_family")) == family]


def _attrs(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("attributes_json")
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def aggregate_target_event_metrics(
    *,
    ledger: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
    attack_direction_manual: str | None = None,
    attack_direction_config: str | None = None,
    interaction_coverage: float | None = 0.8,
    period_id: str = "period_1",
    half_id: str = "first_half",
    anonymous_team_id: str | None = "anon_team_a",
) -> tuple[dict[str, Any], dict[str, Any]]:
    thr = dict(config.get("thresholds") or {})
    min_cov = float(thr.get("min_joint_coverage_ratio", 0.1))
    min_passes = int(thr.get("min_evaluable_passes", 1))
    min_duels = int(thr.get("min_evaluable_duels", 1))
    min_take = int(thr.get("min_evaluable_take_ons", 1))
    min_aerial = int(thr.get("min_evaluable_aerials", 1))
    min_long = int(thr.get("min_evaluable_long_passes", 1))
    long_thr = float(thr.get("long_pass_distance_m", 30.0))
    coverage = interaction_coverage
    low_cov = coverage is None or float(coverage) < min_cov

    attack = resolve_period_attack_direction(
        run_id=run_id,
        video_id=video_id,
        period_id=period_id,
        half_id=half_id,
        anonymous_team_id=anonymous_team_id,
        config_direction=attack_direction_config
        or str((config.get("attack_direction") or {}).get("default", "unknown")),
        manual_direction=attack_direction_manual,
    )
    directional_ok = attack_relative_evaluable(attack)
    attack_dir = str(attack["attack_direction"])

    rows = _active(ledger)
    live_rows = [r for r in rows if r.get("live_event_eligible") is True]

    def count_metric(metric_id: str, n: int, sources: Sequence[str]) -> dict[str, Any]:
        if low_cov:
            return _metric(
                metric_id=metric_id,
                value=None,
                status="not_evaluable",
                numerator=None,
                denominator=None,
                coverage=coverage,
                reason="insufficient_coverage",
                sources=sources,
                warnings=["low_coverage"],
            )
        return _metric(
            metric_id=metric_id,
            value=n,
            status="provisional",
            numerator=n,
            denominator=None,
            coverage=coverage,
            confidence=0.7,
            sources=sources,
        )

    def rate_metric(
        metric_id: str,
        num: int,
        den: int,
        *,
        min_obs: int,
        sources: Sequence[str],
        extra_reason: str | None = None,
    ) -> dict[str, Any]:
        if low_cov:
            return _metric(
                metric_id=metric_id,
                value=None,
                status="not_evaluable",
                coverage=coverage,
                reason="insufficient_coverage",
                sources=sources,
            )
        if den < min_obs:
            return _metric(
                metric_id=metric_id,
                value=None,
                status="not_evaluable",
                numerator=num,
                denominator=den,
                coverage=coverage,
                reason=extra_reason or "insufficient_observations",
                sources=sources,
            )
        return _metric(
            metric_id=metric_id,
            value=float(num) / float(den),
            status="provisional",
            numerator=num,
            denominator=den,
            coverage=coverage,
            confidence=0.7,
            sources=sources,
        )

    passes = [
        r
        for r in _family(live_rows, "pass")
        if str(r.get("target_relationship")) == "confirmed_target"
        and str(r.get("outcome")) in {"completed", "incomplete", "failed", "attempted", "uncertain"}
    ]
    # Prefer pass_outcomes-like rows
    evaluable_passes = [
        r for r in passes if str(r.get("outcome")) in {"completed", "incomplete", "failed"}
    ]
    completed = [r for r in evaluable_passes if str(r.get("outcome")) == "completed"]
    receptions = [
        r
        for r in _family(live_rows, "reception")
        if str(r.get("target_relationship")) == "confirmed_target"
    ]
    take_ons = _family(live_rows, "take_on")
    take_ok = [r for r in take_ons if str(r.get("outcome")) in {"beaten", "retained", "successful"}]
    take_fail = [r for r in take_ons if str(r.get("outcome")) in {"lost", "failed"}]
    take_eval = take_ok + take_fail
    ground = _family(live_rows, "ground_duel")
    ground_won = [r for r in ground if str(r.get("outcome")) == "won"]
    ground_lost = [r for r in ground if str(r.get("outcome")) == "lost"]
    ground_eval = ground_won + ground_lost
    aerial = _family(live_rows, "aerial_duel")
    aerial_won = [r for r in aerial if str(r.get("outcome")) == "won"]
    aerial_lost = [r for r in aerial if str(r.get("outcome")) == "lost"]
    aerial_eval = aerial_won + aerial_lost
    tackles = _family(live_rows, "tackle")
    recoveries = _family(live_rows, "recovery")
    turnovers = _family(live_rows, "turnover")
    clearances = [
        r for r in _family(live_rows, "clearance") if r.get("suppressed_duplicate") is not True
    ]
    # clearance implies_clearance may be in attributes; still count family rows with implies
    clear_ok = []
    for r in clearances:
        attrs = _attrs(r)
        if attrs.get("implies_clearance") is False:
            continue
        clear_ok.append(r)
    if not clear_ok and clearances:
        clear_ok = [r for r in clearances if str(r.get("outcome")) not in {"rejected"}]

    touches = _family(live_rows, "touch")
    box_touches = []
    for r in touches:
        attrs = _attrs(r)
        if (
            attrs.get("is_box_touch_candidate") is True
            and attrs.get("penalty_presence_alone") is not True
        ) or (
            attrs.get("in_penalty_area") is True and attrs.get("has_possession_or_contact") is True
        ):
            box_touches.append(r)

    long_attempts = []
    for r in evaluable_passes:
        attrs = _attrs(r)
        if attrs.get("is_long_pass") is True or (
            attrs.get("pass_distance_m") is not None and float(attrs["pass_distance_m"]) >= long_thr
        ):
            long_attempts.append(r)
    long_completed = [r for r in long_attempts if str(r.get("outcome")) == "completed"]

    metrics: dict[str, Any] = {}
    metrics["pass_attempts"] = count_metric(
        "pass_attempts", len(evaluable_passes), [r["ledger_event_id"] for r in evaluable_passes]
    )
    metrics["pass_completion_rate"] = rate_metric(
        "pass_completion_rate",
        len(completed),
        len(evaluable_passes),
        min_obs=min_passes,
        sources=[r["ledger_event_id"] for r in evaluable_passes],
    )
    metrics["receptions"] = count_metric(
        "receptions", len(receptions), [r["ledger_event_id"] for r in receptions]
    )

    if directional_ok:
        if attack_dir == "toward_goal_b":
            first, mid, final = "goal_a", "middle", "goal_b"
        else:
            first, mid, final = "goal_b", "middle", "goal_a"
        p12 = p23 = 0
        src12: list[str] = []
        src23: list[str] = []
        for r in evaluable_passes:
            attrs = _attrs(r)
            sz = str(attrs.get("start_zone_neutral") or r.get("start_zone_neutral") or "")
            ez = str(attrs.get("end_zone_neutral") or r.get("end_zone_neutral") or "")
            if sz == first and ez == mid and str(r.get("outcome")) == "completed":
                p12 += 1
                src12.append(r["ledger_event_id"])
            if sz == mid and ez == final and str(r.get("outcome")) == "completed":
                p23 += 1
                src23.append(r["ledger_event_id"])
        metrics["progressive_passes_def_to_mid"] = count_metric(
            "progressive_passes_def_to_mid", p12, src12
        )
        metrics["progressive_passes_mid_to_att"] = count_metric(
            "progressive_passes_mid_to_att", p23, src23
        )
    else:
        metrics["progressive_passes_def_to_mid"] = _metric(
            metric_id="progressive_passes_def_to_mid",
            value=None,
            status="not_evaluable",
            coverage=coverage,
            reason="attack_direction_unknown",
            warnings=["attack_direction_unknown"],
        )
        metrics["progressive_passes_mid_to_att"] = _metric(
            metric_id="progressive_passes_mid_to_att",
            value=None,
            status="not_evaluable",
            coverage=coverage,
            reason="attack_direction_unknown",
            warnings=["attack_direction_unknown"],
        )

    metrics["long_pass_attempts"] = count_metric(
        "long_pass_attempts", len(long_attempts), [r["ledger_event_id"] for r in long_attempts]
    )
    metrics["long_pass_completion_rate"] = rate_metric(
        "long_pass_completion_rate",
        len(long_completed),
        len(long_attempts),
        min_obs=min_long,
        sources=[r["ledger_event_id"] for r in long_attempts],
    )
    metrics["dribbles_successful"] = count_metric(
        "dribbles_successful", len(take_ok), [r["ledger_event_id"] for r in take_ok]
    )
    metrics["dribbles_failed"] = count_metric(
        "dribbles_failed", len(take_fail), [r["ledger_event_id"] for r in take_fail]
    )
    metrics["take_on_success_rate"] = rate_metric(
        "take_on_success_rate",
        len(take_ok),
        len(take_eval),
        min_obs=min_take,
        sources=[r["ledger_event_id"] for r in take_eval],
    )
    metrics["duels_won"] = count_metric(
        "duels_won", len(ground_won), [r["ledger_event_id"] for r in ground_won]
    )
    metrics["duel_win_rate"] = rate_metric(
        "duel_win_rate",
        len(ground_won),
        len(ground_eval),
        min_obs=min_duels,
        sources=[r["ledger_event_id"] for r in ground_eval],
    )
    metrics["aerial_duels"] = count_metric(
        "aerial_duels", len(aerial), [r["ledger_event_id"] for r in aerial]
    )
    metrics["aerial_win_rate"] = rate_metric(
        "aerial_win_rate",
        len(aerial_won),
        len(aerial_eval),
        min_obs=min_aerial,
        sources=[r["ledger_event_id"] for r in aerial_eval],
    )
    tir_n = len(tackles) + len(recoveries)
    metrics["tackles_interceptions_recoveries"] = _metric(
        metric_id="tackles_interceptions_recoveries",
        value=None if low_cov else tir_n,
        status="not_evaluable" if low_cov else "provisional",
        numerator=None if low_cov else {"tackles": len(tackles), "recoveries": len(recoveries)},
        denominator=None,
        coverage=coverage,
        confidence=None if low_cov else 0.7,
        reason="insufficient_coverage" if low_cov else None,
        sources=[r["ledger_event_id"] for r in list(tackles) + list(recoveries)],
        warnings=[],
    )
    metrics["ball_losses"] = count_metric(
        "ball_losses", len(turnovers), [r["ledger_event_id"] for r in turnovers]
    )
    metrics["clearances"] = count_metric(
        "clearances", len(clear_ok), [r["ledger_event_id"] for r in clear_ok]
    )
    metrics["penalty_area_ball_touches"] = count_metric(
        "penalty_area_ball_touches", len(box_touches), [r["ledger_event_id"] for r in box_touches]
    )
    metrics["interaction_coverage"] = _metric(
        metric_id="interaction_coverage",
        value=coverage,
        status="provisional" if coverage is not None else "not_evaluable",
        numerator=coverage,
        denominator=1.0,
        coverage=coverage,
        confidence=1.0 if coverage is not None else None,
        reason=None if coverage is not None else "coverage_missing",
        sources=[],
        provenance="project_generated",
    )

    # Ensure all requested metrics present
    requested = list(config.get("requested_metrics") or metrics.keys())
    for mid in requested:
        metrics.setdefault(
            mid,
            _metric(
                metric_id=mid,
                value=None,
                status="not_evaluable",
                coverage=coverage,
                reason="not_produced",
            ),
        )

    bundle = {
        "schema_version": 1,
        "dictionary_version": int(config.get("dictionary_version", 1)),
        "run_id": run_id,
        "video_id": video_id,
        "target_scope": "single_target_player",
        "attack_direction": attack_dir,
        "attack_direction_evaluable": directional_ok,
        "policy_fingerprint": policy_fp,
        "metrics": metrics,
        "created_at_utc": _utc_now(),
    }
    return bundle, attack


def compute_target_event_metrics(
    *,
    output_dir: Path,
    ledger: Sequence[Mapping[str, Any]],
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | None = None,
    attack_direction_manual: str | None = None,
    interaction_coverage: float | None = 0.8,
) -> MetricsServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_metrics_config(config_path, project_root=root)
    except MetricsConfigError as exc:
        raise MetricsServiceError(str(exc)) from exc
    config_fp = metrics_config_fingerprint(config)
    policy = load_events_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)
    run_id = run_id or generate_run_id()
    video_id = video_id or "synthetic_video_13d"

    metrics_bundle, attack = aggregate_target_event_metrics(
        ledger=ledger,
        config=config,
        policy_fp=policy_fp,
        run_id=run_id,
        video_id=video_id,
        attack_direction_manual=attack_direction_manual,
        interaction_coverage=interaction_coverage,
    )
    metrics_path = output_dir / "target_event_metrics.json"
    attack_path = output_dir / "attack_direction_evidence.json"
    write_json_record(metrics_path, metrics_bundle, overwrite=True)
    write_json_record(attack_path, attack, overwrite=True)
    evaluation = evaluate_events(ledger_rows=ledger)
    eval_path = output_dir / "evaluation.json"
    write_json_record(eval_path, evaluation, overwrite=True)
    summary = {
        "run_id": run_id,
        "video_id": video_id,
        "metric_count": len(metrics_bundle["metrics"]),
        "evaluable_count": sum(
            1 for m in metrics_bundle["metrics"].values() if m.get("status") != "not_evaluable"
        ),
        "not_evaluable_count": sum(
            1 for m in metrics_bundle["metrics"].values() if m.get("status") == "not_evaluable"
        ),
        "attack_direction": attack.get("attack_direction"),
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "evaluation_status": evaluation["evaluation_status"],
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "created_at_utc": _utc_now(),
        "NOT_EVALUATED": NOT_EVALUATED,
    }
    summary_path = output_dir / "summary.json"
    write_json_record(summary_path, summary, overwrite=True)
    return MetricsServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        metrics_json=str(metrics_path),
        attack_direction_json=str(attack_path),
        summary_json=str(summary_path),
        evaluation_json=str(eval_path),
        summary=summary,
        metrics=metrics_bundle,
    )


__all__ = [
    "MetricsServiceError",
    "MetricsServiceResult",
    "aggregate_target_event_metrics",
    "compute_target_event_metrics",
]
