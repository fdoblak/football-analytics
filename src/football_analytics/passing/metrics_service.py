"""Stage 11C coverage-aware target passing metrics service."""

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
from football_analytics.passing.attack_direction import (
    attack_relative_evaluable,
    resolve_attack_direction,
)
from football_analytics.passing.evaluation import NOT_EVALUATED_PASSING, evaluate_passing
from football_analytics.passing.metrics_config import (
    MetricsConfigError,
    load_metrics_config,
    metrics_config_fingerprint,
)
from football_analytics.passing.policy import load_passing_policy, policy_fingerprint
from football_analytics.passing.semantics import (
    box_touch_eligible,
    neutral_transition,
    neutral_zone_from_x,
)
from football_analytics.passing.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN


class MetricsServiceError(RuntimeError):
    """Passing metrics baseline service failure."""


@dataclass
class MetricsServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    metrics_json: str | None
    progression_parquet: str | None
    touches_parquet: str | None
    attack_direction_json: str | None
    summary_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    progression: Sequence[Mapping[str, Any]]
    touches: Sequence[Mapping[str, Any]]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "metrics_json": self.metrics_json,
            "progression_parquet": self.progression_parquet,
            "touches_parquet": self.touches_parquet,
            "attack_direction_json": self.attack_direction_json,
            "summary_json": self.summary_json,
            "evaluation_json": self.evaluation_json,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def _metric_value(
    value: Any, *, status: str, coverage: float | None, reason: str | None = None
) -> dict[str, Any]:
    return {
        "value": value,
        "status": status,
        "coverage": coverage,
        "reason": reason,
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
    }


def compute_target_passing_metrics(
    *,
    passes: Sequence[Mapping[str, Any]],
    receptions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    touch_inputs: Sequence[Mapping[str, Any]] | None = None,
    config: Mapping[str, Any],
    policy_fp: str,
    run_id: str,
    video_id: str,
    attack_direction_config: str | None = None,
    attack_direction_manual: str | None = None,
    joint_coverage_ratio: float | None = 0.8,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    thr = dict(config.get("thresholds") or {})
    long_thr = float(thr.get("long_pass_distance_m", 30.0))
    min_passes = int(thr.get("min_evaluable_passes", 1))
    min_cov = float(thr.get("min_joint_coverage_ratio", 0.1))

    attack_ev = resolve_attack_direction(
        run_id=run_id,
        video_id=video_id,
        config_direction=attack_direction_config
        or str((config.get("attack_direction") or {}).get("default", "unknown")),
        manual_direction=attack_direction_manual,
    )
    directional_ok = attack_relative_evaluable(attack_ev)
    attack_dir = str(attack_ev["attack_direction"])

    target_outcomes = [o for o in outcomes if o.get("passer_is_target") is True]
    evaluable = [
        o
        for o in target_outcomes
        if str(o.get("outcome")) in {"completed", "incomplete"}
        and str(o.get("outcome_state")) in {"candidate", "provisional", "confirmed"}
    ]
    completed = [o for o in evaluable if str(o.get("outcome")) == "completed"]
    incomplete = [o for o in evaluable if str(o.get("outcome")) == "incomplete"]
    attempts = list(evaluable)

    coverage = joint_coverage_ratio
    low_cov = coverage is None or float(coverage) < min_cov

    def count_or_ne(n: int) -> dict[str, Any]:
        if low_cov:
            return _metric_value(
                None, status="not_evaluable", coverage=coverage, reason="low_coverage"
            )
        return _metric_value(n, status="provisional", coverage=coverage)

    accuracy: dict[str, Any]
    if low_cov or len(evaluable) < min_passes:
        accuracy = _metric_value(
            None,
            status="not_evaluable",
            coverage=coverage,
            reason="insufficient_evaluable_passes" if not low_cov else "low_coverage",
        )
    else:
        accuracy = _metric_value(
            float(len(completed)) / float(len(evaluable)),
            status="provisional",
            coverage=coverage,
        )

    target_receptions = [
        r for r in receptions if str(r.get("target_relationship")) == "confirmed_target"
    ]

    # Neutral Goal A/B/Middle transitions
    transitions: dict[str, int] = {}
    for o in attempts:
        key = neutral_transition(str(o.get("start_zone_neutral")), str(o.get("end_zone_neutral")))
        transitions[key] = transitions.get(key, 0) + 1

    long_attempts = [
        o
        for o in attempts
        if o.get("is_long_pass") is True
        or (o.get("pass_distance_m") is not None and float(o["pass_distance_m"]) >= long_thr)
    ]
    long_completed = [o for o in long_attempts if str(o.get("outcome")) == "completed"]
    if low_cov:
        long_ratio = _metric_value(
            None, status="not_evaluable", coverage=coverage, reason="low_coverage"
        )
    elif not attempts:
        long_ratio = _metric_value(
            None, status="not_evaluable", coverage=coverage, reason="no_attempts"
        )
    else:
        long_ratio = _metric_value(
            float(len(long_attempts)) / float(len(attempts)),
            status="provisional",
            coverage=coverage,
        )

    # Progression 1→2 / 2→3 only if attack direction resolved
    if directional_ok:
        # Map attack toward_goal_b: goal_a=1st, middle=2nd, goal_b=3rd
        if attack_dir == "toward_goal_b":
            first, mid, final = "goal_a", "middle", "goal_b"
        else:
            first, mid, final = "goal_b", "middle", "goal_a"
        p12 = sum(
            1
            for o in attempts
            if str(o.get("start_zone_neutral")) == first and str(o.get("end_zone_neutral")) == mid
        )
        p23 = sum(
            1
            for o in attempts
            if str(o.get("start_zone_neutral")) == mid and str(o.get("end_zone_neutral")) == final
        )
        prog_12 = count_or_ne(p12)
        prog_23 = count_or_ne(p23)
        prog_flag_12 = "true" if p12 else "false"
        prog_flag_23 = "true" if p23 else "false"
    else:
        prog_12 = _metric_value(
            None, status="not_evaluable", coverage=coverage, reason="attack_direction_unknown"
        )
        prog_23 = _metric_value(
            None, status="not_evaluable", coverage=coverage, reason="attack_direction_unknown"
        )
        prog_flag_12 = "not_evaluable"
        prog_flag_23 = "not_evaluable"

    progression_rows: list[dict[str, Any]] = []
    for i, o in enumerate(attempts, start=1):
        sz = str(o.get("start_zone_neutral"))
        ez = str(o.get("end_zone_neutral"))
        progression_rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "segment_id": f"seg_{i:02d}",
                "pass_candidate_id": o.get("pass_candidate_id"),
                "outcome_id": o.get("outcome_id"),
                "start_time_us": (
                    int(o.get("start_time_us", i * 1_000_000))
                    if "start_time_us" in o
                    else i * 1_000_000
                ),
                "end_time_us": (
                    int(o.get("end_time_us", i * 1_000_000 + 500_000))
                    if "end_time_us" in o
                    else i * 1_000_000 + 500_000
                ),
                "start_x_m": None,
                "start_y_m": None,
                "end_x_m": None,
                "end_y_m": None,
                "start_zone_neutral": sz,
                "end_zone_neutral": ez,
                "neutral_transition": neutral_transition(sz, ez),
                "attack_direction": attack_dir,
                "progression_1_to_2": prog_flag_12 if directional_ok else "not_evaluable",
                "progression_2_to_3": prog_flag_23 if directional_ok else "not_evaluable",
                "segment_state": "provisional",
                "target_relationship": "confirmed_target",
                "cut_or_replay": bool(o.get("cut_or_replay")),
                "hard_gap": bool(o.get("hard_gap")),
                "automatic_ceiling": "provisional",
                "review_status": "unreviewed",
                "manual_review_required": False,
                "evidence_refs": list(o.get("evidence_refs") or []),
                "reason_codes": [] if directional_ok else ["ATTACK_DIRECTION_UNKNOWN"],
                "quality_flags": [],
                "metric_origin": METRIC_ORIGIN,
                "definition_style": DEFINITION_STYLE,
                "policy_fingerprint": policy_fp,
                "provenance_json": None,
                "contract_version": CONTRACT_VERSION,
            }
        )

    # Fix times from pass rows when available
    pass_by_id = {p["pass_candidate_id"]: p for p in passes}
    for row in progression_rows:
        src = pass_by_id.get(str(row.get("pass_candidate_id")))
        if src:
            row["start_time_us"] = int(src["start_time_us"])
            row["end_time_us"] = int(src["end_time_us"])
            row["start_x_m"] = src.get("start_x_m")
            row["start_y_m"] = src.get("start_y_m")
            row["end_x_m"] = src.get("end_x_m")
            row["end_y_m"] = src.get("end_y_m")

    touch_rows: list[dict[str, Any]] = []
    for i, t in enumerate(touch_inputs or [], start=1):
        in_pen = bool(t.get("in_penalty_area"))
        has_pc = bool(t.get("has_possession_or_contact"))
        has_map = bool(t.get("has_pitch_mapping", True))
        play = str(t.get("playability_status", "playable"))
        eligible = box_touch_eligible(
            in_penalty=in_pen,
            has_possession_or_contact=has_pc,
            has_pitch_mapping=has_map,
            playability_status=play,
        )
        presence_alone = in_pen and not has_pc
        touch_rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "touch_id": f"touch_{i:02d}",
                "human_track_id": int(t.get("human_track_id", 1)),
                "touch_time_us": int(t.get("touch_time_us", i * 3_000_000)),
                "touch_x_m": t.get("touch_x_m"),
                "touch_y_m": t.get("touch_y_m"),
                "in_penalty_area": in_pen,
                "penalty_side_neutral": str(
                    t.get("penalty_side_neutral")
                    or (neutral_zone_from_x(x_m=t.get("touch_x_m")) if in_pen else "none")
                ),
                "is_box_touch_candidate": eligible,
                "penalty_presence_alone": presence_alone,
                "has_possession_or_contact": has_pc,
                "has_pitch_mapping": has_map,
                "playability_status": play,
                "calibration_status": str(t.get("calibration_status", "valid")),
                "touch_state": "provisional" if eligible else "rejected",
                "target_relationship": "confirmed_target",
                "possession_hypothesis_id": t.get("possession_hypothesis_id"),
                "contact_candidate_ids": list(t.get("contact_candidate_ids") or []),
                "evidence_refs": list(t.get("evidence_refs") or []),
                "automatic_ceiling": "provisional",
                "review_status": "unreviewed",
                "manual_review_required": False,
                "uncertainty": 0.3,
                "reason_codes": [] if eligible else ["PENALTY_PRESENCE_NOT_BOX_TOUCH"],
                "quality_flags": [],
                "metric_origin": METRIC_ORIGIN,
                "definition_style": DEFINITION_STYLE,
                "policy_fingerprint": policy_fp,
                "provenance_json": None,
                "contract_version": CONTRACT_VERSION,
            }
        )

    box_candidates = [t for t in touch_rows if t.get("is_box_touch_candidate") is True]

    metrics = {
        "pass_attempts": count_or_ne(len(attempts)),
        "pass_completed": count_or_ne(len(completed)),
        "pass_incomplete": count_or_ne(len(incomplete)),
        "pass_accuracy": accuracy,
        "receptions": count_or_ne(len(target_receptions)),
        "neutral_zone_transitions": _metric_value(
            dict(transitions),
            status="not_evaluable" if low_cov else "provisional",
            coverage=coverage,
            reason="low_coverage" if low_cov else None,
        ),
        "long_pass_attempts": count_or_ne(len(long_attempts)),
        "long_pass_completions": count_or_ne(len(long_completed)),
        "long_pass_ratio": long_ratio,
        "box_contact_candidates": count_or_ne(len(box_candidates)),
        "progression_1_to_2": prog_12,
        "progression_2_to_3": prog_23,
        "attack_direction": attack_dir,
        "attack_relative_evaluable": directional_ok,
        "coverage": coverage,
        "evaluation_status": NOT_EVALUATED_PASSING,
        "opta_accuracy_validated": False,
        "real_football_accuracy_validated": False,
    }
    return metrics, progression_rows, touch_rows, attack_ev


def compute_passing_metrics(
    *,
    output_dir: Path,
    passes: Sequence[Mapping[str, Any]],
    receptions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    touch_inputs: Sequence[Mapping[str, Any]] | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    attack_direction_manual: str | None = None,
    joint_coverage_ratio: float | None = 0.8,
    project_root: Path | None = None,
) -> MetricsServiceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or default_project_root()
    try:
        config = load_metrics_config(config_path, project_root=root)
    except MetricsConfigError as exc:
        raise MetricsServiceError(str(exc)) from exc
    config_fp = metrics_config_fingerprint(config)
    policy = load_passing_policy(project_root=root)
    policy_fp = policy_fingerprint(policy)
    rid = run_id or generate_run_id()
    vid = video_id or "video_synth_01"

    metrics, progression, touches, attack_ev = compute_target_passing_metrics(
        passes=passes,
        receptions=receptions,
        outcomes=outcomes,
        touch_inputs=touch_inputs,
        config=config,
        policy_fp=policy_fp,
        run_id=rid,
        video_id=vid,
        attack_direction_manual=attack_direction_manual,
        joint_coverage_ratio=joint_coverage_ratio,
    )

    prog_path = output_dir / "ball_progression_segments.parquet"
    touch_path = output_dir / "target_ball_touches.parquet"
    write_contract_parquet(
        _cast("ball_progression_segments", progression),
        prog_path,
        get_contract("ball_progression_segments", 1),
        overwrite=False,
    )
    write_contract_parquet(
        _cast("target_ball_touches", touches),
        touch_path,
        get_contract("target_ball_touches", 1),
        overwrite=False,
    )

    metrics_path = output_dir / "passing_metrics.json"
    attack_path = output_dir / "attack_direction_evidence.json"
    eval_path = output_dir / "passing_evaluation.json"
    summary_path = output_dir / "passing_metrics_summary.json"
    write_json_record(
        metrics_path,
        {"schema_version": 1, "run_id": rid, "video_id": vid, "metrics": metrics},
        overwrite=False,
    )
    write_json_record(attack_path, attack_ev, overwrite=False)
    evaluation = evaluate_passing(passes=passes, receptions=receptions, outcomes=outcomes).to_dict(
        run_id=rid, video_id=vid, config_fingerprint=config_fp
    )
    write_json_record(eval_path, evaluation, overwrite=False)
    summary = {
        "schema_version": 1,
        "stage": "11C",
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": config_fp,
        "policy_fingerprint": policy_fp,
        "progression_segment_count": len(progression),
        "target_ball_touch_count": len(touches),
        "attack_direction": attack_ev.get("attack_direction"),
        "evaluation_status": NOT_EVALUATED_PASSING,
        "opta_accuracy_validated": False,
        "real_football_accuracy_validated": False,
        "artifact_hashes": {
            "progression": sha256_file(prog_path),
            "touches": sha256_file(touch_path),
        },
        "created_at_utc": _utc_now(),
    }
    write_json_record(summary_path, summary, overwrite=False)

    return MetricsServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=config_fp,
        metrics_json=str(metrics_path),
        progression_parquet=str(prog_path),
        touches_parquet=str(touch_path),
        attack_direction_json=str(attack_path),
        summary_json=str(summary_path),
        evaluation_json=str(eval_path),
        summary=summary,
        metrics=metrics,
        progression=progression,
        touches=touches,
    )


__all__ = [
    "MetricsServiceError",
    "MetricsServiceResult",
    "compute_target_passing_metrics",
    "compute_passing_metrics",
]
