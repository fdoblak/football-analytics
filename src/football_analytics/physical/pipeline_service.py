"""Stage 9E physical metric fusion service."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.data.compiler import get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.physical.pipeline_config import (
    PipelineConfigError,
    load_pipeline_config,
    pipeline_config_fingerprint,
)
from football_analytics.physical.pipeline_evaluation import (
    NOT_EVALUATED_PIPELINE,
    evaluate_physical_pipeline,
)
from football_analytics.physical.pipeline_integrity import (
    PipelineIntegrityError,
    assert_confirmed_identity,
    assert_finite_non_negative,
    assert_receipt_fresh,
    assert_same_target_scope,
    check_distance_recount,
    check_duration_mass_consistency,
)
from football_analytics.physical.pipeline_quality import (
    derive_overall_status,
    metric_entry,
    normalize_status,
)


class PipelineServiceError(RuntimeError):
    """Physical metric pipeline failure."""


@dataclass
class PipelineServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    summary_json: str | None
    quality_json: str | None
    receipt_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "summary_json": self.summary_json,
            "quality_json": self.quality_json,
            "receipt_json": self.receipt_json,
            "evaluation_json": self.evaluation_json,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise PipelineServiceError(f"missing or symlink input: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PipelineServiceError(f"JSON root must be object: {p}")
    return data


def integrate_physical_metrics(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    trajectory_summary: Mapping[str, Any] | None = None,
    trajectory_receipt: Mapping[str, Any] | None = None,
    motion_summary: Mapping[str, Any] | None = None,
    motion_receipt: Mapping[str, Any] | None = None,
    spatial_summary: Mapping[str, Any] | None = None,
    spatial_receipt: Mapping[str, Any] | None = None,
    heatmap_ref: Mapping[str, Any] | None = None,
    zone_ref: Mapping[str, Any] | None = None,
    activity_ref: Mapping[str, Any] | None = None,
    recounted_distance_m: float | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    analysis_start_us: int | None = None,
    analysis_end_us: int | None = None,
    force_source_inconsistent: Sequence[str] | None = None,
    cleanup_on_failure: bool = True,
) -> PipelineServiceResult:
    """Fuse Stage 9B–9D summaries into one target physical analysis package."""
    cfg = config or load_pipeline_config(config_path)
    cfg_fp = pipeline_config_fingerprint(cfg)
    out = Path(output_dir)
    created = False
    findings: list[str] = []
    warnings: list[str] = []
    try:
        if out.exists() and any(out.iterdir()) and cfg.get("overwrite_allowed") is not True:
            raise PipelineServiceError("output_dir not empty and overwrite forbidden")
        out.mkdir(parents=True, exist_ok=True)
        created = True

        # Identity gate
        id_err = assert_confirmed_identity(
            identity,
            require_confirmed=bool(cfg["require_confirmed_identity"]),
            forbid_revoked=bool(cfg["forbid_revoked_identity"]),
        )
        identity_ok = id_err is None

        # Normalize optional summaries to identity scope when ids omitted
        def _with_ids(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
            if block is None:
                return None
            out = dict(block)
            out.setdefault("run_id", identity.get("run_id"))
            out.setdefault("video_id", identity.get("video_id"))
            out.setdefault(
                "target_player_id",
                identity.get("target_player_id") or identity.get("target_id"),
            )
            return out

        trajectory_summary = _with_ids(trajectory_summary)
        trajectory_receipt = _with_ids(trajectory_receipt)
        motion_summary = _with_ids(motion_summary)
        motion_receipt = _with_ids(motion_receipt)
        spatial_summary = _with_ids(spatial_summary)
        spatial_receipt = _with_ids(spatial_receipt)

        # Scope integrity across available blocks
        blocks = [
            identity,
            trajectory_receipt,
            motion_receipt,
            spatial_receipt,
            trajectory_summary,
            motion_summary,
            spatial_summary,
        ]
        try:
            run_id, video_id, target_id = assert_same_target_scope(
                *[b for b in blocks if b],
                require=bool(cfg["integrity"]["fail_on_source_mix"]),
            )
        except PipelineIntegrityError as exc:
            raise PipelineServiceError(str(exc)) from exc

        # Explicit mix: summary target differs from identity after normalization skip —
        # re-check raw provided blocks that already had target ids
        for label, raw in (
            ("motion_summary", motion_summary),
            ("spatial_summary", spatial_summary),
            ("trajectory_summary", trajectory_summary),
        ):
            if (
                raw
                and raw.get("target_player_id")
                and str(raw["target_player_id"])
                != str(identity.get("target_player_id") or identity.get("target_id"))
            ):
                raise PipelineServiceError(f"SOURCE_MIX run/video/target mismatch on {label}")

        # Receipt freshness
        for label, receipt in (
            ("trajectory", trajectory_receipt),
            ("motion", motion_receipt),
            ("spatial", spatial_receipt),
        ):
            if receipt is None:
                continue
            try:
                assert_receipt_fresh(receipt)
            except PipelineIntegrityError as exc:
                if cfg["integrity"]["fail_on_stale_receipt"]:
                    raise PipelineServiceError(f"{label}:{exc}") from exc
                findings.append(f"{label}:{exc}")

        # Numeric sanity
        for label, val in (
            ("eligible_duration", (trajectory_summary or {}).get("eligible_duration_us")),
            ("measured_distance", (motion_summary or {}).get("measured_distance_m")),
            ("dwell", (spatial_summary or {}).get("total_dwell_seconds")),
        ):
            try:
                assert_finite_non_negative(float(val) if val is not None else None, label=label)
            except PipelineIntegrityError as exc:
                raise PipelineServiceError(str(exc)) from exc
            except (TypeError, ValueError):
                pass

        inconsistent: list[str] = list(force_source_inconsistent or [])
        # Distance recount
        dist_code = check_distance_recount(
            reported_m=(motion_summary or {}).get("measured_distance_m"),
            recounted_m=recounted_distance_m,
        )
        if dist_code:
            inconsistent.append(dist_code)

        # Heatmap / activity mass vs eligible
        elig = int(
            (motion_summary or {}).get("measured_eligible_duration_us")
            or (spatial_summary or {}).get("eligible_observed_duration_us")
            or (trajectory_summary or {}).get("eligible_duration_us")
            or 0
        )
        if activity_ref and activity_ref.get("classes"):
            comps = [int(c.get("duration_us") or 0) for c in activity_ref["classes"]]
            code = check_duration_mass_consistency(
                eligible_us=elig, component_us=comps, label="activity"
            )
            if code:
                inconsistent.append(code)
        if heatmap_ref and heatmap_ref.get("total_dwell_seconds") is not None and elig > 0:
            dwell_us = int(float(heatmap_ref["total_dwell_seconds"]) * 1_000_000)
            if dwell_us > elig + 2 and elig > 0:
                # dwell can be less than eligible; exceeding is inconsistency
                inconsistent.append("source_inconsistent:heatmap_mass_exceeds_eligible")

        # Build per-metric entries
        metrics: list[dict[str, Any]] = []

        def _push(**kwargs: Any) -> None:
            metrics.append(metric_entry(**kwargs))

        if not identity_ok:
            st = id_err or "identity_unconfirmed"
            for name, unit in (
                ("trajectory_coverage", "ratio"),
                ("measured_distance", "m"),
                ("robust_mean_speed", "m_s"),
                ("robust_peak_speed", "m_s"),
                ("sprint_count", "count"),
                ("heatmap_dwell", "s"),
                ("zone_occupancy", "s"),
                ("activity_distribution", "ratio"),
                ("movement_activity_index", "ratio"),
            ):
                _push(
                    name=name,
                    value=None,
                    unit=unit,
                    status=st,
                    reason_codes=[st],
                    provenance={"stage": "9E", "identity_gate": True},
                )
        else:
            traj = trajectory_summary or {}
            mot = motion_summary or {}
            spat = spatial_summary or {}

            cov = mot.get("coverage_ratio_distance")
            if cov is None:
                cov = spat.get("coverage_ratio")
            traj_status = "evaluable" if traj or mot or spat else "not_observed"
            if cov is not None and float(cov) < float(
                cfg["coverage"]["min_coverage_ratio_for_overall_evaluable"]
            ):
                traj_status = "insufficient_coverage"
            _push(
                name="trajectory_coverage",
                value=float(cov) if cov is not None else None,
                unit="ratio",
                status=traj_status,
                coverage_ratio=float(cov) if cov is not None else None,
                reason_codes=[],
                provenance={"sources": ["9B", "9C", "9D"]},
            )

            dist_status = normalize_status(str(mot.get("distance_status") or "not_evaluable"))
            if any("distance" in c for c in inconsistent):
                dist_status = "source_inconsistent"
            _push(
                name="measured_distance",
                value=mot.get("measured_distance_m"),
                unit="m",
                status=dist_status,
                coverage_ratio=mot.get("coverage_ratio_distance"),
                reason_codes=list(mot.get("not_evaluable_reasons") or []),
                provenance={
                    "semantics": "measured_eligible_not_full_match",
                    "layer": cfg["primary_sample_layer"],
                },
            )

            speed_status = normalize_status(str(mot.get("speed_status") or "not_evaluable"))
            _push(
                name="robust_mean_speed",
                value=mot.get("robust_mean_speed_mps"),
                unit="m_s",
                status=speed_status,
                coverage_ratio=mot.get("coverage_ratio_speed"),
                provenance={"vs": "diagnostic_raw_speed"},
            )
            peak = mot.get("robust_peak_speed_mps")
            _push(
                name="robust_peak_speed",
                value=peak,
                unit="m_s",
                status="evaluable" if peak is not None else "not_evaluable",
                coverage_ratio=mot.get("coverage_ratio_speed"),
                provenance={"min_support_required": True},
            )

            sprint_count = mot.get("sprint_count")
            _push(
                name="sprint_count",
                value=float(sprint_count) if sprint_count is not None else None,
                unit="count",
                status="evaluable" if sprint_count is not None else "not_evaluable",
                provenance={
                    "metric_origin": "project_generated",
                    "not_official_opta": True,
                },
            )
            _push(
                name="sprint_distance",
                value=mot.get("sprint_distance_m"),
                unit="m",
                status="evaluable" if mot.get("sprint_distance_m") is not None else "not_evaluable",
                provenance={"metric_origin": "project_generated"},
            )
            _push(
                name="sprint_duration",
                value=(
                    float(mot["sprint_duration_us"]) / 1_000_000.0
                    if mot.get("sprint_duration_us") is not None
                    else None
                ),
                unit="s",
                status=(
                    "evaluable" if mot.get("sprint_duration_us") is not None else "not_evaluable"
                ),
                provenance={"metric_origin": "project_generated"},
            )

            hm_status = normalize_status(str(spat.get("heatmap_status") or "not_evaluable"))
            if any("heatmap" in c for c in inconsistent):
                hm_status = "source_inconsistent"
            _push(
                name="heatmap_dwell",
                value=spat.get("total_dwell_seconds"),
                unit="s",
                status=hm_status,
                coverage_ratio=spat.get("coverage_ratio"),
                provenance={
                    "weighting": "time_weighted",
                    "ref_fingerprint": (heatmap_ref or {}).get("config_fingerprint"),
                },
            )

            zone_status = normalize_status(str(spat.get("zone_status") or "not_evaluable"))
            zone_dwell = None
            if zone_ref and zone_ref.get("zones"):
                zone_dwell = sum(float(z.get("dwell_seconds") or 0) for z in zone_ref["zones"])
            _push(
                name="zone_occupancy",
                value=zone_dwell,
                unit="s",
                status=zone_status,
                provenance={
                    "attack_direction": "unknown",
                    "penalty_is_not_touch": True,
                    "ref": bool(zone_ref),
                },
            )

            act_status = normalize_status(str(spat.get("activity_status") or "not_evaluable"))
            if any("activity" in c for c in inconsistent):
                act_status = "source_inconsistent"
            _push(
                name="activity_distribution",
                value=(activity_ref or {}).get("moving_to_eligible_ratio"),
                unit="ratio",
                status=act_status,
                provenance={"missing_coverage_counted_as_inactive": False},
            )
            mai = spat.get("movement_activity_index")
            mai_st = normalize_status(
                str(spat.get("movement_activity_index_status") or "not_evaluable")
            )
            _push(
                name="movement_activity_index",
                value=mai,
                unit="ratio",
                status=mai_st,
                provenance={
                    "metric_origin": "project_generated",
                    "not_official_opta": True,
                    "not_possession_or_tactical": True,
                },
            )

            if inconsistent:
                findings.extend(inconsistent)
                for m in metrics:
                    for code in inconsistent:
                        key = code.split(":")[-1].split("_")[0]
                        match = key in m["metric_name"] or (
                            "distance" in code and m["metric_name"] == "measured_distance"
                        )
                        if match and m["status"] == "evaluable":
                            m["status"] = "source_inconsistent"
                            m["reason_codes"] = list(set(m["reason_codes"] + [code]))

        overall = derive_overall_status(
            metrics,
            critical=list(cfg["overall_status_rules"]["critical_metrics"]),
            identity_ok=identity_ok,
        )

        t0 = analysis_start_us if analysis_start_us is not None else 0
        t1 = analysis_end_us if analysis_end_us is not None else max(t0, elig)
        summary = {
            "schema_version": 1,
            "document": "target_physical_metric_summary",
            "stage": "9E",
            "run_id": run_id,
            "video_id": video_id,
            "target_player_id": target_id,
            "identity_assignment_id": identity.get("identity_assignment_id"),
            "identity_status": ("confirmed" if identity_ok else (id_err or "identity_unconfirmed")),
            "assignment_revoked": identity.get("assignment_revoked") is True,
            "attack_direction": "unknown",
            "analysis_interval": {"start_us": int(t0), "end_us": int(t1)},
            "eligible_observed_duration_us": elig,
            "observed_coverage_us": (motion_summary or {}).get("observed_coverage_us")
            or (spatial_summary or {}).get("observed_coverage_us"),
            "derived_coverage_us": (motion_summary or {}).get("derived_coverage_us") or 0,
            "coverage_ratio": (motion_summary or {}).get("coverage_ratio_distance")
            or (spatial_summary or {}).get("coverage_ratio"),
            "measured_distance_m": next(
                (m["value"] for m in metrics if m["metric_name"] == "measured_distance"), None
            ),
            "robust_mean_speed_mps": next(
                (m["value"] for m in metrics if m["metric_name"] == "robust_mean_speed"), None
            ),
            "robust_peak_speed_mps": next(
                (m["value"] for m in metrics if m["metric_name"] == "robust_peak_speed"), None
            ),
            "sprint_count": next(
                (m["value"] for m in metrics if m["metric_name"] == "sprint_count"), None
            ),
            "sprint_distance_m": next(
                (m["value"] for m in metrics if m["metric_name"] == "sprint_distance"), None
            ),
            "sprint_duration_s": next(
                (m["value"] for m in metrics if m["metric_name"] == "sprint_duration"), None
            ),
            "heatmap_reference": {
                "present": heatmap_ref is not None,
                "fingerprint": (heatmap_ref or {}).get("config_fingerprint"),
                "total_dwell_seconds": (heatmap_ref or {}).get("total_dwell_seconds"),
            },
            "zone_dwell_entries": (zone_ref or {}).get("zones"),
            "activity_class_durations": (activity_ref or {}).get("classes"),
            "movement_activity_index": next(
                (m["value"] for m in metrics if m["metric_name"] == "movement_activity_index"),
                None,
            ),
            "metrics": metrics,
            "overall_physical_analysis_status": overall,
            "partial_is_not_full_match": True,
            "no_coverage_extrapolation": True,
            "missing_coverage_not_inactive": True,
            "penalty_occupancy_is_not_touch": True,
            "not_official_opta": True,
            "final_customer_visual_created": False,
            "warnings": warnings,
            "findings": findings,
            "config_fingerprint": cfg_fp,
            "source_fingerprints": {
                "trajectory_receipt": (trajectory_receipt or {}).get("config_fingerprint"),
                "motion_receipt": (motion_receipt or {}).get("config_fingerprint"),
                "spatial_receipt": (spatial_receipt or {}).get("config_fingerprint"),
            },
            "evaluation_status": NOT_EVALUATED_PIPELINE,
            "metric_origin": cfg["metric_origin"],
            "definition_style": cfg["definition_style"],
        }

        quality = {
            "schema_version": 1,
            "document": "target_physical_metric_quality",
            "stage": "9E",
            "created_at_utc": _utc_now(),
            "overall_status": overall,
            "identity_ok": identity_ok,
            "metric_status_counts": {
                s: sum(1 for m in metrics if m["status"] == s)
                for s in sorted({m["status"] for m in metrics})
            },
            "inconsistency_codes": inconsistent,
            "findings": findings,
            "warnings": warnings,
            "gates": {
                "confirmed_identity_required": True,
                "no_silent_repair": True,
                "zero_distinct_from_null": True,
                "no_full_match_extrapolation": True,
            },
            "config_fingerprint": cfg_fp,
        }

        ev = evaluate_physical_pipeline(has_reviewed_ground_truth=False)
        evaluation = ev.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=cfg_fp)

        sum_path = out / "target_physical_metric_summary.json"
        q_path = out / "target_physical_metric_quality.json"
        e_path = out / "target_physical_metric_evaluation.json"
        write_json_record(sum_path, summary, overwrite=False)
        write_json_record(q_path, quality, overwrite=False)
        write_json_record(e_path, evaluation, overwrite=False)

        receipt = {
            "schema_version": 1,
            "document": "target_physical_metric_receipt",
            "receipt_id": "physical_pipeline_receipt_01",
            "request_id": "physical_pipeline_req_01",
            "run_id": run_id,
            "video_id": video_id,
            "target_player_id": target_id,
            "status": "succeeded" if overall in {"succeeded", "partial"} else "failed",
            "overall_physical_analysis_status": overall,
            "config_fingerprint": cfg_fp,
            "input_fingerprints": {
                "physical_metric_results": contract_fingerprint(
                    get_contract("physical_metric_results", 1)
                ),
            },
            "artifact_hashes": {
                "summary": {"sha256": sha256_file(sum_path), "size": sum_path.stat().st_size},
                "quality": {"sha256": sha256_file(q_path), "size": q_path.stat().st_size},
                "evaluation": {"sha256": sha256_file(e_path), "size": e_path.stat().st_size},
            },
            "evaluation_status": NOT_EVALUATED_PIPELINE,
            "created_at_utc": _utc_now(),
            "provenance": {
                "stage": "9E",
                "label": "physical_metric_fusion",
                "fused_from": ["9B", "9C", "9D"],
                "not_official_opta": True,
                "final_visual": False,
            },
            "completion_status": (
                "succeeded" if overall in {"succeeded", "partial", "not_evaluable"} else "failed"
            ),
            "quality_validated": True,
        }
        r_path = out / "target_physical_metric_receipt.json"
        write_json_record(r_path, receipt, overwrite=False)

        # Success requires receipt + quality present
        if not r_path.is_file() or not q_path.is_file():
            raise PipelineServiceError("receipt_or_quality_missing")

        return PipelineServiceResult(
            accepted=True,
            exit_code=0 if overall != "failed" else 1,
            error_code=None if overall != "failed" else "OVERALL_FAILED",
            config_fingerprint=cfg_fp,
            summary_json=str(sum_path),
            quality_json=str(q_path),
            receipt_json=str(r_path),
            evaluation_json=str(e_path),
            summary={
                "overall_physical_analysis_status": overall,
                "identity_status": summary["identity_status"],
                "evaluation_status": NOT_EVALUATED_PIPELINE,
                "metric_count": len(metrics),
                "findings": findings,
                "final_customer_visual_created": False,
            },
        )
    except (
        PipelineServiceError,
        PipelineConfigError,
        PipelineIntegrityError,
        OSError,
        ValueError,
    ) as exc:
        if cleanup_on_failure and created and out.exists():
            shutil.rmtree(out, ignore_errors=True)
        return PipelineServiceResult(
            accepted=False,
            exit_code=1,
            error_code=type(exc).__name__,
            config_fingerprint=cfg_fp,
            summary_json=None,
            quality_json=None,
            receipt_json=None,
            evaluation_json=None,
            summary={"error": str(exc)},
        )


def integrate_from_paths(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    trajectory_summary_path: Path | None = None,
    trajectory_receipt_path: Path | None = None,
    motion_summary_path: Path | None = None,
    motion_receipt_path: Path | None = None,
    spatial_summary_path: Path | None = None,
    spatial_receipt_path: Path | None = None,
    heatmap_path: Path | None = None,
    zones_path: Path | None = None,
    activity_path: Path | None = None,
    config_path: Path | None = None,
    recounted_distance_m: float | None = None,
) -> PipelineServiceResult:
    return integrate_physical_metrics(
        output_dir=output_dir,
        identity=identity,
        trajectory_summary=_load_json(trajectory_summary_path),
        trajectory_receipt=_load_json(trajectory_receipt_path),
        motion_summary=_load_json(motion_summary_path),
        motion_receipt=_load_json(motion_receipt_path),
        spatial_summary=_load_json(spatial_summary_path),
        spatial_receipt=_load_json(spatial_receipt_path),
        heatmap_ref=_load_json(heatmap_path),
        zone_ref=_load_json(zones_path),
        activity_ref=_load_json(activity_path),
        recounted_distance_m=recounted_distance_m,
        config_path=config_path,
    )


__all__ = [
    "PipelineServiceError",
    "PipelineServiceResult",
    "integrate_physical_metrics",
    "integrate_from_paths",
]
