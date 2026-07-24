"""Physical / trajectory bundle validation (Stage 9A — contracts only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.physical.eligibility import assert_no_attack_relative
from football_analytics.physical.semantics import (
    assert_derived_provenance,
    assert_finite_pitch_point,
    assert_no_uncontrolled_duplicates,
    assert_strictly_increasing_times,
    distance_bridge_allowed,
    segment_metric_sufficient,
)
from football_analytics.physical.types import PhysicalContractError
from football_analytics.physical.zones import assert_zone_name_allowed


@dataclass
class PhysicalValidationResult:
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        self.status = "FAIL"

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _rows(table: Any | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_pylist"):
        return list(table.to_pylist())
    if isinstance(table, list):
        return [dict(r) for r in table]
    raise TypeError("expected pyarrow.Table or list of mappings")


def validate_physical_bundle(
    *,
    samples: Any | None = None,
    segments: Any | None = None,
    gaps: Any | None = None,
    metric_results: Any | None = None,
    policy: Mapping[str, Any] | None = None,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> PhysicalValidationResult:
    result = PhysicalValidationResult()
    sample_rows = _rows(samples)
    segment_rows = _rows(segments)
    gap_rows = _rows(gaps)
    metric_rows = _rows(metric_results)

    try:
        assert_no_uncontrolled_duplicates(sample_rows)
    except PhysicalContractError as exc:
        result.err(str(exc))

    by_target: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for s in sample_rows:
        key = (str(s["run_id"]), str(s["video_id"]), str(s["target_player_id"]))
        by_target.setdefault(key, []).append(s)
    for _key, rows in by_target.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["video_time_us"]))
        try:
            assert_strictly_increasing_times([int(r["video_time_us"]) for r in rows_sorted])
        except PhysicalContractError as exc:
            result.err(str(exc))
        for s in rows:
            try:
                assert_no_attack_relative(str(s["pitch_coordinate_frame_id"]))
                if str(s.get("eligibility_status")) == "eligible":
                    assert_finite_pitch_point(
                        float(s["pitch_x_m"]),
                        float(s["pitch_y_m"]),
                        length_m=pitch_length_m,
                        width_m=pitch_width_m,
                    )
                assert_derived_provenance(s)
            except PhysicalContractError as exc:
                result.err(str(exc))
            if str(s.get("sample_source")) == "raw_observed" and list(
                s.get("derived_from_sample_ids") or []
            ):
                result.warn("raw sample unexpectedly lists derived_from_sample_ids")

    for seg in segment_rows:
        if int(seg["end_time_us"]) < int(seg["start_time_us"]):
            result.err(f"segment interval inverted: {seg.get('trajectory_segment_id')}")
        if (
            int(seg.get("eligible_sample_count", 0)) < 2
            and str(seg.get("metric_eligibility")) == "eligible"
        ):
            result.err("SINGLE_SAMPLE_SEGMENT_INSUFFICIENT eligibility")
        if not segment_metric_sufficient(seg) and str(seg.get("metric_eligibility")) == "eligible":
            result.err(f"segment not metric-sufficient: {seg.get('trajectory_segment_id')}")

    # Overlapping confirmed segments for same target (hard conflict)
    conf_segs = [s for s in segment_rows if str(s.get("metric_eligibility")) == "eligible"]
    for i, a in enumerate(conf_segs):
        for b in conf_segs[i + 1 :]:
            if (
                a["run_id"] == b["run_id"]
                and a["video_id"] == b["video_id"]
                and a["target_player_id"] == b["target_player_id"]
                and int(a["start_time_us"]) < int(b["end_time_us"])
                and int(b["start_time_us"]) < int(a["end_time_us"])
            ):
                result.err(
                    "OVERLAPPING_CONFIRMED_TRAJECTORY_SEGMENT: "
                    f"{a['trajectory_segment_id']} vs {b['trajectory_segment_id']}"
                )

    for g in gap_rows:
        try:
            distance_bridge_allowed(g)
        except PhysicalContractError as exc:
            result.err(str(exc))
        if (
            g.get("allows_interpolation_default") is True
            and policy is not None
            and policy.get("gap_policy", {}).get("default_allows_interpolation") is False
        ):
            result.err("gap interpolation default must be false")

    for m in metric_rows:
        if (
            str(m.get("status")) == "computed"
            and m.get("value") is not None
            and "STAGE_9A" not in str(m.get("reason_codes"))
        ):
            result.err("computed physical metric value forbidden in Stage 9A contracts")
        if m.get("value") == 0.0 and str(m.get("status")) == "not_evaluable":
            result.err("ZERO_VS_NULL_VS_NOT_EVALUABLE conflict")

    if policy is not None:
        zones = policy.get("zones", {})
        for name in list(zones.get("neutral_geometric_zones_allowed") or []):
            try:
                assert_zone_name_allowed(str(name))
            except PhysicalContractError as exc:
                result.err(str(exc))
        for forbidden in ("first_third", "final_third"):
            try:
                assert_zone_name_allowed(forbidden)
                result.err(f"attack-relative zone unexpectedly allowed: {forbidden}")
            except PhysicalContractError:
                pass

    return result


def recount_sample_layers(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"raw_observed": 0, "filtered": 0, "resampled": 0, "total": 0}
    for s in samples:
        src = str(s.get("sample_source"))
        if src in counts:
            counts[src] += 1
        counts["total"] += 1
    return counts


__all__ = [
    "PhysicalValidationResult",
    "validate_physical_bundle",
    "recount_sample_layers",
]
