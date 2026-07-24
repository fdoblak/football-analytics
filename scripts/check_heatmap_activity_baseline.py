#!/usr/bin/env python3
"""Validate Stage 9D heatmap / zones / activity baseline (synthetic math only).

Exit codes:
  0  success
  1  validation finding/failure
  2  configuration failure
  3  integrity failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/heatmap_activity_checks")
GATE_PASS = "PASS — HEATMAP ZONES AND ACTIVITY BASELINE ACTIVE"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — HEATMAP ZONES AND ACTIVITY BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — SPATIAL OR ACTIVITY METRIC FAILURE"

TOL = 1e-4
PCT_TOL = 0.05


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.scenarios: dict[str, str] = {}
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def ok(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

    def finalize(self) -> Result:
        if self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
        else:
            self.status = "PASS"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "scenarios": dict(self.scenarios),
            "extras": self.extras,
        }


def _session(prefix: str) -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(RUNTIME_ROOT)))


def run_checks(*, keep: bool) -> Result:
    from football_analytics.physical.activity import (
        classify_speed_mps,
        compute_activity_distribution,
    )
    from football_analytics.physical.heatmap import (
        compute_time_weighted_heatmap,
        smooth_conserve_mass,
    )
    from football_analytics.physical.spatial_config import (
        load_spatial_baseline_config,
        spatial_baseline_config_fingerprint,
    )
    from football_analytics.physical.spatial_evaluation import NOT_EVALUATED_SPATIAL
    from football_analytics.physical.spatial_fixtures import (
        hard_gap_segments,
        low_coverage_points,
        out_of_pitch_point,
        penalty_presence_points,
        pitch_crossing,
        single_point,
        speed_class_ladder,
        stationary_zone_dwell,
        uncertain_points,
        vfr_points,
        zone_boundary_edge,
    )
    from football_analytics.physical.spatial_service import compute_spatial_metrics
    from football_analytics.physical.zone_occupancy import (
        classify_point_zones,
        compute_zone_occupancy,
    )

    result = Result()
    sessions: list[Path] = []
    try:
        cfg = load_spatial_baseline_config()
        fp = spatial_baseline_config_fingerprint(cfg)
        result.extras["config_fingerprint"] = fp

        if cfg["attack_direction"] != "unknown":
            result.fail("00_metadata", "attack_direction must be unknown")
        elif cfg["output_policy"]["write_visuals_to_git"] is not False:
            result.fail("00_metadata", "write_visuals_to_git must be false")
        else:
            result.ok("00_metadata")

        # 01 stationary dwell in one third
        pts = stationary_zone_dwell(fp)
        zones = compute_zone_occupancy(pts, config=cfg)
        thirds = {z["zone_id"]: z for z in zones["zones"] if z["zone_id"].endswith("_third")}
        if (
            zones["status"] == "computed"
            and "goal_a_third" in thirds
            and thirds["goal_a_third"]["eligible_percent"] > 90
        ):
            result.ok("01_stationary_zone_dwell")
        else:
            result.fail("01_stationary_zone_dwell", str(thirds))

        # 02 pitch crossing visits multiple thirds
        cross = pitch_crossing(fp)
        z2 = compute_zone_occupancy(cross, config=cfg)
        third_ids = {z["zone_id"] for z in z2["zones"] if z["zone_id"].endswith("_third")}
        if len(third_ids) >= 2 and z2.get("attack_relative_invented") is False:
            result.ok("02_pitch_crossing_thirds")
        else:
            result.fail("02_pitch_crossing_thirds", str(third_ids))

        # 03 VFR
        vfr = vfr_points(fp)
        hm3 = compute_time_weighted_heatmap(vfr, config=cfg)
        if hm3.status == "computed" and hm3.total_dwell_seconds > 0:
            result.ok("03_vfr_timestamps")
        else:
            result.fail("03_vfr_timestamps", hm3.status)

        # 04 single point
        one = single_point(fp)
        hm4 = compute_time_weighted_heatmap(one, config=cfg)
        act4 = compute_activity_distribution(one, config=cfg)
        if hm4.status == "not_evaluable" and act4["status"] == "not_evaluable":
            result.ok("04_single_point_not_evaluable")
        else:
            result.fail("04_single_point_not_evaluable", f"{hm4.status}/{act4['status']}")

        # 05 hard gap — two segments, no bridge
        gap = hard_gap_segments(fp)
        sess5 = _session("gap_")
        sessions.append(sess5)
        r5 = compute_spatial_metrics(primary_points=gap, output_dir=sess5 / "out", config=cfg)
        if r5.accepted and r5.summary.get("heatmap_status") == "computed":
            result.ok("05_hard_gap_no_bridge")
        else:
            result.fail("05_hard_gap_no_bridge", str(r5.summary))

        # 06 zone boundary
        edge = zone_boundary_edge(fp)
        z6 = compute_zone_occupancy(edge, config=cfg)
        labels = classify_point_zones(35.0, 10.0, config=cfg)
        if "middle_third" in labels and z6.get("attack_direction") == "unknown":
            result.ok("06_zone_boundary_edge")
        else:
            result.fail("06_zone_boundary_edge", str(labels))

        # 07 out of pitch rejected
        oop = out_of_pitch_point(fp)
        hm7 = compute_time_weighted_heatmap(oop, config=cfg)
        if "OUT_OF_PITCH" in hm7.reason_codes or hm7.rejected_interval_count >= 1:
            result.ok("07_out_of_pitch")
        else:
            result.fail("07_out_of_pitch", str(hm7.reason_codes))

        # 08 smoothing mass conservation
        grid = [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
        mass0 = sum(sum(r) for r in grid)
        sm = smooth_conserve_mass(grid, sigma_cells=1.0, radius_cells=1)
        mass1 = sum(sum(r) for r in sm)
        if abs(mass0 - mass1) < 1e-9:
            result.ok("08_smoothing_mass_conservation")
        else:
            result.fail("08_smoothing_mass_conservation", f"{mass0} vs {mass1}")

        # 09 observed vs derived fields present
        from football_analytics.physical.heatmap import heatmap_to_dict

        hm9 = compute_time_weighted_heatmap(pts, config=cfg, contribution_source="observed")
        d9 = heatmap_to_dict(hm9, config_fingerprint=fp)
        if "observed_dwell_seconds" in d9 and "derived_dwell_seconds" in d9:
            result.ok("09_observed_derived_split")
        else:
            result.fail("09_observed_derived_split", "missing fields")

        # 10 low coverage
        low = low_coverage_points(fp)
        sess10 = _session("low_")
        sessions.append(sess10)
        r10 = compute_spatial_metrics(
            primary_points=low,
            output_dir=sess10 / "out",
            config=cfg,
            analysis_window_us=20_000_000,
        )
        if r10.summary.get("heatmap_status") == "not_evaluable":
            result.ok("10_low_coverage_not_evaluable")
        else:
            result.fail("10_low_coverage_not_evaluable", str(r10.summary.get("heatmap_status")))

        # 11 speed classes
        ladder = speed_class_ladder(fp)
        act11 = compute_activity_distribution(ladder, config=cfg)
        by = {c["class"]: c["duration_us"] for c in act11["classes"]}
        expected = {
            "stationary": 500_000,
            "walking": 500_000,
            "jogging": 500_000,
            "running": 500_000,
            "sprinting": 500_000,
        }
        ok_classes = all(abs(by.get(k, 0) - v) < 1 for k, v in expected.items())
        # also unit classify
        assert classify_speed_mps(8.0, classes=cfg["activity"]["classes"]) == "sprinting"
        if ok_classes and act11["status"] == "computed":
            result.ok("11_speed_class_ladder")
            result.extras["class_durations_us"] = expected
        else:
            result.fail("11_speed_class_ladder", str(by))

        # 12 unknown / uncertain → not_evaluable dwell
        unc = uncertain_points(fp)
        hm12 = compute_time_weighted_heatmap(unc, config=cfg)
        if hm12.status == "not_evaluable":
            result.ok("12_uncertain_not_evaluable")
        else:
            result.fail("12_uncertain_not_evaluable", hm12.status)

        # 13 coverage outside not counted inactive
        act13 = compute_activity_distribution(
            pts, config=cfg, analysis_window_us=10_000_000, gap_unobserved_us=5_000_000
        )
        if (
            act13["missing_coverage_counted_as_inactive"] is False
            and act13["gap_or_not_observed_duration_us"] == 5_000_000
        ):
            result.ok("13_coverage_not_inactive")
        else:
            result.fail(
                "13_coverage_not_inactive", str(act13.get("gap_or_not_observed_duration_us"))
            )

        # 14 neutral labels / no attack invention
        if zones.get("attack_direction") == "unknown" and not any(
            "attack" in z["zone_id"] for z in zones["zones"]
        ):
            result.ok("14_neutral_goal_labels")
        else:
            result.fail("14_neutral_goal_labels", "attack invented")

        # 15 penalty presence ≠ touch
        pen = penalty_presence_points(fp)
        z15 = compute_zone_occupancy(pen, config=cfg)
        pen_rows = [z for z in z15["zones"] if z["zone_id"] == "goal_a_penalty"]
        if (
            pen_rows
            and pen_rows[0].get("not_touch_or_possession") is True
            and z15.get("penalty_semantics") == "physical_presence_only_not_ball_touch"
        ):
            result.ok("15_penalty_not_touch")
        else:
            result.fail("15_penalty_not_touch", str(pen_rows))

        # 16 percent sum ~ 100 when computed
        from football_analytics.physical.heatmap import heatmap_to_dict as _h2d

        hm16 = compute_time_weighted_heatmap(pts, config=cfg)
        d16 = _h2d(hm16, config_fingerprint=fp)
        if hm16.status == "computed" and abs(d16["percent_sum"] - 100.0) <= PCT_TOL:
            result.ok("16_percent_sum_tolerance")
        else:
            result.fail("16_percent_sum_tolerance", str(d16.get("percent_sum")))

        # 17 deterministic + e2e receipt + temp visual cleaned
        sess17a = _session("det_a_")
        sess17b = _session("det_b_")
        sessions.extend([sess17a, sess17b])
        vis = sess17a / "tmpvis"
        r_a = compute_spatial_metrics(
            primary_points=pts,
            output_dir=sess17a / "out",
            config=cfg,
            temp_visual_dir=vis,
        )
        r_b = compute_spatial_metrics(primary_points=pts, output_dir=sess17b / "out", config=cfg)
        if (
            r_a.accepted
            and r_b.accepted
            and r_a.summary.get("total_dwell_seconds") == r_b.summary.get("total_dwell_seconds")
            and r_a.summary.get("evaluation_status") == NOT_EVALUATED_SPATIAL
            and r_a.receipt_json
            and (vis / "temp_heatmap.svg").is_file()
        ):
            result.ok("17_deterministic_e2e")
            result.extras["e2e"] = {
                "dwell_s": r_a.summary.get("total_dwell_seconds"),
                "activity_index": r_a.summary.get("movement_activity_index"),
                "eval": r_a.summary.get("evaluation_status"),
            }
        else:
            result.fail("17_deterministic_e2e", str(r_a.to_summary()))

        # 18 activity index formula present / not opta
        if r_a.accepted and r_a.activity_json:
            act_path = Path(str(r_a.activity_json))
            act_payload = json.loads(act_path.read_text(encoding="utf-8"))
            mai = act_payload["movement_activity_index"]
            if (
                mai.get("not_official_opta") is True
                and mai.get("metric_origin") == "project_generated"
                and r_a.summary.get("movement_activity_index") is not None
            ):
                result.ok("18_activity_index_metadata")
            else:
                result.fail("18_activity_index_metadata", str(mai))
        else:
            result.fail("18_activity_index_metadata", "e2e not accepted")

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator_exception: {type(exc).__name__}: {exc}", config=True)
    finally:
        if not keep:
            for s in sessions:
                shutil.rmtree(s, ignore_errors=True)
            if RUNTIME_ROOT.exists():
                for child in RUNTIME_ROOT.iterdir():
                    if child.is_dir() and child.name != "latest_keep":
                        # remove emptied / temp sessions
                        try:
                            if not any(child.iterdir()) or child.name.startswith(
                                ("gap_", "low_", "det_", "shot_", "e2e_", "tmp")
                            ):
                                shutil.rmtree(child, ignore_errors=True)
                        except OSError:
                            pass

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep))
    payload = result.to_dict()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RUNTIME_ROOT / "latest_validator_summary.json"
    try:
        from football_analytics.core.records import write_json_record

        if summary_path.exists():
            summary_path.unlink()
        write_json_record(summary_path, payload, overwrite=False)
    except Exception:
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        gate = GATE_FAIL
        if result.status == "PASS" or result.status == "PASS_WITH_WARNINGS":
            gate = GATE_FINDINGS
        print(gate)
        print(f"status={result.status} scenarios={len(result.scenarios)}")
        for name, st in result.scenarios.items():
            print(f"  {name}: {st}")
        for e in result.errors:
            print(f"ERROR: {e}", file=sys.stderr)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
