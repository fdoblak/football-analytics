#!/usr/bin/env python3
"""Validate Stage 9C distance / speed / sprint baseline (synthetic math only).

Exit codes:
  0  success
  1  validation finding/failure
  2  configuration failure
  3  integrity failure
"""

from __future__ import annotations

import argparse
import json
import math
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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/distance_speed_sprint_checks")
GATE_PASS = "PASS — DISTANCE SPEED AND SPRINT BASELINE ACTIVE"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — DISTANCE SPEED AND SPRINT BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — PHYSICAL MOTION METRIC FAILURE"

TOL = 1e-6


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

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

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


def _near(a: float | None, b: float, *, tol: float = TOL) -> bool:
    if a is None:
        return False
    return abs(float(a) - float(b)) <= tol


def run_checks(*, keep: bool) -> Result:
    from football_analytics.physical.distance import compute_segment_distance
    from football_analytics.physical.motion_config import (
        load_motion_baseline_config,
        motion_baseline_config_fingerprint,
    )
    from football_analytics.physical.motion_evaluation import NOT_EVALUATED_MOTION
    from football_analytics.physical.motion_fixtures import (
        below_sprint_threshold_points,
        constant_speed_points,
        gap_split_sprints,
        hard_gap_two_segments,
        hysteresis_sprint_points,
        known_distance_points,
        low_coverage_points,
        multi_sprint_points,
        outlier_spike_points,
        short_burst_below_min_duration,
        shot_boundary_split,
        single_point_segment,
        single_sprint_points,
        uncertain_not_evaluable_points,
        vfr_constant_speed_points,
        zero_delta_points,
    )
    from football_analytics.physical.motion_service import compute_physical_motion
    from football_analytics.physical.speed import compute_segment_speeds
    from football_analytics.physical.sprint import (
        count_evaluable_sprints,
        extract_sprint_bouts_for_segment,
    )

    result = Result()
    sessions: list[Path] = []
    try:
        cfg = load_motion_baseline_config()
        fp = motion_baseline_config_fingerprint(cfg)
        result.extras["config_fingerprint"] = fp
        result.extras["primary_sample_layer"] = cfg["primary_sample_layer"]
        result.extras["metric_origin"] = cfg["metric_origin"]
        result.extras["definition_style"] = cfg["definition_style"]

        if (
            cfg["metric_origin"] != "project_generated"
            or cfg["sprint"]["not_official_opta"] is not True
        ):
            result.fail("00_metadata", "must be project_generated / not official Opta")
        else:
            result.ok("00_metadata")

        # 01 constant speed straight run
        pts = constant_speed_points(fp, speed_mps=5.0, n=8)
        expected_dist = 5.0 * 7 * 0.1  # 3.5
        d = compute_segment_distance(
            pts, trajectory_segment_id="traj_seg_01", sample_layer="filtered", config=cfg
        )
        s = compute_segment_speeds(
            pts, trajectory_segment_id="traj_seg_01", sample_layer="filtered", config=cfg
        )
        if (
            d.status == "computed"
            and _near(d.distance_m, expected_dist)
            and _near(s.robust_mean_mps, 5.0, tol=0.05)
        ):
            result.ok("01_constant_speed")
            result.extras["constant_speed"] = {
                "expected_distance_m": expected_dist,
                "actual_distance_m": d.distance_m,
                "expected_speed_mps": 5.0,
                "actual_mean_mps": s.robust_mean_mps,
            }
        else:
            result.fail(
                "01_constant_speed",
                f"dist={d.distance_m} mean={s.robust_mean_mps} status={d.status}/{s.status}",
            )

        # 02 known distance
        pts2 = known_distance_points(fp)
        d2 = compute_segment_distance(
            pts2, trajectory_segment_id="traj_seg_known", sample_layer="filtered", config=cfg
        )
        if d2.status == "computed" and _near(d2.distance_m, 10.0):
            result.ok("02_known_distance")
        else:
            result.fail("02_known_distance", f"got {d2.distance_m}")

        # 03 VFR timestamps
        pts3 = vfr_constant_speed_points(fp)
        s3 = compute_segment_speeds(
            pts3, trajectory_segment_id="traj_seg_vfr", sample_layer="filtered", config=cfg
        )
        if s3.status == "computed" and _near(s3.robust_mean_mps, 4.0, tol=0.05):
            result.ok("03_vfr_timestamps")
        else:
            result.fail("03_vfr_timestamps", f"mean={s3.robust_mean_mps}")

        # 04 zero/negative delta
        pts4 = zero_delta_points(fp)
        d4 = compute_segment_distance(
            pts4, trajectory_segment_id="traj_seg_zd", sample_layer="filtered", config=cfg
        )
        if d4.status == "not_evaluable" and "ZERO_OR_NEGATIVE_DELTA_TIME" in d4.reason_codes:
            result.ok("04_zero_delta_rejected")
        else:
            result.fail("04_zero_delta_rejected", f"{d4.status} {d4.reason_codes}")

        # 05 single point
        pts5 = single_point_segment(fp)
        d5 = compute_segment_distance(
            pts5, trajectory_segment_id="traj_seg_single", sample_layer="filtered", config=cfg
        )
        s5 = compute_segment_speeds(
            pts5, trajectory_segment_id="traj_seg_single", sample_layer="filtered", config=cfg
        )
        if d5.status == "not_evaluable" and s5.status == "not_evaluable":
            result.ok("05_single_point_not_evaluable")
        else:
            result.fail("05_single_point_not_evaluable", f"{d5.status}/{s5.status}")

        # 06 hard gap no bridge
        gap = hard_gap_two_segments(fp)
        da = compute_segment_distance(
            gap["traj_seg_a"],
            trajectory_segment_id="traj_seg_a",
            sample_layer="filtered",
            config=cfg,
        )
        db = compute_segment_distance(
            gap["traj_seg_b"],
            trajectory_segment_id="traj_seg_b",
            sample_layer="filtered",
            config=cfg,
        )
        bridged = (da.distance_m or 0) + (db.distance_m or 0)
        # Would-be bridge from last of a to first of b
        a_last = gap["traj_seg_a"][-1]
        b_first = gap["traj_seg_b"][0]
        bridge_dist = math.hypot(
            float(b_first["pitch_x_m"]) - float(a_last["pitch_x_m"]),
            float(b_first["pitch_y_m"]) - float(a_last["pitch_y_m"]),
        )
        sess = _session("gap_")
        sessions.append(sess)
        flat = gap["traj_seg_a"] + gap["traj_seg_b"]
        res_gap = compute_physical_motion(primary_points=flat, output_dir=sess / "out", config=cfg)
        measured = res_gap.summary.get("measured_distance_m")
        if (
            res_gap.accepted
            and measured is not None
            and _near(measured, bridged, tol=1e-4)
            and measured < bridged + bridge_dist - 1.0
        ):
            result.ok("06_hard_gap_no_bridge")
        else:
            result.fail(
                "06_hard_gap_no_bridge",
                f"measured={measured} sum_seg={bridged} bridge={bridge_dist}",
            )

        # 07 shot/track boundary
        shot = shot_boundary_split(fp)
        flat_shot = shot["traj_seg_pre_shot"] + shot["traj_seg_post_shot"]
        sess7 = _session("shot_")
        sessions.append(sess7)
        res7 = compute_physical_motion(
            primary_points=flat_shot, output_dir=sess7 / "out", config=cfg
        )
        d_pre = compute_segment_distance(
            shot["traj_seg_pre_shot"],
            trajectory_segment_id="traj_seg_pre_shot",
            sample_layer="filtered",
            config=cfg,
        )
        d_post = compute_segment_distance(
            shot["traj_seg_post_shot"],
            trajectory_segment_id="traj_seg_post_shot",
            sample_layer="filtered",
            config=cfg,
        )
        expected7 = float(d_pre.distance_m or 0) + float(d_post.distance_m or 0)
        if (
            res7.accepted
            and res7.summary.get("measured_distance_m") is not None
            and _near(float(res7.summary["measured_distance_m"]), expected7, tol=1e-4)
        ):
            result.ok("07_shot_boundary_split")
        else:
            result.fail(
                "07_shot_boundary_split",
                f"measured={res7.summary.get('measured_distance_m')} expected={expected7}",
            )

        # 08 outlier spike not peak
        pts8 = outlier_spike_points(fp)
        s8 = compute_segment_speeds(
            pts8, trajectory_segment_id="traj_seg_01", sample_layer="filtered", config=cfg
        )
        if (
            s8.diagnostic_raw_peak_mps is not None
            and s8.diagnostic_raw_peak_mps > 12.0
            and (s8.robust_peak_mps is None or s8.robust_peak_mps < 12.0)
            and s8.robust_mean_mps is not None
            and s8.robust_mean_mps < 12.0
        ):
            result.ok("08_outlier_not_customer_peak")
        else:
            result.fail(
                "08_outlier_not_customer_peak",
                f"raw={s8.diagnostic_raw_peak_mps} peak={s8.robust_peak_mps} "
                f"mean={s8.robust_mean_mps}",
            )

        # 09 low coverage → not_evaluable at aggregate
        pts9 = low_coverage_points(fp)
        sess9 = _session("lowcov_")
        sessions.append(sess9)
        # Force large analysis window
        res9 = compute_physical_motion(
            primary_points=pts9,
            output_dir=sess9 / "out",
            config=cfg,
            analysis_window_us=10_000_000,
        )
        if res9.summary.get("distance_status") == "not_evaluable":
            result.ok("09_low_coverage_not_evaluable")
        else:
            result.fail("09_low_coverage_not_evaluable", str(res9.summary.get("distance_status")))

        # 10 below sprint threshold
        pts10 = below_sprint_threshold_points(fp)
        b10 = extract_sprint_bouts_for_segment(
            pts10,
            trajectory_segment_id="traj_seg_nosprint",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        if count_evaluable_sprints(b10)["sprint_count"] == 0:
            result.ok("10_below_sprint_threshold")
        else:
            result.fail("10_below_sprint_threshold", f"count={len(b10)}")

        # 11 hysteresis
        pts11 = hysteresis_sprint_points(fp)
        b11 = extract_sprint_bouts_for_segment(
            pts11,
            trajectory_segment_id="traj_seg_hyst",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        ev11 = count_evaluable_sprints(b11)["sprint_count"]
        # 8 intervals at 8 m/s + 3 at 6.5 (still in) = 1.1s+, distance > 5
        if ev11 == 1:
            result.ok("11_hysteresis_sprint")
        else:
            result.fail("11_hysteresis_sprint", f"evaluable={ev11} bouts={len(b11)}")

        # 12 short burst
        pts12 = short_burst_below_min_duration(fp)
        b12 = extract_sprint_bouts_for_segment(
            pts12,
            trajectory_segment_id="traj_seg_short",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        if count_evaluable_sprints(b12)["sprint_count"] == 0:
            result.ok("12_short_burst_not_sprint")
        else:
            result.fail("12_short_burst_not_sprint", f"{b12}")

        # 13 single sprint
        pts13 = single_sprint_points(fp)
        b13 = extract_sprint_bouts_for_segment(
            pts13,
            trajectory_segment_id="traj_seg_sprint1",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        st13 = count_evaluable_sprints(b13)
        if st13["sprint_count"] == 1 and st13["sprint_distance_m"] >= 5.0:
            result.ok("13_single_sprint")
            result.extras["single_sprint"] = st13
        else:
            result.fail("13_single_sprint", str(st13))

        # 14 multi sprint
        pts14 = multi_sprint_points(fp)
        b14 = extract_sprint_bouts_for_segment(
            pts14,
            trajectory_segment_id="traj_seg_multi",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        if count_evaluable_sprints(b14)["sprint_count"] == 2:
            result.ok("14_multi_sprint")
        else:
            result.fail(
                "14_multi_sprint",
                f"count={count_evaluable_sprints(b14)['sprint_count']} n={len(b14)}",
            )

        # 15 gap-separated sprints not merged
        gs = gap_split_sprints(fp)
        b_a = extract_sprint_bouts_for_segment(
            gs["traj_seg_sa"],
            trajectory_segment_id="traj_seg_sa",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        b_b = extract_sprint_bouts_for_segment(
            gs["traj_seg_sb"],
            trajectory_segment_id="traj_seg_sb",
            sample_layer="filtered",
            config=cfg,
            config_fingerprint=fp,
        )
        if (
            count_evaluable_sprints(b_a)["sprint_count"] == 1
            and count_evaluable_sprints(b_b)["sprint_count"] == 1
        ):
            result.ok("15_gap_split_sprints")
        else:
            result.fail("15_gap_split_sprints", f"a={len(b_a)} b={len(b_b)}")

        # 16 uncertainty not_evaluable
        pts16 = uncertain_not_evaluable_points(fp)
        d16 = compute_segment_distance(
            pts16, trajectory_segment_id="traj_seg_01", sample_layer="filtered", config=cfg
        )
        if d16.status == "not_evaluable":
            result.ok("16_uncertainty_not_evaluable")
        else:
            result.fail("16_uncertainty_not_evaluable", f"{d16.status} {d16.distance_m}")

        # 17 deterministic repeat
        sess17a = _session("det_a_")
        sess17b = _session("det_b_")
        sessions.extend([sess17a, sess17b])
        pts17 = constant_speed_points(fp, speed_mps=5.0, n=8)
        r_a = compute_physical_motion(primary_points=pts17, output_dir=sess17a / "out", config=cfg)
        r_b = compute_physical_motion(primary_points=pts17, output_dir=sess17b / "out", config=cfg)
        if (
            r_a.accepted
            and r_b.accepted
            and r_a.summary.get("measured_distance_m") == r_b.summary.get("measured_distance_m")
            and r_a.config_fingerprint == r_b.config_fingerprint
            and r_a.summary.get("evaluation_status") == NOT_EVALUATED_MOTION
        ):
            result.ok("17_deterministic_repeat")
        else:
            result.fail("17_deterministic_repeat", "mismatch")

        # 18 end-to-end smoke + receipt required
        sess18 = _session("e2e_")
        sessions.append(sess18)
        pts18 = single_sprint_points(fp)
        r18 = compute_physical_motion(primary_points=pts18, output_dir=sess18 / "out", config=cfg)
        if (
            r18.accepted
            and r18.receipt_json
            and Path(r18.receipt_json).is_file()
            and r18.summary.get("sprint_count", 0) >= 1
            and r18.summary.get("evaluation_status") == NOT_EVALUATED_MOTION
        ):
            result.ok("18_e2e_receipt")
            result.extras["e2e_summary"] = {
                k: r18.summary.get(k)
                for k in (
                    "measured_distance_m",
                    "robust_mean_speed_mps",
                    "robust_peak_speed_mps",
                    "sprint_count",
                    "evaluation_status",
                )
            }
        else:
            result.fail("18_e2e_receipt", str(r18.to_summary()))

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator_exception: {type(exc).__name__}: {exc}", config=True)
    finally:
        if not keep:
            for s in sessions:
                shutil.rmtree(s, ignore_errors=True)
            # cleanup leftover empty runtime root children if any
            if RUNTIME_ROOT.exists():
                for child in RUNTIME_ROOT.iterdir():
                    if child.is_dir() and not any(child.iterdir()):
                        shutil.rmtree(child, ignore_errors=True)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep runtime session dirs")
    parser.add_argument("--json", action="store_true", help="Emit JSON report to stdout")
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
        gate = (
            GATE_PASS
            if result.status == "PASS"
            else (GATE_FINDINGS if result.status == "PASS_WITH_WARNINGS" else GATE_FAIL)
        )
        if result.status == "PASS":
            # Math pass still does not validate real football accuracy.
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
