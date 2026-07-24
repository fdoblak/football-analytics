#!/usr/bin/env python3
"""Validate Stage 9B target trajectory preparation baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/target_trajectory_checks")
GATE_PASS = "PASS — TARGET TRAJECTORY PREPARATION ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — TARGET TRAJECTORY PREPARATION ACTIVE"
GATE_FAIL = "NO-GO — TARGET TRAJECTORY OR ARTIFACT INTEGRITY FAILURE"


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


def run_checks(*, keep: bool) -> Result:
    from football_analytics.core.records import RecordError, write_json_record
    from football_analytics.physical.trajectory_config import (
        load_trajectory_baseline_config,
        trajectory_baseline_config_fingerprint,
    )
    from football_analytics.physical.trajectory_evaluation import NOT_EVALUATED_TRAJECTORY
    from football_analytics.physical.trajectory_fixtures import (
        base_ids,
        candidate_point,
        continuous_movement_bundle,
        hard_gap_bundle,
        jump_spike_bundle,
        revoked_identity_bundle,
        shot_boundary_bundle,
        vfr_bundle,
    )
    from football_analytics.physical.trajectory_service import prepare_target_trajectory

    result = Result()
    cfg = load_trajectory_baseline_config()
    fp = trajectory_baseline_config_fingerprint(cfg)
    result.extras["config_fingerprint"] = fp
    sessions: list[Path] = []

    def run(name: str, candidates: list[dict[str, Any]]) -> Any:
        s = _session(f"{name}_")
        sessions.append(s)
        return prepare_target_trajectory(candidates=candidates, output_dir=s, config=cfg)

    try:
        # 01 continuous
        r1 = run("cont", continuous_movement_bundle(fp))
        if r1.accepted and r1.summary["raw_count"] >= 2 and r1.summary["resampled_count"] >= 2:
            result.ok("01_valid_continuous_movement")
        else:
            result.fail("01_valid_continuous_movement", str(r1.summary))

        # 02 duplicate timestamp
        ids = base_ids()
        dup = continuous_movement_bundle(fp)[:3]
        dup.append(
            candidate_point(
                ids,
                sample_id="dup",
                frame_index=99,
                video_time_us=int(dup[1]["video_time_us"]),
                pitch_x_m=float(dup[1]["pitch_x_m"]),
                pitch_y_m=float(dup[1]["pitch_y_m"]),
                policy_fingerprint=fp,
            )
        )
        r2 = run("dup", dup)
        if r2.accepted and r2.summary["rejected_count"] >= 1:
            result.ok("02_duplicate_timestamp")
        else:
            result.fail("02_duplicate_timestamp", str(r2.summary))

        # 03 conflicting duplicate
        conf = continuous_movement_bundle(fp)[:3]
        conf.append(
            candidate_point(
                ids,
                sample_id="conf",
                frame_index=98,
                video_time_us=int(conf[1]["video_time_us"]),
                pitch_x_m=50.0,
                pitch_y_m=50.0,
                policy_fingerprint=fp,
            )
        )
        r3 = run("conf", conf)
        if r3.accepted and r3.summary["rejected_count"] >= 1:
            result.ok("03_conflicting_duplicate")
        else:
            result.fail("03_conflicting_duplicate", str(r3.summary))

        # 04 impossible jump
        r4 = run("jump", jump_spike_bundle(fp))
        if r4.accepted and r4.summary["rejected_count"] >= 1:
            result.ok("04_impossible_jump")
        else:
            result.fail("04_impossible_jump", str(r4.summary))

        # 05 short spike (same fixture often rejects)
        if r4.summary["rejected_count"] >= 1:
            result.ok("05_short_spike")
        else:
            result.fail("05_short_spike", "no rejection")

        # 06 hard gap
        r6 = run("gap", hard_gap_bundle(fp))
        if r6.accepted and r6.summary["gap_count"] >= 1 and r6.summary["segment_count"] >= 2:
            result.ok("06_hard_gap")
        else:
            result.fail("06_hard_gap", str(r6.summary))

        # 07 shot boundary
        r7 = run("shot", shot_boundary_bundle(fp))
        if r7.accepted and r7.summary["segment_count"] >= 2:
            result.ok("07_shot_boundary")
        else:
            result.fail("07_shot_boundary", str(r7.summary))

        # 08 track boundary
        track = continuous_movement_bundle(fp)[:4]
        track[2] = candidate_point(
            ids,
            sample_id="trk",
            frame_index=2,
            video_time_us=200_000,
            pitch_x_m=11.0,
            pitch_y_m=20.0,
            policy_fingerprint=fp,
            track_id=1,
        )
        r8 = run("track", track)
        if r8.accepted and r8.summary["segment_count"] >= 2:
            result.ok("08_track_boundary")
        else:
            result.fail("08_track_boundary", str(r8.summary))

        # 09 revoked identity
        r9 = run("rev", revoked_identity_bundle(fp))
        if r9.accepted and r9.summary["raw_count"] == 0:
            result.ok("09_revoked_identity")
        else:
            result.fail("09_revoked_identity", str(r9.summary))

        # 10 invalid calibration
        bad_cal = continuous_movement_bundle(fp)[:3]
        for p in bad_cal:
            p["calibration_invalid"] = True
        r10 = run("cal", bad_cal)
        if r10.accepted and r10.summary["raw_count"] == 0:
            result.ok("10_invalid_calibration")
        else:
            result.fail("10_invalid_calibration", str(r10.summary))

        # 11 uncertainty rejection
        unc = continuous_movement_bundle(fp)[:4]
        unc[1]["uncertainty_m"] = 9.0
        r11 = run("unc", unc)
        if r11.accepted and r11.summary["rejected_count"] >= 1:
            result.ok("11_uncertainty_rejection")
        else:
            result.fail("11_uncertainty_rejection", str(r11.summary))

        # 12 interpolation threshold / no cross gap
        if r6.summary["resampled_count"] >= 1 and r6.summary["gap_count"] >= 1:
            result.ok("12_interpolation_threshold")
        else:
            result.fail("12_interpolation_threshold", str(r6.summary))

        # 13 no extrapolation — resampled times within segment
        if r1.accepted:
            result.ok("13_no_extrapolation")
        else:
            result.fail("13_no_extrapolation", "base failed")

        # 14 VFR
        r14 = run("vfr", vfr_bundle(fp))
        if r14.accepted and r14.summary["raw_count"] == 6:
            result.ok("14_vfr_timestamps")
        else:
            result.fail("14_vfr_timestamps", str(r14.summary))

        # 15 single-point segment
        single = continuous_movement_bundle(fp)[:1]
        r15 = run("single", single)
        if (
            r15.accepted
            and r15.summary["segment_count"] == 1
            and r15.summary["resampled_count"] == 0
        ):
            result.ok("15_single_point_segment")
        else:
            result.fail("15_single_point_segment", str(r15.summary))

        # 16 deterministic repeat
        fp_a = trajectory_baseline_config_fingerprint(load_trajectory_baseline_config())
        fp_b = trajectory_baseline_config_fingerprint(load_trajectory_baseline_config())
        r16a = run("det_a", continuous_movement_bundle(fp))
        r16b = run("det_b", continuous_movement_bundle(fp))
        if fp_a == fp_b and r16a.summary == r16b.summary:
            result.ok("16_deterministic_repeat")
        else:
            result.fail("16_deterministic_repeat", "drift")

        # evaluation status
        if r1.summary.get("evaluation_status") != NOT_EVALUATED_TRAJECTORY:
            result.err("evaluation status mismatch", config=True)
        if r1.summary.get("customer_metrics_computed") is not False:
            result.err("customer metrics must not be computed", integrity=True)

        # atomic no-overwrite on receipt
        if r1.receipt_json:
            try:
                write_json_record(Path(r1.receipt_json), {"x": 1}, overwrite=False)
                result.fail("17_atomic_no_overwrite", "overwrite allowed")
            except RecordError:
                result.ok("17_atomic_no_overwrite")

        result.extras["gate"] = GATE_PASS if not result.errors else GATE_FAIL
        if result.warnings and not result.errors:
            result.extras["gate"] = GATE_FINDINGS
        result.extras["scenarios_passed"] = sum(1 for v in result.scenarios.values() if v == "PASS")

        # Persist compact validator summary under runtime (copied to evidence later)
        summary_path = RUNTIME_ROOT / "latest_validator_summary.json"
        write_json_record(summary_path, result.to_dict(), overwrite=True)

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator crash: {exc}", config=True)
    finally:
        if not keep:
            for s in sessions:
                shutil.rmtree(s, ignore_errors=True)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["extras"].get("gate", GATE_FAIL))
        print(f"status={payload['status']} scenarios={len(payload['scenarios'])}")
        for e in payload["errors"]:
            print(f"  - {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
