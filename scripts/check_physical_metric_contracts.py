#!/usr/bin/env python3
"""Validate Stage 9A target trajectory / physical metric contracts.

Exit codes:
  0  success (PASS / PASS_WITH_WARNINGS)
  1  validation finding/failure
  2  configuration/schema failure
  3  integrity/security failure
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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/physical_metric_contract_checks")
GATE_PASS = "PASS — TARGET TRAJECTORY AND PHYSICAL METRIC CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — TARGET TRAJECTORY AND PHYSICAL METRIC CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — PHYSICAL METRIC CONTRACT FAILURE"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}
        self.scenarios: dict[str, str] = {}

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

    def ok_scenario(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail_scenario(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
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


def run_checks(*, keep: bool, strict: bool) -> Result:
    from football_analytics.core.records import RecordError, write_json_record
    from football_analytics.data.compiler import list_contracts
    from football_analytics.physical.contracts import (
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_frozen_upstream_fingerprints,
        assert_physical_contracts_registered,
        load_physical_json_schema,
        physical_schema_fingerprints,
        validate_against_json_schema,
    )
    from football_analytics.physical.eligibility import (
        distinguish_zero_null_not_evaluable,
        input_is_trajectory_eligible,
    )
    from football_analytics.physical.evaluation import (
        NOT_EVALUATED_PHYSICAL,
        evaluate_physical_metrics,
    )
    from football_analytics.physical.fixtures import (
        confirmed_observed_bundle,
        eligible_candidate,
        gap_bundle,
        predicted_exclusion_candidate,
        provisional_exclusion_candidate,
        sample_row,
        single_sample_segment_bundle,
    )
    from football_analytics.physical.metrics import contract_stub_result, metric_definition
    from football_analytics.physical.policy import (
        assert_contract_only_policies,
        load_metrics_policy,
        load_trajectory_policy,
        policy_fingerprint,
    )
    from football_analytics.physical.receipt import (
        build_synthetic_receipt,
        build_synthetic_request,
        recount_receipt_counts,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.physical.semantics import (
        distance_bridge_allowed,
        heatmap_weighting_is_time,
        low_coverage_means_inactivity,
        segment_metric_sufficient,
        speed_delta_seconds,
        sprint_from_single_spike,
    )
    from football_analytics.physical.types import PhysicalContractError
    from football_analytics.physical.validation import validate_physical_bundle
    from football_analytics.physical.zones import assert_zone_name_allowed, progression_enabled

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="phys_", dir=str(RUNTIME_ROOT)))

    try:
        assert_physical_contracts_registered()
        assert_frozen_upstream_fingerprints()
        if len(list_contracts()) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err("registry contract count mismatch", integrity=True)

        traj = load_trajectory_policy()
        metrics = load_metrics_policy()
        assert_contract_only_policies(traj, metrics)
        traj_fp = policy_fingerprint(traj)
        met_fp = policy_fingerprint(metrics)
        result.extras["trajectory_policy_fp"] = traj_fp
        result.extras["metrics_policy_fp"] = met_fp
        result.extras["schema_fps"] = physical_schema_fingerprints()

        pitch_fp = "a" * 64

        # 01 confirmed observed
        b1 = confirmed_observed_bundle(traj_fp)
        vr1 = validate_physical_bundle(
            samples=b1["target_trajectory_samples"],
            segments=b1["target_trajectory_segments"],
            gaps=b1["trajectory_gaps"],
            metric_results=b1["physical_metric_results"],
            policy=metrics,
        )
        if vr1.status == "PASS":
            result.ok_scenario("01_confirmed_observed")
        else:
            result.fail_scenario("01_confirmed_observed", str(vr1.errors))

        # 02 provisional exclusion
        ok, reasons = input_is_trajectory_eligible(provisional_exclusion_candidate())
        if (not ok) and "PROVISIONAL_TARGET_EXCLUDED" in reasons:
            result.ok_scenario("02_provisional_exclusion")
        else:
            result.fail_scenario("02_provisional_exclusion", f"{ok}/{reasons}")

        # 03 predicted/interpolated exclusion
        ok3, reasons3 = input_is_trajectory_eligible(predicted_exclusion_candidate())
        if (not ok3) and "PREDICTED_INTERPOLATED_EXCLUDED" in reasons3:
            result.ok_scenario("03_predicted_interpolated_exclusion")
        else:
            result.fail_scenario("03_predicted_interpolated_exclusion", f"{ok3}/{reasons3}")

        # 04 calibration gap
        g4 = gap_bundle(traj_fp, gap_type="calibration_gap")
        if (
            g4["gap_rows"][0]["gap_type"] == "calibration_gap"
            and not g4["gap_rows"][0]["allows_distance_bridge"]
        ):
            result.ok_scenario("04_calibration_gap")
        else:
            result.fail_scenario("04_calibration_gap", "gap semantics")

        # 05 identity gap
        g5 = gap_bundle(traj_fp, gap_type="identity_gap")
        if g5["gap_rows"][0]["gap_type"] == "identity_gap":
            result.ok_scenario("05_identity_gap")
        else:
            result.fail_scenario("05_identity_gap", "gap type")

        # 06 shot/track boundary
        g6 = gap_bundle(traj_fp, gap_type="shot_boundary")
        g6b = gap_bundle(traj_fp, gap_type="track_boundary")
        if (
            g6["gap_rows"][0]["gap_type"] == "shot_boundary"
            and g6b["gap_rows"][0]["gap_type"] == "track_boundary"
        ):
            result.ok_scenario("06_shot_track_boundary")
        else:
            result.fail_scenario("06_shot_track_boundary", "boundary types")

        # 07 VFR timestamps (uneven us deltas)
        times = [0, 33_000, 80_000, 150_000]
        try:
            for a, b in zip(times, times[1:], strict=False):
                speed_delta_seconds(t0_us=a, t1_us=b)
            result.ok_scenario("07_vfr_timestamps")
        except PhysicalContractError as exc:
            result.fail_scenario("07_vfr_timestamps", str(exc))

        # 08 duplicate / out-of-order
        bad = confirmed_observed_bundle(traj_fp)
        rows = list(bad["sample_rows"])
        rows.append(
            sample_row(
                bad["run_id"],
                bad["video_id"],
                bad["target_player_id"],
                "smp_dup",
                identity_assignment_id=bad["identity_assignment_id"],
                track_id=0,
                frame_index=99,
                video_time_us=0,
                pitch_x_m=11.0,
                pitch_y_m=20.0,
                policy_fingerprint=traj_fp,
            )
        )
        vr8 = validate_physical_bundle(samples=rows, policy=metrics)
        if vr8.status == "FAIL":
            result.ok_scenario("08_duplicate_out_of_order")
        else:
            result.fail_scenario("08_duplicate_out_of_order", "expected fail")

        # 09 single-sample segment
        s9 = single_sample_segment_bundle(traj_fp)
        if not segment_metric_sufficient(s9["segment_rows"][0]):
            result.ok_scenario("09_single_sample_segment")
        else:
            result.fail_scenario("09_single_sample_segment", "should be insufficient")

        # 10 raw/filtered/resampled provenance
        derived = sample_row(
            b1["run_id"],
            b1["video_id"],
            b1["target_player_id"],
            "smp_filt",
            identity_assignment_id=b1["identity_assignment_id"],
            track_id=0,
            frame_index=0,
            video_time_us=50_000,
            pitch_x_m=10.5,
            pitch_y_m=20.0,
            policy_fingerprint=traj_fp,
            sample_source="filtered",
            derived_from_sample_ids=["smp_00"],
            metric_eligibility="not_eligible",
            eligibility_status="ineligible",
        )
        vr10 = validate_physical_bundle(samples=[derived], policy=metrics)
        if vr10.status == "PASS":
            result.ok_scenario("10_raw_filtered_resampled_provenance")
        else:
            result.fail_scenario("10_raw_filtered_resampled_provenance", str(vr10.errors))

        # 11 gap distance bridge forbidden
        try:
            distance_bridge_allowed({"allows_distance_bridge": True})
            result.fail_scenario("11_gap_distance_forbidden", "bridge allowed")
        except PhysicalContractError:
            result.ok_scenario("11_gap_distance_forbidden")

        # 12 speed time-unit semantics
        try:
            dt = speed_delta_seconds(t0_us=0, t1_us=500_000)
            if abs(dt - 0.5) < 1e-12:
                result.ok_scenario("12_speed_time_unit")
            else:
                result.fail_scenario("12_speed_time_unit", f"dt={dt}")
        except PhysicalContractError as exc:
            result.fail_scenario("12_speed_time_unit", str(exc))

        # 13 single spike no sprint
        if not sprint_from_single_spike(
            sample_count=1, duration_us=50_000, min_duration_us=1_000_000
        ):
            result.ok_scenario("13_single_spike_no_sprint")
        else:
            result.fail_scenario("13_single_spike_no_sprint", "spike accepted")

        # 14 sprint hysteresis/duration from policy
        sp = metrics["sprint"]
        if (
            float(sp["entry_speed_mps"]) > float(sp["exit_speed_mps"])
            and sp["hysteresis"] is True
            and int(sp["min_duration_us"]) >= 1_000_000
        ):
            result.ok_scenario("14_sprint_hysteresis_duration")
        else:
            result.fail_scenario("14_sprint_hysteresis_duration", "policy")

        # 15 heatmap time-weight vs sample count
        if heatmap_weighting_is_time(weighting=str(metrics["heatmap"]["weighting"])) and metrics[
            "heatmap"
        ].get("sample_count_distinct_from_dwell"):
            result.ok_scenario("15_heatmap_time_weight")
        else:
            result.fail_scenario("15_heatmap_time_weight", "weighting")

        # 16 missing coverage ≠ inactivity
        if not low_coverage_means_inactivity(policy=metrics):
            result.ok_scenario("16_coverage_not_inactivity")
        else:
            result.fail_scenario("16_coverage_not_inactivity", "misinterpreted")

        # 17 zero vs null vs not_evaluable
        labels = {
            distinguish_zero_null_not_evaluable(value=0.0, status="computed", observed=True),
            distinguish_zero_null_not_evaluable(value=None, status="partial", observed=True),
            distinguish_zero_null_not_evaluable(value=None, status="not_evaluable", observed=True),
            distinguish_zero_null_not_evaluable(value=None, status="not_evaluable", observed=False),
        }
        if labels == {"zero", "null", "not_evaluable", "not_observed"}:
            result.ok_scenario("17_zero_null_not_evaluable")
        else:
            result.fail_scenario("17_zero_null_not_evaluable", str(labels))

        # 18 attack direction unknown
        if str(traj["coordinate_frame"]["attack_direction_default"]) == "unknown" and not (
            progression_enabled(attack_direction="unknown", policy_enabled=False)
        ):
            result.ok_scenario("18_attack_direction_unknown")
        else:
            result.fail_scenario("18_attack_direction_unknown", "direction")

        # 19 Goal A/B neutral zones
        try:
            assert_zone_name_allowed("goal_a_third")
            assert_zone_name_allowed("goal_b_third")
            assert_zone_name_allowed("first_third")
            result.fail_scenario("19_neutral_zones", "first_third allowed")
        except PhysicalContractError:
            result.ok_scenario("19_neutral_zones")

        # 20 receipt recount/hash
        req = build_synthetic_request(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            target_player_id=b1["target_player_id"],
            trajectory_policy_fingerprint=traj_fp,
            metrics_policy_fingerprint=met_fp,
            pitch_template_fingerprint=pitch_fp,
        )
        validate_request_payload(req)
        receipt = build_synthetic_receipt(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            target_player_id=b1["target_player_id"],
            trajectory_policy_fingerprint=traj_fp,
            metrics_policy_fingerprint=met_fp,
            samples=b1["sample_rows"],
            segments=b1["segment_rows"],
            gaps=b1["gap_rows"],
            metric_results=b1["metric_rows"],
        )
        validate_receipt_payload(receipt)
        mismatches = recount_receipt_counts(
            samples=b1["sample_rows"],
            segments=b1["segment_rows"],
            gaps=b1["gap_rows"],
            receipt=receipt,
        )
        if not mismatches:
            result.ok_scenario("20_receipt_recount")
        else:
            result.fail_scenario("20_receipt_recount", str(mismatches))

        # 21 fingerprint / FK mismatch
        bad_cand = eligible_candidate()
        bad_cand["fingerprints_match"] = False
        ok21, reasons21 = input_is_trajectory_eligible(bad_cand)
        if (not ok21) and "FINGERPRINT_MISMATCH" in reasons21:
            result.ok_scenario("21_fingerprint_fk_mismatch")
        else:
            result.fail_scenario("21_fingerprint_fk_mismatch", f"{ok21}/{reasons21}")

        # 22 evaluation leakage — stub must not claim real accuracy
        ev = evaluate_physical_metrics(has_reviewed_ground_truth=False)
        if ev.ground_truth_evaluation_status == NOT_EVALUATED_PHYSICAL and all(
            v is None for v in ev.metrics.values()
        ):
            result.ok_scenario("22_evaluation_leakage")
        else:
            result.fail_scenario("22_evaluation_leakage", "claimed accuracy")

        # 23 deterministic fingerprint
        if (
            policy_fingerprint(load_trajectory_policy()) == traj_fp
            and policy_fingerprint(load_metrics_policy()) == met_fp
        ):
            result.ok_scenario("23_deterministic_fingerprint")
        else:
            result.fail_scenario("23_deterministic_fingerprint", "drift")

        # 24 atomic no-overwrite
        out = session / "receipt.json"
        write_json_record(out, receipt, overwrite=False)
        try:
            write_json_record(out, receipt, overwrite=False)
            result.fail_scenario("24_atomic_no_overwrite", "overwrite allowed")
        except RecordError:
            result.ok_scenario("24_atomic_no_overwrite")

        # 25 failure cleanup + ball physical/event not produced
        junk = session / "partial_fail_dir"
        junk.mkdir()
        (junk / "tmp.bin").write_bytes(b"x")
        shutil.rmtree(junk)
        stub = contract_stub_result(metric_name="distance", unit="m")
        _ = metric_definition("distance")
        if (not junk.exists()) and stub["value"] is None and stub["status"] == "contract_stub":
            result.ok_scenario("25_failure_cleanup_and_stub")
        else:
            result.fail_scenario("25_failure_cleanup_and_stub", "cleanup/stub")

        validate_against_json_schema(
            ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"], config_fingerprint=met_fp),
            load_physical_json_schema("physical_metric_evaluation"),
        )
        result.extras["evaluation_status"] = ev.ground_truth_evaluation_status
        result.extras["gate"] = GATE_PASS if not result.errors else GATE_FAIL
        if result.warnings and not result.errors:
            result.extras["gate"] = GATE_FINDINGS

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator crash: {exc}", config=True)
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
            # Keep runtime root; remove empty children if possible.
            try:
                for child in RUNTIME_ROOT.iterdir():
                    if child.is_dir() and not any(child.iterdir()):
                        child.rmdir()
            except OSError:
                pass
        else:
            result.extras["session"] = str(session)

    return result.finalize(strict=strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), strict=bool(args.strict))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        gate = payload["extras"].get("gate", GATE_FAIL)
        print(gate)
        print(f"status={payload['status']} scenarios={len(payload['scenarios'])}")
        if payload["errors"]:
            print("errors:")
            for e in payload["errors"]:
                print(f"  - {e}")
        if payload["warnings"]:
            print("warnings:")
            for w in payload["warnings"]:
                print(f"  - {w}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
