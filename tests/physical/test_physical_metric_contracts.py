#!/usr/bin/env python3
"""Stage 9A target trajectory / physical metric contract tests."""

from __future__ import annotations

import unittest

from football_analytics.core.records import RecordError, write_json_record
from football_analytics.data.compiler import get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.physical.contracts import (
    EXPECTED_CALIBRATIONS_FP,
    EXPECTED_PROJECTED_POSITIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_frozen_upstream_fingerprints,
    assert_physical_contracts_registered,
    compile_physical_schemas,
    load_physical_json_schema,
    physical_schema_fingerprints,
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
    segment_metric_sufficient,
    speed_delta_seconds,
    sprint_from_single_spike,
)
from football_analytics.physical.types import PhysicalContractError
from football_analytics.physical.validation import validate_physical_bundle
from football_analytics.physical.zones import assert_zone_name_allowed, progression_enabled


class PhysicalMetricContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = default_project_root()
        self.reg = load_schema_registry(default_registry_path(), project_root=self.root)
        self.traj = load_trajectory_policy(project_root=self.root)
        self.metrics = load_metrics_policy(project_root=self.root)
        self.traj_fp = policy_fingerprint(self.traj)
        self.met_fp = policy_fingerprint(self.metrics)

    def test_01_registry_and_frozen_upstream(self) -> None:
        assert_physical_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_frozen_upstream_fingerprints(registry=self.reg)
        fps = physical_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["projected_positions"], EXPECTED_PROJECTED_POSITIONS_FP)
        self.assertEqual(fps["calibrations"], EXPECTED_CALIBRATIONS_FP)
        compile_physical_schemas(registry=self.reg)
        for name in (
            "target_trajectory_samples",
            "target_trajectory_segments",
            "trajectory_gaps",
            "physical_metric_results",
        ):
            fp = contract_fingerprint(get_contract(name, 1, registry=self.reg))
            self.assertEqual(len(fp), 64)

    def test_02_policies_contract_only(self) -> None:
        assert_contract_only_policies(self.traj, self.metrics)
        self.assertFalse(self.traj["filter_resample_placeholders"]["enabled"])
        self.assertFalse(self.metrics["activity_coverage"]["composite_score_enabled"])
        self.assertTrue(self.metrics["safety"]["no_real_metric_computation"])
        self.assertEqual(policy_fingerprint(self.traj), self.traj_fp)

    def test_03_eligibility_gates(self) -> None:
        ok, _ = input_is_trajectory_eligible(eligible_candidate())
        self.assertTrue(ok)
        ok_p, reasons_p = input_is_trajectory_eligible(provisional_exclusion_candidate())
        self.assertFalse(ok_p)
        self.assertIn("PROVISIONAL_TARGET_EXCLUDED", reasons_p)
        ok_pred, reasons_pred = input_is_trajectory_eligible(predicted_exclusion_candidate())
        self.assertFalse(ok_pred)
        self.assertIn("PREDICTED_INTERPOLATED_EXCLUDED", reasons_pred)

    def test_04_bundle_and_gaps(self) -> None:
        b = confirmed_observed_bundle(self.traj_fp)
        vr = validate_physical_bundle(
            samples=b["target_trajectory_samples"],
            segments=b["target_trajectory_segments"],
            gaps=b["trajectory_gaps"],
            metric_results=b["physical_metric_results"],
            policy=self.metrics,
        )
        self.assertEqual(vr.status, "PASS")
        g = gap_bundle(self.traj_fp, gap_type="calibration_gap")
        self.assertFalse(g["gap_rows"][0]["allows_distance_bridge"])
        s = single_sample_segment_bundle(self.traj_fp)
        self.assertFalse(segment_metric_sufficient(s["segment_rows"][0]))

    def test_05_speed_sprint_heatmap_semantics(self) -> None:
        self.assertAlmostEqual(speed_delta_seconds(t0_us=0, t1_us=1_000_000), 1.0)
        self.assertFalse(
            sprint_from_single_spike(sample_count=1, duration_us=10, min_duration_us=1_000_000)
        )
        self.assertEqual(self.metrics["heatmap"]["weighting"], "time_weighted")
        self.assertTrue(self.metrics["activity_coverage"]["low_coverage_is_not_low_activity"])

    def test_06_zones_and_attack_direction(self) -> None:
        assert_zone_name_allowed("goal_a_third")
        with self.assertRaises(PhysicalContractError):
            assert_zone_name_allowed("final_third")
        self.assertFalse(progression_enabled(attack_direction="unknown", policy_enabled=True))
        self.assertEqual(self.traj["coordinate_frame"]["attack_direction_default"], "unknown")

    def test_07_zero_null_not_evaluable(self) -> None:
        self.assertEqual(
            distinguish_zero_null_not_evaluable(value=0.0, status="computed", observed=True),
            "zero",
        )
        self.assertEqual(
            distinguish_zero_null_not_evaluable(value=None, status="not_evaluable", observed=True),
            "not_evaluable",
        )
        stub = contract_stub_result(metric_name="distance", unit="m")
        self.assertIsNone(stub["value"])
        self.assertEqual(stub["status"], "contract_stub")
        self.assertEqual(metric_definition("speed")["canonical_unit"], "m_s")

    def test_08_request_receipt_evaluation(self) -> None:
        b = confirmed_observed_bundle(self.traj_fp)
        req = build_synthetic_request(
            run_id=b["run_id"],
            video_id=b["video_id"],
            target_player_id=b["target_player_id"],
            trajectory_policy_fingerprint=self.traj_fp,
            metrics_policy_fingerprint=self.met_fp,
            pitch_template_fingerprint="a" * 64,
        )
        validate_request_payload(req)
        receipt = build_synthetic_receipt(
            run_id=b["run_id"],
            video_id=b["video_id"],
            target_player_id=b["target_player_id"],
            trajectory_policy_fingerprint=self.traj_fp,
            metrics_policy_fingerprint=self.met_fp,
            samples=b["sample_rows"],
            segments=b["segment_rows"],
            gaps=b["gap_rows"],
            metric_results=b["metric_rows"],
        )
        validate_receipt_payload(receipt)
        self.assertEqual(
            recount_receipt_counts(
                samples=b["sample_rows"],
                segments=b["segment_rows"],
                gaps=b["gap_rows"],
                receipt=receipt,
            ),
            [],
        )
        ev = evaluate_physical_metrics(has_reviewed_ground_truth=False)
        self.assertEqual(ev.ground_truth_evaluation_status, NOT_EVALUATED_PHYSICAL)
        load_physical_json_schema("physical_metric_evaluation")

    def test_09_atomic_no_overwrite(self) -> None:
        import tempfile
        from pathlib import Path

        b = confirmed_observed_bundle(self.traj_fp)
        receipt = build_synthetic_receipt(
            run_id=b["run_id"],
            video_id=b["video_id"],
            target_player_id=b["target_player_id"],
            trajectory_policy_fingerprint=self.traj_fp,
            metrics_policy_fingerprint=self.met_fp,
            samples=b["sample_rows"],
            segments=b["segment_rows"],
            gaps=b["gap_rows"],
            metric_results=b["metric_rows"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            write_json_record(path, receipt, overwrite=False)
            with self.assertRaises(RecordError):
                write_json_record(path, receipt, overwrite=False)


if __name__ == "__main__":
    unittest.main()
