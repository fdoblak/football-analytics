#!/usr/bin/env python3
"""Stage 10A human-ball interaction / possession contract tests."""

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
from football_analytics.interaction.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_PHYSICAL_METRIC_RESULTS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_frozen_upstream_fingerprints,
    assert_interaction_contracts_registered,
    compile_interaction_schemas,
    interaction_schema_fingerprints,
    load_interaction_json_schema,
)
from football_analytics.interaction.eligibility import (
    missing_ball_means_no_possession,
    pitch_distance_usable,
)
from football_analytics.interaction.evaluation import (
    NOT_EVALUATED_INTERACTION,
    evaluate_human_ball_interaction,
)
from football_analytics.interaction.fixtures import (
    contested_two_player_bundle,
    coverage_example,
    nearest_not_possession_rows,
    single_player_proximity_bundle,
)
from football_analytics.interaction.policy import (
    assert_contract_only_policy,
    load_interaction_policy,
    policy_fingerprint,
)
from football_analytics.interaction.receipt import (
    build_synthetic_quality,
    build_synthetic_receipt,
    build_synthetic_request,
    validate_quality_payload,
    validate_receipt_payload,
    validate_request_payload,
)
from football_analytics.interaction.semantics import (
    hard_gap_allows_possession_carry,
    nearest_player_is_possession,
    penalty_presence_is_box_touch,
    proximity_is_contact,
)
from football_analytics.interaction.validation import validate_interaction_bundle


class HumanBallInteractionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = default_project_root()
        self.reg = load_schema_registry(default_registry_path(), project_root=self.root)
        self.policy = load_interaction_policy(project_root=self.root)
        self.pol_fp = policy_fingerprint(self.policy)

    def test_01_registry_and_frozen_upstream(self) -> None:
        assert_interaction_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_frozen_upstream_fingerprints(registry=self.reg)
        fps = interaction_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(fps["physical_metric_results"], EXPECTED_PHYSICAL_METRIC_RESULTS_FP)
        compile_interaction_schemas(registry=self.reg)
        for name in (
            "human_ball_proximity",
            "ball_contact_candidates",
            "possession_hypotheses",
        ):
            fp = contract_fingerprint(get_contract(name, 1, registry=self.reg))
            self.assertEqual(len(fp), 64)

    def test_02_policy_contract_only(self) -> None:
        assert_contract_only_policy(self.policy)
        self.assertTrue(self.policy["no_real_interaction_inference"])
        self.assertEqual(self.policy["automatic_baseline"]["max_state"], "provisional")
        self.assertFalse(missing_ball_means_no_possession(policy=self.policy))
        self.assertEqual(policy_fingerprint(self.policy), self.pol_fp)

    def test_03_nearest_not_possession_and_separations(self) -> None:
        self.assertFalse(nearest_player_is_possession(is_nearest=True))
        self.assertFalse(proximity_is_contact(proximity_only=True))
        self.assertFalse(penalty_presence_is_box_touch(in_penalty=True))
        self.assertFalse(hard_gap_allows_possession_carry(hard_gap=True))
        nn = nearest_not_possession_rows(self.pol_fp)
        self.assertTrue(nn["proximity_rows"][0]["is_nearest_human"])
        self.assertFalse(nn["proximity_rows"][0]["nearest_implies_possession"])

    def test_04_bundle_and_contested(self) -> None:
        b = single_player_proximity_bundle(self.pol_fp)
        vr = validate_interaction_bundle(
            proximity=b["human_ball_proximity"],
            contacts=b["ball_contact_candidates"],
            possessions=b["possession_hypotheses"],
            policy=self.policy,
            expected_run_id=b["run_id"],
            expected_video_id=b["video_id"],
        )
        self.assertEqual(vr.status, "PASS", vr.errors)
        c = contested_two_player_bundle(self.pol_fp)
        vr2 = validate_interaction_bundle(
            proximity=c["human_ball_proximity"],
            contacts=c["ball_contact_candidates"],
            possessions=c["possession_hypotheses"],
            policy=self.policy,
        )
        self.assertEqual(vr2.status, "PASS", vr2.errors)
        self.assertEqual(c["possession_rows"][0]["possession_state"], "contested")

    def test_05_airborne_blocks_pitch(self) -> None:
        b = single_player_proximity_bundle(self.pol_fp)
        row = dict(b["proximity_rows"][0])
        row["ball_air_state"] = "unknown"
        ok, reasons = pitch_distance_usable(row)
        self.assertFalse(ok)
        self.assertIn("AIRBORNE_UNKNOWN_BLOCKS_PITCH", reasons)

    def test_06_request_receipt_quality_evaluation(self) -> None:
        b = single_player_proximity_bundle(self.pol_fp)
        req = build_synthetic_request(
            run_id=b["run_id"],
            video_id=b["video_id"],
            interaction_policy_fingerprint=self.pol_fp,
        )
        validate_request_payload(req)
        cov = coverage_example()
        receipt = build_synthetic_receipt(
            run_id=b["run_id"],
            video_id=b["video_id"],
            interaction_policy_fingerprint=self.pol_fp,
            proximity=b["proximity_rows"],
            contacts=b["contact_rows"],
            possessions=b["possession_rows"],
            coverage_summary=cov,
        )
        validate_receipt_payload(receipt)
        quality = build_synthetic_quality(
            run_id=b["run_id"],
            video_id=b["video_id"],
            coverage=cov,
            interaction_policy_fingerprint=self.pol_fp,
        )
        validate_quality_payload(quality)
        for name in (
            "human_ball_interaction_request",
            "human_ball_interaction_run_receipt",
            "human_ball_interaction_evaluation",
            "human_ball_interaction_quality",
            "human_ball_interaction_manual_review_queue",
        ):
            schema = load_interaction_json_schema(name)
            self.assertEqual(schema["type"], "object")
        ev = evaluate_human_ball_interaction(has_reviewed_ground_truth=False)
        self.assertEqual(ev.ground_truth_evaluation_status, NOT_EVALUATED_INTERACTION)
        self.assertTrue(all(v is None for v in ev.metrics.values()))

    def test_07_no_overwrite(self) -> None:
        import tempfile
        from pathlib import Path

        b = single_player_proximity_bundle(self.pol_fp)
        req = build_synthetic_request(
            run_id=b["run_id"],
            video_id=b["video_id"],
            interaction_policy_fingerprint=self.pol_fp,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "req.json"
            write_json_record(path, req, overwrite=False)
            with self.assertRaises(RecordError):
                write_json_record(path, req, overwrite=False)


if __name__ == "__main__":
    unittest.main()
