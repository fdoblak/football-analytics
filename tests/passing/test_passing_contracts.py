"""Stage 11A–11D passing contract and baseline tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.data.compiler import list_contracts
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.passing.attack_direction import resolve_attack_direction
from football_analytics.passing.contracts import (
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    PASSING_ARROW_CONTRACTS,
    assert_frozen_upstream_fingerprints,
    assert_passing_contracts_registered,
)
from football_analytics.passing.evaluation import NOT_EVALUATED_PASSING, evaluate_passing
from football_analytics.passing.fixtures import single_target_pass_bundle
from football_analytics.passing.metrics_service import compute_passing_metrics
from football_analytics.passing.pass_fixtures import load_fixture
from football_analytics.passing.pass_service import compute_pass_reception
from football_analytics.passing.pipeline_fixtures import load_pipeline_fixture
from football_analytics.passing.pipeline_service import integrate_passing
from football_analytics.passing.policy import (
    assert_contract_only_policy,
    load_passing_policy,
    policy_fingerprint,
)
from football_analytics.passing.semantics import (
    owner_change_alone_is_completed_pass,
    penalty_presence_is_box_touch,
)
from football_analytics.passing.validation import validate_passing_bundle


class PassingContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = default_project_root()
        cls.reg = load_schema_registry(default_registry_path(), project_root=cls.root)

    def test_registry_count(self) -> None:
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        self.assertEqual(EXPECTED_REGISTRY_CONTRACT_COUNT, 35)

    def test_passing_contracts_registered(self) -> None:
        assert_passing_contracts_registered(registry=self.reg)
        names = set(list_contracts(registry=self.reg))
        for n in PASSING_ARROW_CONTRACTS:
            self.assertIn(n, names)

    def test_frozen_upstream(self) -> None:
        assert_frozen_upstream_fingerprints(registry=self.reg)

    def test_policy_and_bundle(self) -> None:
        policy = load_passing_policy()
        assert_contract_only_policy(policy)
        pol_fp = policy_fingerprint(policy)
        bundle = single_target_pass_bundle(pol_fp)
        vr = validate_passing_bundle(
            passes=bundle["pass_candidates"],
            receptions=bundle["reception_candidates"],
            outcomes=bundle["pass_outcomes"],
            progression=bundle["ball_progression_segments"],
            touches=bundle["target_ball_touches"],
            policy=policy,
            expected_run_id=bundle["run_id"],
            expected_video_id=bundle["video_id"],
        )
        self.assertEqual(vr.status, "PASS")

    def test_semantics(self) -> None:
        self.assertFalse(owner_change_alone_is_completed_pass(owner_changed=True))
        self.assertFalse(penalty_presence_is_box_touch(in_penalty=True))

    def test_not_evaluated(self) -> None:
        report = evaluate_passing()
        self.assertEqual(report.ground_truth_evaluation_status, NOT_EVALUATED_PASSING)


class PassingBaselineTests(unittest.TestCase):
    def test_completed_pass_service(self) -> None:
        fx = load_fixture("completed_pass")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_pass_reception(
                output_dir=Path(tmp),
                transitions=fx["transitions"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertEqual(r.outcomes[0]["outcome"], "completed")
            self.assertNotEqual(r.outcomes[0]["outcome_state"], "confirmed")

    def test_owner_change_alone_not_completed(self) -> None:
        fx = load_fixture("owner_change_alone")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_pass_reception(
                output_dir=Path(tmp),
                transitions=fx["transitions"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertEqual(r.outcomes[0]["outcome"], "not_evaluable")
            self.assertFalse(r.passes[0]["implies_completed_pass"])

    def test_attack_direction_conflict(self) -> None:
        ev = resolve_attack_direction(
            run_id="run_test_conflict_xxxxxx",
            video_id="video_synth_01",
            config_direction="toward_goal_a",
            manual_direction="toward_goal_b",
        )
        self.assertEqual(ev["attack_direction"], "unknown")
        self.assertTrue(ev["conflict"])
        self.assertFalse(ev["invented"])

    def test_metrics_directional_not_evaluable(self) -> None:
        fx = load_fixture("completed_pass")
        with tempfile.TemporaryDirectory() as tmp:
            pr = compute_pass_reception(
                output_dir=Path(tmp) / "b",
                transitions=fx["transitions"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            mr = compute_passing_metrics(
                output_dir=Path(tmp) / "m",
                passes=pr.passes,
                receptions=pr.receptions,
                outcomes=pr.outcomes,
                run_id=pr.summary.get("run_id"),
                video_id=pr.summary.get("video_id"),
            )
            self.assertTrue(mr.accepted)
            self.assertEqual(mr.metrics["progression_1_to_2"]["status"], "not_evaluable")
            self.assertEqual(mr.metrics["evaluation_status"], NOT_EVALUATED_PASSING)

    def test_pipeline_fuse(self) -> None:
        fx = load_pipeline_fixture("completed_with_box")
        with tempfile.TemporaryDirectory() as tmp:
            r = integrate_passing(
                output_dir=Path(tmp),
                transitions=fx["transitions"],
                touch_inputs=fx.get("touch_inputs"),
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertIn(
                "REAL FOOTBALL ACCURACY NOT YET VALIDATED", str(r.summary.get("gate_hint"))
            )
            self.assertFalse(r.summary.get("real_football_accuracy_validated"))


if __name__ == "__main__":
    unittest.main()
