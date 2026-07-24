"""Stage 12A–12E duels contract and baseline tests."""

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
from football_analytics.duels.aerial_fixtures import load_fixture as load_aerial_fixture
from football_analytics.duels.aerial_service import compute_aerial_clearance
from football_analytics.duels.contracts import (
    DUELS_ARROW_CONTRACTS,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_duels_contracts_registered,
    assert_frozen_upstream_fingerprints,
)
from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS, evaluate_duels
from football_analytics.duels.fixtures import single_target_duels_bundle
from football_analytics.duels.ground_fixtures import load_fixture as load_ground_fixture
from football_analytics.duels.ground_service import compute_ground_family
from football_analytics.duels.pipeline_fixtures import load_pipeline_fixture
from football_analytics.duels.pipeline_service import integrate_duels
from football_analytics.duels.policy import (
    assert_contract_only_policy,
    load_duels_policy,
    policy_fingerprint,
)
from football_analytics.duels.semantics import (
    long_ball_alone_is_clearance,
    nearby_opponent_alone_is_take_on,
    nearest_switch_alone_is_duel_outcome,
)
from football_analytics.duels.take_on_fixtures import load_fixture as load_take_on_fixture
from football_analytics.duels.take_on_service import compute_take_ons
from football_analytics.duels.validation import validate_duels_bundle


class DuelsContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = default_project_root()
        cls.reg = load_schema_registry(default_registry_path(), project_root=cls.root)

    def test_registry_count(self) -> None:
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        self.assertEqual(EXPECTED_REGISTRY_CONTRACT_COUNT, 42)

    def test_duels_contracts_registered(self) -> None:
        assert_duels_contracts_registered(registry=self.reg)
        names = set(list_contracts(registry=self.reg))
        for n in DUELS_ARROW_CONTRACTS:
            self.assertIn(n, names)

    def test_frozen_upstream(self) -> None:
        assert_frozen_upstream_fingerprints(registry=self.reg)

    def test_policy_and_bundle(self) -> None:
        policy = load_duels_policy()
        assert_contract_only_policy(policy)
        pol_fp = policy_fingerprint(policy)
        bundle = single_target_duels_bundle(pol_fp)
        vr = validate_duels_bundle(
            take_ons=bundle["take_on_attempts"],
            ground_duels=bundle["ground_duel_candidates"],
            aerial_duels=bundle["aerial_duel_candidates"],
            tackles=bundle["tackle_events"],
            recoveries=bundle["recovery_events"],
            turnovers=bundle["turnover_events"],
            clearances=bundle["clearance_events"],
            policy=policy,
            expected_run_id=bundle["run_id"],
            expected_video_id=bundle["video_id"],
        )
        self.assertEqual(vr.status, "PASS")

    def test_semantics(self) -> None:
        self.assertFalse(nearby_opponent_alone_is_take_on(nearby_opponent_alone=True))
        self.assertFalse(nearest_switch_alone_is_duel_outcome(nearest_switch_alone=True))
        self.assertFalse(long_ball_alone_is_clearance(long_ball_alone=True))

    def test_not_evaluated(self) -> None:
        report = evaluate_duels()
        self.assertEqual(report.ground_truth_evaluation_status, NOT_EVALUATED_DUELS)


class DuelsBaselineTests(unittest.TestCase):
    def test_take_on_service(self) -> None:
        fx = load_take_on_fixture("successful_take_on")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_take_ons(
                output_dir=Path(tmp),
                contexts=fx["contexts"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertTrue(r.take_ons[0]["implies_take_on"])
            self.assertNotEqual(r.take_ons[0]["event_state"], "confirmed")

    def test_nearby_opponent_alone_not_take_on(self) -> None:
        fx = load_take_on_fixture("nearby_opponent_alone")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_take_ons(
                output_dir=Path(tmp),
                contexts=fx["contexts"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertFalse(r.take_ons[0]["implies_take_on"])

    def test_ground_family(self) -> None:
        fx = load_ground_fixture("contested_ground")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_ground_family(
                output_dir=Path(tmp),
                contexts=fx["contexts"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertGreaterEqual(len(r.ground_duels), 1)
            self.assertGreaterEqual(len(r.tackles), 1)

    def test_aerial_monocular(self) -> None:
        fx = load_aerial_fixture("monocular_aerial")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_aerial_clearance(
                output_dir=Path(tmp),
                contexts=fx["contexts"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            self.assertTrue(r.accepted)
            self.assertIsNone(r.aerial_duels[0]["exact_3d_height_m"])
            self.assertFalse(r.aerial_duels[0]["exact_3d_height_claimed"])

    def test_pipeline_fuse(self) -> None:
        fx = load_pipeline_fixture("full_package")
        with tempfile.TemporaryDirectory() as tmp:
            r = integrate_duels(
                output_dir=Path(tmp),
                take_on_contexts=fx.get("take_on_contexts"),
                ground_contexts=fx.get("ground_contexts"),
                aerial_contexts=fx.get("aerial_contexts"),
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
