"""Stage 13A–13E target events tests."""

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
from football_analytics.events.attack_direction import (
    resolve_match_attack_directions,
    resolve_period_attack_direction,
)
from football_analytics.events.camera_position import resolve_camera_position
from football_analytics.events.contracts import (
    EVENTS_ARROW_CONTRACTS,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_events_contracts_registered,
    assert_frozen_upstream_fingerprints,
)
from football_analytics.events.dedup import suppress_duplicate_events
from football_analytics.events.eligibility import live_event_eligible
from football_analytics.events.evaluation import NOT_EVALUATED_EVENTS, evaluate_events
from football_analytics.events.fixtures import (
    pipeline_fixture,
    replay_contexts_fixture,
    source_events_fixture,
)
from football_analytics.events.ledger_service import build_target_event_ledger
from football_analytics.events.metrics_service import compute_target_event_metrics
from football_analytics.events.pipeline_service import integrate_target_events
from football_analytics.events.replay_service import compute_replay_candidates
from football_analytics.passing.attack_direction import resolve_attack_direction


class EventsContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = default_project_root()
        cls.reg = load_schema_registry(default_registry_path(), project_root=cls.root)

    def test_registry_count(self) -> None:
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        self.assertEqual(EXPECTED_REGISTRY_CONTRACT_COUNT, 45)
        self.assertEqual(len(EVENTS_ARROW_CONTRACTS), 3)

    def test_registered_and_frozen(self) -> None:
        assert_events_contracts_registered(registry=self.reg)
        assert_frozen_upstream_fingerprints(registry=self.reg)

    def test_unknown_replay_blocks_live(self) -> None:
        self.assertFalse(live_event_eligible(replay_status="unknown"))
        self.assertTrue(live_event_eligible(replay_status="live"))

    def test_camera_position_supported_only(self) -> None:
        self.assertEqual(resolve_camera_position(view_family="main_broadcast"), "sideline")
        self.assertEqual(resolve_camera_position(view_family="crowd"), "unknown")


class EventsBaselinesTests(unittest.TestCase):
    def test_replay_uncertain_never_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_replay_candidates(
                output_dir=Path(tmp),
                contexts=replay_contexts_fixture("uncertain_blocks_live"),
            )
            self.assertTrue(r.accepted)
            self.assertTrue(all(not x["live_event_eligible"] for x in r.replays))

    def test_attack_conflict_and_half_flip(self) -> None:
        conflict = resolve_period_attack_direction(
            run_id="run_test_attack_direction_01",
            video_id="v",
            config_direction="toward_goal_a",
            manual_direction="toward_goal_b",
        )
        self.assertEqual(conflict["attack_direction"], "unknown")
        periods = resolve_match_attack_directions(
            run_id="run_test_attack_direction_01",
            video_id="v",
            periods=[
                {
                    "period_id": "period_1",
                    "half_id": "first_half",
                    "config_direction": "toward_goal_b",
                    "apply_half_boundary_flip": False,
                },
                {
                    "period_id": "period_2",
                    "half_id": "second_half",
                    "config_direction": "toward_goal_b",
                    "apply_half_boundary_flip": True,
                },
            ],
        )
        self.assertEqual(periods[1]["attack_direction"], "toward_goal_a")
        # Stage 11 wrapper still works
        stub = resolve_attack_direction(
            run_id="run_test_attack_direction_01",
            video_id="v",
            config_direction="toward_goal_a",
            manual_direction="toward_goal_a",
        )
        self.assertEqual(stub["attack_direction"], "toward_goal_a")
        self.assertNotIn("period_id", stub)

    def test_dedup_and_ledger(self) -> None:
        rows = suppress_duplicate_events(
            [
                {
                    "ledger_event_id": "a",
                    "event_family": "take_on",
                    "start_time_us": 0,
                    "end_time_us": 500_000,
                    "confidence": 0.9,
                    "target_relationship": "confirmed_target",
                    "reason_codes": [],
                },
                {
                    "ledger_event_id": "b",
                    "event_family": "take_on",
                    "start_time_us": 100_000,
                    "end_time_us": 600_000,
                    "confidence": 0.4,
                    "target_relationship": "confirmed_target",
                    "reason_codes": [],
                },
            ]
        )
        self.assertEqual(sum(1 for r in rows if r["suppressed_duplicate"]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            r = build_target_event_ledger(
                output_dir=Path(tmp), sources=source_events_fixture("duplicate_overlap")
            )
            self.assertTrue(r.accepted)
            self.assertGreaterEqual(int(r.summary["suppressed_count"]), 1)

    def test_metrics_and_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            led = build_target_event_ledger(
                output_dir=root / "led", sources=source_events_fixture("full_package")
            )
            met = compute_target_event_metrics(
                output_dir=root / "met",
                ledger=led.ledger,
                attack_direction_manual="toward_goal_b",
            )
            self.assertTrue(met.accepted)
            self.assertIn("pass_attempts", met.metrics["metrics"])
            fx = pipeline_fixture("full_package")
            pipe = integrate_target_events(
                output_dir=root / "pipe",
                sources=fx["sources"],
                replay_contexts=fx["replay_contexts"],
                attack_periods=fx["attack_periods"],
                run_id=fx["run_id"],
                video_id=fx["video_id"],
            )
            self.assertTrue(pipe.accepted)
            self.assertEqual(pipe.summary["evaluation_status"], NOT_EVALUATED_EVENTS)
            self.assertIn("TARGET EVENTS PIPELINE ACTIVE", str(pipe.summary["gate_hint"]))
            ev = evaluate_events()
            self.assertEqual(ev["evaluation_status"], NOT_EVALUATED_EVENTS)


if __name__ == "__main__":
    unittest.main()
