"""Stage 14 orchestration / review / report tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.orchestration.config import (
    load_pipeline_config,
    pipeline_config_fingerprint,
)
from football_analytics.orchestration.contracts import (
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    GATE_HINT,
    STAGE_CHAIN,
    assert_registry_contract_count_unchanged,
)
from football_analytics.orchestration.fixtures import synthetic_pipeline_request
from football_analytics.orchestration.planner import plan_pipeline
from football_analytics.orchestration.report.builder import (
    build_single_player_report,
    synthetic_metric_bundle,
)
from football_analytics.orchestration.review.hub import (
    apply_decision,
    assert_package_cas,
    build_decision,
    prepare_review_package,
    revoke_decision,
)
from football_analytics.orchestration.runner import run_pipeline
from football_analytics.orchestration.types import OrchestrationError, StaleArtifactError
from football_analytics.visualization.report_renderer import render_single_player_summary_png


class Stage14Tests(unittest.TestCase):
    def test_00_registry_unchanged(self) -> None:
        self.assertEqual(
            assert_registry_contract_count_unchanged(), EXPECTED_REGISTRY_CONTRACT_COUNT
        )

    def test_01_plan_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = synthetic_pipeline_request(output_directory=tmp)
            plan = plan_pipeline(req)
            self.assertEqual([s["name"] for s in plan["stages"]], list(STAGE_CHAIN))
            self.assertEqual(len(plan["plan_fingerprint"]), 64)

    def test_02_run_light_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = synthetic_pipeline_request(output_directory=tmp, force_restart=True)
            res = run_pipeline(req, light_stubs_only=True)
            self.assertIn(res.overall_status, {"succeeded", "partial"})
            self.assertTrue(Path(res.status_path).is_file())
            self.assertIsNotNone(res.report_json_path)
            self.assertTrue(Path(str(res.report_json_path)).is_file())
            self.assertIn("SINGLE PLAYER PIPELINE ACTIVE", GATE_HINT)
            self.assertFalse(res.summary["user_video_mutated"])

    def test_03_cancel_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = synthetic_pipeline_request(
                output_directory=tmp,
                cancel_requested=True,
                force_restart=True,
                run_id="run_stage14_cancel_test",
                request_id="req_cancel_test",
            )
            res = run_pipeline(req, light_stubs_only=True)
            self.assertEqual(res.overall_status, "cancelled")
            self.assertIsNotNone(res.cancellation_receipt_path)

    def test_04_review_cas_and_revoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = prepare_review_package(
                package_id="pkg_t",
                run_id="run_review_test01",
                video_id="vid_t",
                target_player_id="target_player_a",
                output_dir=root,
            )
            with self.assertRaises(OrchestrationError):
                bad = build_decision(
                    decision_id="dec_bad",
                    package=pkg,
                    domain="identity",
                    item_id="identity_item_001",
                    decision="confirm",
                    reviewer_id="auto",
                    reason="nope",
                )
                apply_decision(decision=bad, package=pkg, output_dir=root)
            dec = build_decision(
                decision_id="dec_ok",
                package=pkg,
                domain="calibration",
                item_id="calibration_item_001",
                decision="keep_provisional",
                reviewer_id="reviewer_a",
                reason="ok",
            )
            apply_decision(decision=dec, package=pkg, output_dir=root)
            stale = dict(dec)
            stale["expected_package_hash"] = "f" * 64
            with self.assertRaises(StaleArtifactError):
                assert_package_cas(stale, package=pkg)
            revoke_decision(
                previous=dec,
                package=pkg,
                output_dir=root,
                reviewer_id="reviewer_a",
                reason="revoke",
                new_decision_id="dec_rev",
            )

    def test_05_report_no_team_summary(self) -> None:
        report = build_single_player_report(
            run_id="run_report_test01",
            git_commit="b" * 40,
            target_player_id="target_player_a",
            display_name="A",
            match_id="m1",
            video_id="v1",
            metrics=synthetic_metric_bundle(),
        )
        self.assertTrue(report["team_summary_forbidden"])
        self.assertNotIn("team_summary", report)
        self.assertTrue(report["not_evaluable_metric_ids"])
        self.assertEqual(len(report["reproducibility_fingerprint"]), 64)

    def test_06_render_and_delete(self) -> None:
        report = build_single_player_report(
            run_id="run_render_test01",
            git_commit="c" * 40,
            target_player_id="target_player_a",
            display_name="A",
            match_id="m1",
            video_id="v1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.png"
            render_single_player_summary_png(report, out)
            self.assertTrue(out.is_file())
            out.unlink()
            self.assertFalse(out.exists())

    def test_07_config_fingerprint_stable(self) -> None:
        a = pipeline_config_fingerprint(load_pipeline_config())
        b = pipeline_config_fingerprint(load_pipeline_config())
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
