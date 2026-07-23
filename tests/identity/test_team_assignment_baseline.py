"""Stage 7C anonymous team assignment baseline tests (synthetic)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.data.compiler import get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.identity.appearance_reid_config import (
    appearance_reid_config_fingerprint,
    default_appearance_reid_config_path,
    load_appearance_reid_config,
)
from football_analytics.identity.appearance_reid_service import build_profiles_from_bundle
from football_analytics.identity.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_JERSEY_OBSERVATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    EXPECTED_TEAM_ASSIGNMENTS_FP,
    EXPECTED_TRACK_LIFECYCLE_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    assert_frozen_upstream_fingerprints,
    assert_identity_contracts_registered,
    identity_schema_fingerprints,
)
from football_analytics.identity.policy import decide_assignment_status, load_identity_policy
from football_analytics.identity.team_assignment_config import (
    default_team_assignment_config_path,
    load_team_assignment_config,
    team_assignment_config_fingerprint,
)
from football_analytics.identity.team_assignment_evaluation import (
    NOT_EVALUATED_TEAM_ASSIGNMENT,
    evaluate_team_assignment,
    permutation_matched_accuracy,
)
from football_analytics.identity.team_assignment_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    fixture_cluster_collapse,
    fixture_cross_shot_alignment,
    fixture_cross_video_reject,
    fixture_goalkeeper_different_kit,
    fixture_insufficient_seeds,
    fixture_similar_kit_hard,
    fixture_third_color_outlier,
    fixture_two_distinct_teams,
    fixture_unknown_role,
    fixture_with_referee,
    fixture_with_staff,
)
from football_analytics.identity.team_assignment_service import run_team_classify
from football_analytics.identity.team_clustering import (
    collect_seed_tracks,
    fit_two_team_clusters,
    team_feature_vector,
)

ROOT = default_project_root()


class TeamAssignmentBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_team_assignment_config(default_team_assignment_config_path())
        cls.cfg_fp = team_assignment_config_fingerprint(cls.cfg)
        cls.app_cfg = load_appearance_reid_config(default_appearance_reid_config_path())
        cls.app_fp = appearance_reid_config_fingerprint(cls.app_cfg)
        cls.policy = load_identity_policy()
        cls.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        assert_runtime_root()

    def _profiles(self, bundle: dict):
        return build_profiles_from_bundle(
            bundle=bundle, config=self.app_cfg, config_fingerprint=self.app_fp
        )

    def test_01_fingerprint_regression(self) -> None:
        assert_identity_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_frozen_upstream_fingerprints(registry=self.reg)
        fps = identity_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["team_assignments"], EXPECTED_TEAM_ASSIGNMENTS_FP)
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(fps["track_observations"], EXPECTED_TRACK_OBSERVATIONS_FP)
        self.assertEqual(fps["track_summaries"], EXPECTED_TRACK_SUMMARIES_FP)
        self.assertEqual(fps["track_lifecycle"], EXPECTED_TRACK_LIFECYCLE_FP)
        self.assertEqual(fps["jersey_observations"], EXPECTED_JERSEY_OBSERVATIONS_FP)
        self.assertEqual(
            contract_fingerprint(get_contract("team_assignments", 1, registry=self.reg)),
            EXPECTED_TEAM_ASSIGNMENTS_FP,
        )

    def test_02_player_only_seed_exclusions(self) -> None:
        for fx, role_tid, role in (
            (fixture_with_referee, 99, "referee"),
            (fixture_with_staff, 88, "staff"),
            (fixture_goalkeeper_different_kit, 77, "goalkeeper"),
            (fixture_unknown_role, 55, "unknown"),
        ):
            b = fx()
            profiles, _, _ = self._profiles(b)
            seeds, rejected = collect_seed_tracks(
                profiles, config=self.cfg, role_by_track=b["role_by_track"]
            )
            seed_ids = {s.track_id for s in seeds}
            self.assertNotIn(role_tid, seed_ids, msg=role)
            self.assertTrue(any(r["track_id"] == role_tid for r in rejected), msg=role)

    def test_03_deterministic_clustering_and_label_order(self) -> None:
        b = fixture_two_distinct_teams()
        profiles, _, _ = self._profiles(b)
        seeds, _ = collect_seed_tracks(profiles, config=self.cfg, role_by_track=b["role_by_track"])
        m1 = fit_two_team_clusters(seeds, config=self.cfg)
        m2 = fit_two_team_clusters(seeds, config=self.cfg)
        self.assertEqual(m1.status, "ok")
        self.assertEqual(m1.centroid_fingerprints, m2.centroid_fingerprints)
        self.assertEqual(m1.label_order, ("team_a", "team_b"))
        self.assertEqual(set(m1.centroids.keys()), {"team_a", "team_b"})
        # Fingerprint ascending defines team_a
        self.assertLessEqual(m1.centroid_fingerprints["team_a"], m1.centroid_fingerprints["team_b"])

    def test_04_two_team_separation(self) -> None:
        b = fixture_two_distinct_teams()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "two",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        self.assertTrue(r.accepted, r.error_code)
        model = r.summary["model"]
        self.assertEqual(model.status, "ok")
        assigned = [d for d in r.summary["decisions"] if d.status == "assigned"]
        self.assertGreaterEqual(len(assigned), 4)
        teams = {d.team_id for d in assigned}
        self.assertEqual(teams, {"team_a", "team_b"})
        for row in r.summary["assignment_rows"]:
            self.assertIn(row["team_id"], {"team_a", "team_b", "unknown"})
            self.assertEqual(row["team_role"], "unknown")

    def test_05_similar_kit_or_collapse_abstain(self) -> None:
        for fx in (fixture_similar_kit_hard, fixture_cluster_collapse):
            b = fx()
            with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
                r = run_team_classify(
                    output_dir=Path(td) / "abs",
                    config=self.cfg,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=b,
                )
            self.assertTrue(r.accepted, r.error_code)
            model = r.summary["model"]
            # Similar kits / collapse → insufficient or all unknown/ambiguous
            if model.status == "ok":
                self.assertTrue(
                    all(
                        d.team_id == "unknown" or d.status == "ambiguous"
                        for d in r.summary["decisions"]
                    )
                    or model.separation is not None
                )
            else:
                self.assertIn(
                    model.status,
                    {"insufficient_team_evidence"},
                )
                self.assertTrue(
                    any(
                        x in model.reason_codes
                        for x in (
                            "SIMILAR_KIT_ABSTAIN",
                            "LOW_SEPARATION",
                            "INSUFFICIENT_SEPARATION",
                            "CLUSTER_COLLAPSE",
                            "HIGH_INTRA_SPREAD",
                        )
                    )
                    or model.status == "insufficient_team_evidence"
                )

    def test_06_insufficient_seeds(self) -> None:
        b = fixture_insufficient_seeds()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "few",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        self.assertTrue(r.accepted)
        self.assertEqual(r.summary["model"].status, "insufficient_team_evidence")
        self.assertTrue(
            all(d.team_id == "unknown" for d in r.summary["decisions"] if d.role == "player")
        )

    def test_07_outlier_third_color(self) -> None:
        b = fixture_third_color_outlier()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "out",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        self.assertTrue(r.accepted, r.error_code)
        by_tid = {d.track_id: d for d in r.summary["decisions"]}
        self.assertEqual(by_tid[66].team_id, "unknown")

    def test_08_role_specials(self) -> None:
        for fx, tid, expect_role, expect_status in (
            (fixture_with_referee, 99, "official", "not_eligible"),
            (fixture_with_staff, 88, "official", "not_eligible"),
            (fixture_goalkeeper_different_kit, 77, "unknown", "unknown"),
        ):
            b = fx()
            with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
                r = run_team_classify(
                    output_dir=Path(td) / f"role{tid}",
                    config=self.cfg,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=b,
                )
            self.assertTrue(r.accepted, r.error_code)
            d = next(x for x in r.summary["decisions"] if x.track_id == tid)
            self.assertEqual(d.team_id, "unknown")
            self.assertEqual(d.team_role, expect_role)
            self.assertEqual(d.status, expect_status)
            if tid == 77:
                self.assertIn("GOALKEEPER_NO_AUTO_TEAM_FROM_KIT", d.reason_codes)

    def test_09_unknown_role_candidate_only(self) -> None:
        b = fixture_unknown_role()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "unk",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        self.assertTrue(r.accepted)
        d = next(x for x in r.summary["decisions"] if x.track_id == 55)
        self.assertIn(d.status, {"candidate", "unknown", "ambiguous"})
        self.assertNotEqual(d.status, "assigned")

    def test_10_team_switch_conflict(self) -> None:
        b = fixture_two_distinct_teams()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r0 = run_team_classify(
                output_dir=Path(td) / "sw0",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
            self.assertTrue(r0.accepted)
            d0 = next(x for x in r0.summary["decisions"] if x.track_id == 1)
            self.assertIn(d0.team_id, {"team_a", "team_b"})
            opposite = "team_b" if d0.team_id == "team_a" else "team_a"
            b2 = dict(b)
            b2["prior_team_by_track"] = {1: opposite}
            r = run_team_classify(
                output_dir=Path(td) / "sw1",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b2,
            )
        self.assertTrue(r.accepted)
        d = next(x for x in r.summary["decisions"] if x.track_id == 1)
        self.assertEqual(d.status, "conflict")
        self.assertIn("TEAM_SWITCH_CONFLICT", d.reason_codes)

    def test_11_team_evidence_no_auto_confirm(self) -> None:
        b = fixture_two_distinct_teams()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "ev",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        self.assertTrue(r.accepted)
        for er in r.summary["evidence_rows"]:
            self.assertEqual(er["evidence_type"], "team_assignment")
            self.assertNotIn(er["reliability_tier"], {"strong", "manual_verified"})
            st, reasons = decide_assignment_status([er], policy=self.policy)
            self.assertEqual(st, "candidate")
            self.assertIn("TEAM_ALONE_INSUFFICIENT", reasons)

    def test_12_permutation_invariant_evaluator(self) -> None:
        pred = ["team_a", "team_a", "team_b", "team_b"]
        truth = ["team_b", "team_b", "team_a", "team_a"]  # swapped labels
        acc = permutation_matched_accuracy(pred, truth)
        self.assertEqual(acc, 1.0)
        report = evaluate_team_assignment(has_reviewed_ground_truth=False)
        self.assertEqual(report.ground_truth_evaluation_status, NOT_EVALUATED_TEAM_ASSIGNMENT)

    def test_13_cross_shot_and_cross_video(self) -> None:
        b = fixture_cross_shot_alignment()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "shot",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        self.assertTrue(r.accepted, r.error_code)
        xv = fixture_cross_video_reject()
        self.assertFalse(self.cfg["clustering"]["cross_video_auto_transfer"])
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r2 = run_team_classify(
                output_dir=Path(td) / "xv",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=xv,
            )
        self.assertTrue(r2.accepted)
        self.assertFalse(r2.summary["receipt"]["cross_video_auto_transfer"])

    def test_14_no_overwrite_and_failure_cleanup(self) -> None:
        b = fixture_two_distinct_teams()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            out = Path(td) / "ow"
            r1 = run_team_classify(
                output_dir=out, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
            self.assertTrue(r1.accepted)
            r2 = run_team_classify(
                output_dir=out, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
            self.assertEqual(r2.error_code, "OVERWRITE_FORBIDDEN")
            fail_out = Path(td) / "fail"
            r3 = run_team_classify(
                output_dir=fail_out,
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
                inject_failure=True,
            )
            self.assertFalse(r3.accepted)
            self.assertFalse((fail_out / "team_assignments.parquet").exists())

    def test_15_assignment_rows_schema_fields(self) -> None:
        b = fixture_two_distinct_teams()
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_team_classify(
                output_dir=Path(td) / "sch",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
            )
        rows = r.summary["assignment_rows"]
        self.assertTrue(rows)
        for row in rows:
            for key in (
                "run_id",
                "video_id",
                "assignment_id",
                "track_id",
                "start_frame_index",
                "end_frame_index",
                "team_id",
                "team_role",
                "source",
                "quality_flags",
            ):
                self.assertIn(key, row)
            self.assertLessEqual(row["start_frame_index"], row["end_frame_index"])
            self.assertNotIn(row["team_id"], {"home", "away", "Galatasaray", "Fenerbahce"})

    def test_16_color_feature_excludes_edge_when_configured(self) -> None:
        b = fixture_two_distinct_teams()
        profiles, _, _ = self._profiles(b)
        v = team_feature_vector(profiles[0].embedding, config=self.cfg)
        self.assertEqual(len(v), 64)


if __name__ == "__main__":
    unittest.main()
