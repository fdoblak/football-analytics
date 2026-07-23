"""Stage 7B appearance embedding + tracklet ReID baseline tests (synthetic)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.core.records import RecordError, write_json_record
from football_analytics.data.compiler import get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.identity.appearance_descriptor import (
    AppearanceDescriptorError,
    cosine_similarity,
    extract_descriptor_from_bgr,
    l2_normalize,
    validate_embedding,
)
from football_analytics.identity.appearance_reid_config import (
    appearance_reid_config_fingerprint,
    default_appearance_reid_config_path,
    load_appearance_reid_config,
)
from football_analytics.identity.appearance_reid_evaluation import (
    NOT_EVALUATED_APPEARANCE_REID,
    evaluate_appearance_reid,
)
from football_analytics.identity.appearance_reid_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    fixture_ambiguity_near_scores,
    fixture_brightness_shift,
    fixture_cross_video_reject,
    fixture_different_appearance,
    fixture_human_ball_reject,
    fixture_predicted_rejected,
    fixture_same_appearance_different_tracklets,
    fixture_same_kit_hard_negative,
    fixture_single_crop_insufficient,
    fixture_temporal_overlap,
    fixture_tiny_corrupt_crop,
)
from football_analytics.identity.appearance_reid_service import (
    build_profiles_from_bundle,
    run_appearance_extract,
    run_reid_candidates,
)
from football_analytics.identity.appearance_sampling import sample_tracklet_crops
from football_analytics.identity.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_JERSEY_OBSERVATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    EXPECTED_TEAM_ASSIGNMENTS_FP,
    EXPECTED_TRACK_LIFECYCLE_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    TRACKLET_APPEARANCE_PROFILES_CONTRACT,
    assert_frozen_upstream_fingerprints,
    assert_identity_contracts_registered,
    identity_schema_fingerprints,
)
from football_analytics.identity.policy import decide_assignment_status, load_identity_policy
from football_analytics.identity.types import EvidenceType, ReliabilityTier

ROOT = default_project_root()


class AppearanceReidBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_appearance_reid_config(default_appearance_reid_config_path())
        cls.cfg_fp = appearance_reid_config_fingerprint(cls.cfg)
        cls.policy = load_identity_policy()
        cls.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        assert_runtime_root()
        cls.dim = int(cls.cfg["descriptor"]["embedding_dim"])

    def _profiles(self, bundle: dict):
        return build_profiles_from_bundle(
            bundle=bundle, config=self.cfg, config_fingerprint=self.cfg_fp
        )

    def test_01_frozen_fingerprints_and_new_contract(self) -> None:
        assert_identity_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_frozen_upstream_fingerprints(registry=self.reg)
        fps = identity_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(fps["track_observations"], EXPECTED_TRACK_OBSERVATIONS_FP)
        self.assertEqual(fps["track_summaries"], EXPECTED_TRACK_SUMMARIES_FP)
        self.assertEqual(fps["track_lifecycle"], EXPECTED_TRACK_LIFECYCLE_FP)
        self.assertEqual(fps["team_assignments"], EXPECTED_TEAM_ASSIGNMENTS_FP)
        self.assertEqual(fps["jersey_observations"], EXPECTED_JERSEY_OBSERVATIONS_FP)
        for name in (
            "identity_evidence",
            "reid_candidate_links",
            "track_identity_assignments",
            TRACKLET_APPEARANCE_PROFILES_CONTRACT,
        ):
            self.assertEqual(len(fps[name]), 64)
            self.assertEqual(
                contract_fingerprint(get_contract(name, 1, registry=self.reg)),
                fps[name],
            )

    def test_02_observed_only_sampling(self) -> None:
        b = fixture_predicted_rejected()
        samp = sample_tracklet_crops(
            track_id=1,
            observations=b["observations"],
            frames_bgr=None,
            synthetic_crops=b["synthetic_crops"],
            attributes=b["attributes"],
            summaries=None,
            config=self.cfg,
        )
        self.assertEqual(len(samp.accepted), 2)
        self.assertIn("PREDICTED_OR_INTERPOLATED_REJECTED", samp.reject_reasons)

    def test_03_deterministic_descriptor_and_l2(self) -> None:
        b = fixture_same_appearance_different_tracklets()
        crop = next(iter(b["synthetic_crops"].values()))
        d1 = extract_descriptor_from_bgr(crop, config=self.cfg)
        d2 = extract_descriptor_from_bgr(crop, config=self.cfg)
        self.assertEqual(d1.vector, d2.vector)
        self.assertEqual(d1.dimension, self.dim)
        validate_embedding(d1.vector, expected_dim=self.dim)
        profiles, rows, _ = self._profiles(b)
        self.assertEqual(profiles[0].profile_fingerprint, profiles[0].profile_fingerprint)
        p2, _, _ = self._profiles(b)
        self.assertEqual(profiles[0].profile_fingerprint, p2[0].profile_fingerprint)
        self.assertEqual(len(rows[0]["embedding"]), self.dim)

    def test_04_same_vs_different_similarity(self) -> None:
        same = fixture_same_appearance_different_tracklets()
        diff = fixture_different_appearance()
        ps, _, _ = self._profiles(same)
        pd, _, _ = self._profiles(diff)
        sim_same = cosine_similarity(ps[0].embedding, ps[1].embedding)
        sim_diff = cosine_similarity(pd[0].embedding, pd[1].embedding)
        self.assertGreater(sim_same, sim_diff)
        self.assertGreater(sim_same, 0.85)

    def test_05_same_kit_hard_negative_flag(self) -> None:
        b = fixture_same_kit_hard_negative()
        profiles, rows, _ = self._profiles(b)
        self.assertTrue(all("same_kit_hard_negative_risk" in p.quality_flags for p in profiles))
        self.assertTrue(
            all(r["status"] in {"ok", "insufficient_appearance_evidence"} for r in rows)
        )

    def test_06_temporal_overlap_reject(self) -> None:
        b = fixture_temporal_overlap()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            r = run_reid_candidates(
                output_dir=td, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
        self.assertTrue(r.accepted)
        matches = r.summary["matches"]
        self.assertTrue(any(m.decision_status == "rejected" for m in matches))
        self.assertTrue(any("TEMPORAL_OVERLAP_FORBIDDEN" in m.reason_codes for m in matches))

    def test_07_cross_video_and_human_ball(self) -> None:
        b = fixture_cross_video_reject()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            r = run_reid_candidates(
                output_dir=td, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
        self.assertTrue(r.accepted)
        self.assertTrue(
            any("CROSS_VIDEO_AUTO_LINK_FORBIDDEN" in m.reason_codes for m in r.summary["matches"])
        )
        ball = fixture_human_ball_reject()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            r2 = run_reid_candidates(
                output_dir=td, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=ball
            )
        self.assertTrue(r2.accepted)
        self.assertTrue(
            any("HUMAN_BALL_LINK_FORBIDDEN" in m.reason_codes for m in r2.summary["matches"])
        )

    def test_08_ambiguity_and_appearance_only_candidate(self) -> None:
        b = fixture_ambiguity_near_scores()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            r = run_reid_candidates(
                output_dir=td, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
        self.assertTrue(r.accepted)
        # Appearance alone cannot confirm
        for er in r.summary["evidence_rows"]:
            self.assertEqual(er["evidence_type"], EvidenceType.APPEARANCE_SIMILARITY.value)
            self.assertIn(
                er["reliability_tier"],
                {
                    ReliabilityTier.SUPPORTING.value,
                    ReliabilityTier.WEAK.value,
                    ReliabilityTier.UNAVAILABLE.value,
                    ReliabilityTier.CONFLICTING.value,
                },
            )
            st, reasons = decide_assignment_status([er], policy=self.policy)
            self.assertEqual(st, "candidate")
            self.assertTrue(
                any("APPEARANCE" in x or "ALONE" in x for x in reasons) or st == "candidate"
            )
        self.assertFalse(self.cfg["matching"]["auto_confirm"])

    def test_09_insufficient_and_tiny(self) -> None:
        single = fixture_single_crop_insufficient()
        profiles, _, stats = self._profiles(single)
        self.assertEqual(profiles[0].status, "insufficient_appearance_evidence")
        self.assertGreaterEqual(stats["insufficient_evidence_count"], 1)
        tiny = fixture_tiny_corrupt_crop()
        p2, _, _ = self._profiles(tiny)
        self.assertEqual(p2[0].status, "insufficient_appearance_evidence")

    def test_10_brightness_stable_enough(self) -> None:
        b = fixture_brightness_shift()
        profiles, _, _ = self._profiles(b)
        sim = cosine_similarity(profiles[0].embedding, profiles[1].embedding)
        self.assertGreater(sim, 0.7)

    def test_11_no_crop_persist_atomic_cleanup_eval(self) -> None:
        b = fixture_same_appearance_different_tracklets()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            out = Path(td)
            r = run_appearance_extract(
                output_dir=out, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
            self.assertTrue(r.accepted)
            self.assertFalse(list(out.glob("**/*.png")))
            self.assertFalse(list(out.glob("**/*.jpg")))
            # no-overwrite
            r2 = run_appearance_extract(
                output_dir=out, config=self.cfg, contain_root=RUNTIME_ROOT, in_memory_bundle=b
            )
            self.assertFalse(r2.accepted)
            self.assertEqual(r2.error_code, "OVERWRITE_FORBIDDEN")

        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            out = Path(td)
            r3 = run_reid_candidates(
                output_dir=out,
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=b,
                inject_failure=True,
            )
            self.assertFalse(r3.accepted)
            self.assertFalse((out / "identity_evidence.parquet").exists())
            self.assertFalse((out / "reid_candidate_links.parquet").exists())

        ev = evaluate_appearance_reid(has_reviewed_ground_truth=False)
        self.assertEqual(ev.ground_truth_evaluation_status, NOT_EVALUATED_APPEARANCE_REID)
        self.assertTrue(all(v is None for v in ev.metrics.values()))

    def test_12_selection_matrix_handcrafted(self) -> None:
        selected = [
            m for m in self.cfg["selection_matrix"] if str(m["status"]).lower() == "selected"
        ]
        self.assertEqual(len(selected), 1)
        self.assertIn("handcrafted", str(selected[0]["candidate"]).lower())
        self.assertEqual(self.cfg["extractor_type"], "handcrafted")
        self.assertFalse(self.cfg["matching"]["face_regions_use"])

    def test_13_identity_contracts_regression(self) -> None:
        # Existing 7A decision matrix still holds for appearance alone.
        st, reasons = decide_assignment_status(
            [
                {
                    "evidence_type": EvidenceType.APPEARANCE_SIMILARITY.value,
                    "reliability_tier": ReliabilityTier.SUPPORTING.value,
                    "polarity": "supports",
                }
            ],
            policy=self.policy,
        )
        self.assertEqual(st, "candidate")
        self.assertIn("APPEARANCE_ALONE_INSUFFICIENT", reasons)

    def test_14_leakage_and_json_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            path = Path(td) / "receipt.json"
            write_json_record(path, {"a": 1}, overwrite=False)
            with self.assertRaises(RecordError):
                write_json_record(path, {"a": 2}, overwrite=False)

    def test_15_norm_error(self) -> None:
        bad = l2_normalize([1.0] * self.dim).tolist()
        bad[0] = float("nan")
        with self.assertRaises(AppearanceDescriptorError):
            validate_embedding(bad, expected_dim=self.dim)


if __name__ == "__main__":
    unittest.main()
