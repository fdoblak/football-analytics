"""Stage 7D jersey region + OCR baseline tests (synthetic)."""

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
from football_analytics.identity.contracts import (
    EXPECTED_JERSEY_OBSERVATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    EXPECTED_TEAM_ASSIGNMENTS_FP,
    assert_frozen_upstream_fingerprints,
    assert_identity_contracts_registered,
    identity_schema_fingerprints,
)
from football_analytics.identity.jersey_consensus import (
    JerseyObservationVote,
    consensus_for_track,
)
from football_analytics.identity.jersey_ocr import (
    clear_digit_template_cache,
    recognize_jersey_number,
)
from football_analytics.identity.jersey_ocr_config import (
    default_jersey_ocr_config_path,
    jersey_ocr_config_fingerprint,
    load_jersey_ocr_config,
)
from football_analytics.identity.jersey_ocr_evaluation import (
    NOT_EVALUATED_JERSEY_OCR,
    evaluate_jersey_ocr,
)
from football_analytics.identity.jersey_ocr_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    fixture_ball_excluded,
    fixture_conflicting_track,
    fixture_leading_zero,
    fixture_leakage_probe,
    fixture_no_number_front,
    fixture_predicted_rejected,
    fixture_referee_excluded,
    fixture_single_digit,
    fixture_single_weak_observation,
    fixture_sponsor_logo_negative,
    fixture_team_jersey_conflict,
    fixture_two_digit,
    fixture_unknown_role,
    render_digit_crop,
)
from football_analytics.identity.jersey_ocr_service import run_jersey_observe
from football_analytics.identity.jersey_region import propose_torso_regions
from football_analytics.identity.policy import decide_assignment_status, load_identity_policy

ROOT = default_project_root()


class JerseyOcrBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clear_digit_template_cache()
        cls.cfg = load_jersey_ocr_config(default_jersey_ocr_config_path())
        cls.cfg_fp = jersey_ocr_config_fingerprint(cls.cfg)
        cls.policy = load_identity_policy()
        cls.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        assert_runtime_root()

    def test_01_fingerprint_regression(self) -> None:
        assert_identity_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_frozen_upstream_fingerprints(registry=self.reg)
        fps = identity_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["jersey_observations"], EXPECTED_JERSEY_OBSERVATIONS_FP)
        self.assertEqual(fps["team_assignments"], EXPECTED_TEAM_ASSIGNMENTS_FP)
        self.assertEqual(
            contract_fingerprint(get_contract("jersey_observations", 1, registry=self.reg)),
            EXPECTED_JERSEY_OBSERVATIONS_FP,
        )

    def test_02_selection_matrix_opencv_selected(self) -> None:
        selected = [
            m for m in self.cfg["selection_matrix"] if str(m.get("status")).lower() == "selected"
        ]
        self.assertEqual(len(selected), 1)
        self.assertIn("opencv", str(selected[0]["candidate"]).lower())
        future = [
            m
            for m in self.cfg["selection_matrix"]
            if "sn-jersey" in str(m.get("candidate")).lower()
        ]
        self.assertTrue(future)
        self.assertEqual(str(future[0]["status"]).lower(), "future")

    def test_03_observed_only_eligibility(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "pred",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_predicted_rejected(),
            )
        self.assertTrue(r.accepted, r.error_code)
        for row in r.summary["observation_rows"]:
            self.assertIn("not_eligible", row["quality_flags"])
            self.assertIsNone(row["raw_text"])

    def test_04_region_containment(self) -> None:
        b = fixture_two_digit()
        s = b["samples"][0]
        bbox = s["bbox"]
        regs = propose_torso_regions(s["frame_image"], bbox, config=self.cfg)
        self.assertTrue(regs)
        bx0, by0, bx1, by1 = bbox
        for c in regs:
            self.assertGreaterEqual(c.x0, int(bx0))
            self.assertGreaterEqual(c.y0, int(by0))
            self.assertLessEqual(c.x1, int(bx1) + 1)
            self.assertLessEqual(c.y1, int(by1) + 1)

    def test_05_one_two_digit_and_leading_zero(self) -> None:
        for text, fx in (
            ("7", fixture_single_digit),
            ("10", fixture_two_digit),
            ("07", fixture_leading_zero),
        ):
            with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
                r = run_jersey_observe(
                    output_dir=Path(td) / text,
                    config=self.cfg,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=fx(),
                )
            self.assertTrue(r.accepted, r.error_code)
            observed = [row for row in r.summary["observation_rows"] if row.get("raw_text")]
            self.assertTrue(observed, msg=text)
            self.assertEqual(observed[0]["raw_text"], text)
            if text == "07":
                self.assertIsNone(observed[0]["normalized_number"])
                self.assertEqual(observed[0]["digit_count"], 2)
                self.assertIn("leading_zero", observed[0]["quality_flags"])
            elif text == "7":
                self.assertEqual(observed[0]["normalized_number"], 7)
                self.assertEqual(observed[0]["digit_count"], 1)
            else:
                self.assertEqual(observed[0]["normalized_number"], 10)
                self.assertEqual(observed[0]["digit_count"], 2)
            self.assertIsNone(observed[0]["confidence"])

    def test_06_negative_false_number_controls(self) -> None:
        for fx in (fixture_no_number_front, fixture_sponsor_logo_negative):
            with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
                r = run_jersey_observe(
                    output_dir=Path(td) / "neg",
                    config=self.cfg,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=fx(),
                )
            self.assertTrue(r.accepted, r.error_code)
            for row in r.summary["observation_rows"]:
                self.assertIsNone(row["raw_text"])
                self.assertIsNone(row["normalized_number"])
                flags = row["quality_flags"]
                self.assertTrue(
                    any(f in flags for f in ("no_digits", "no_region", "not_eligible", "ambiguous"))
                )
            self.assertEqual(r.summary.get("false_number_emission_rate"), 0.0)

    def test_07_digit_ordering_direct(self) -> None:
        clear_digit_template_cache()
        r = recognize_jersey_number(render_digit_crop("12"), config=self.cfg)
        self.assertEqual(r.status, "observed")
        self.assertEqual(r.raw_text, "12")

    def test_08_track_consensus_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "conf",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_conflicting_track(),
            )
        self.assertTrue(r.accepted, r.error_code)
        cons = r.summary["consensus"]
        self.assertTrue(cons)
        self.assertEqual(cons[0].status, "ambiguous")
        self.assertTrue(cons[0].review_required)

        weak = [
            JerseyObservationVote(
                track_id=1,
                frame_index=0,
                observation_id=0,
                raw_text="5",
                normalized_number=5,
                quality=0.5,
                score=0.6,
                status="observed",
            )
        ]
        c = consensus_for_track(weak, config=self.cfg)
        self.assertEqual(c.status, "ambiguous")
        self.assertIn("INSUFFICIENT_OBSERVATIONS", c.reason_codes)

    def test_09_jersey_alone_no_auto_confirm(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "alone",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
        self.assertTrue(r.accepted, r.error_code)
        supporting = [
            e
            for e in r.summary["evidence_rows"]
            if e["polarity"] == "supports" and e["evidence_type"] == "jersey_number"
        ]
        self.assertTrue(supporting)
        for er in supporting:
            st, reasons = decide_assignment_status([er], policy=self.policy)
            self.assertEqual(st, "candidate")
            self.assertIn("JERSEY_ALONE_INSUFFICIENT", reasons)
            self.assertNotIn(er["reliability_tier"], {"strong", "manual_verified"})

    def test_10_team_jersey_conflict_review(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "tj",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_team_jersey_conflict(),
            )
        self.assertTrue(r.accepted, r.error_code)
        conflict = [
            e for e in r.summary["evidence_rows"] if "TEAM_JERSEY_CONFLICT" in e["reason_codes"]
        ]
        self.assertTrue(conflict)
        self.assertEqual(conflict[0]["polarity"], "conflicts")
        self.assertEqual(conflict[0]["review_status"], "needs_review")

    def test_11_role_and_entity_exclusions(self) -> None:
        for fx in (fixture_referee_excluded, fixture_ball_excluded):
            with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
                r = run_jersey_observe(
                    output_dir=Path(td) / "ex",
                    config=self.cfg,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=fx(),
                )
            self.assertTrue(r.accepted, r.error_code)
            for row in r.summary["observation_rows"]:
                self.assertIn("not_eligible", row["quality_flags"])
                self.assertIsNone(row["raw_text"])

    def test_12_unknown_role_conservative(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "unk",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_unknown_role(),
            )
        self.assertTrue(r.accepted, r.error_code)
        # May observe or abstain; must never auto-confirm and may flag conservative.
        for er in r.summary["evidence_rows"]:
            if er["polarity"] == "supports":
                st, _ = decide_assignment_status([er], policy=self.policy)
                self.assertEqual(st, "candidate")

    def test_13_leakage_rejection(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "leak",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_leakage_probe(),
            )
        self.assertFalse(r.accepted)
        self.assertEqual(r.error_code, "LEAKAGE_SEPARATION_VIOLATION")

    def test_14_no_overwrite_and_failure_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            out = Path(td) / "ow"
            r1 = run_jersey_observe(
                output_dir=out,
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
            self.assertTrue(r1.accepted)
            r2 = run_jersey_observe(
                output_dir=out,
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
            self.assertEqual(r2.error_code, "OVERWRITE_FORBIDDEN")
            r3 = run_jersey_observe(
                output_dir=Path(td) / "fail",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
                inject_failure=True,
            )
            self.assertFalse(r3.accepted)
            self.assertFalse((Path(td) / "fail" / "jersey_observations.parquet").exists())

    def test_15_no_crop_persistence(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            out = Path(td) / "nocrop"
            r = run_jersey_observe(
                output_dir=out,
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
        self.assertTrue(r.accepted)
        crop_files = list(out.rglob("*.png")) + list(out.rglob("*.jpg"))
        self.assertEqual(crop_files, [])
        for row in r.summary["observation_rows"]:
            self.assertIsNone(row["crop_artifact_id"])

    def test_16_receipt_recount_and_eval_code(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "rcpt",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
        self.assertTrue(r.accepted)
        receipt = r.summary["receipt"]
        self.assertEqual(receipt["counts"]["observed"], r.summary["quality"]["counts"]["observed"])
        self.assertFalse(receipt["auto_confirm"])
        self.assertFalse(receipt["persist_crops"])
        self.assertFalse(receipt["face_recognition_used"])
        self.assertEqual(receipt["evaluation_status"], NOT_EVALUATED_JERSEY_OCR)
        ev = evaluate_jersey_ocr(has_reviewed_ground_truth=False)
        self.assertEqual(ev.ground_truth_evaluation_status, NOT_EVALUATED_JERSEY_OCR)

    def test_17_determinism(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            a = run_jersey_observe(
                output_dir=Path(td) / "a",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
            b = run_jersey_observe(
                output_dir=Path(td) / "b",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_two_digit(),
            )
        self.assertTrue(a.accepted and b.accepted)
        self.assertEqual(
            [row["raw_text"] for row in a.summary["observation_rows"]],
            [row["raw_text"] for row in b.summary["observation_rows"]],
        )

    def test_18_weak_single_no_hard_consensus(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT)) as td:
            r = run_jersey_observe(
                output_dir=Path(td) / "weak",
                config=self.cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fixture_single_weak_observation(),
            )
        self.assertTrue(r.accepted)
        # Frame may observe, but track consensus must not hard-confirm from one vote.
        if r.summary["consensus"]:
            self.assertNotEqual(r.summary["consensus"][0].status, "observed")


if __name__ == "__main__":
    unittest.main()
