"""Stage 7E target identity fusion + manual approval tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.identity.contracts import (
    EXPECTED_JERSEY_OBSERVATIONS_FP,
    EXPECTED_TEAM_ASSIGNMENTS_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    assert_frozen_upstream_fingerprints,
    assert_identity_contracts_registered,
    load_identity_contract,
)
from football_analytics.identity.metric_eligibility import resolve_metric_eligibility
from football_analytics.identity.policy import decide_assignment_status, load_identity_policy
from football_analytics.identity.target_decisions import (
    TargetDecisionError,
    assert_manifest_cas,
    build_target_decision,
)
from football_analytics.identity.target_eligibility_timeline import build_eligibility_timeline
from football_analytics.identity.target_fusion import (
    detect_confirmed_overlaps,
    fuse_track_evidence,
)
from football_analytics.identity.target_fusion_config import (
    load_target_fusion_config,
    target_fusion_config_fingerprint,
)
from football_analytics.identity.target_fusion_evaluation import (
    NOT_EVALUATED_TARGET_IDENTITY,
    evaluate_target_fusion,
)
from football_analytics.identity.target_fusion_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    get_fixture,
)
from football_analytics.identity.target_fusion_service import (
    apply_decision,
    prepare_review,
    resolve_fusion,
    run_fixture_decision,
    validate_fusion_outputs,
)
from football_analytics.identity.target_ranking import rank_candidates
from football_analytics.identity.target_review import validate_review_manifest


class TestTargetIdentityPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert_runtime_root()
        cls.config = load_target_fusion_config()
        cls.policy = load_identity_policy()
        cls.cfg_fp = target_fusion_config_fingerprint(cls.config)

    def _tmpdir(self):
        return tempfile.TemporaryDirectory(dir=str(RUNTIME_ROOT))

    def test_01_fingerprint_pins(self) -> None:
        assert_identity_contracts_registered()
        assert_frozen_upstream_fingerprints()
        team = load_identity_contract("team_assignments", 1)
        jer = load_identity_contract("jersey_observations", 1)
        trk = load_identity_contract("track_observations", 1)
        self.assertEqual(contract_fingerprint(team), EXPECTED_TEAM_ASSIGNMENTS_FP)
        self.assertEqual(contract_fingerprint(jer), EXPECTED_JERSEY_OBSERVATIONS_FP)
        self.assertEqual(contract_fingerprint(trk), EXPECTED_TRACK_OBSERVATIONS_FP)
        self.assertFalse(self.config["safety"]["auto_confirm"])
        self.assertFalse(self.config["safety"]["face_recognition"])
        self.assertFalse(self.config["safety"]["cross_video_auto_link"])

    def test_02_appearance_only_no_confirm(self) -> None:
        fx = get_fixture("appearance_only")
        status, reasons, _, _ = fuse_track_evidence(
            fx["evidence"], policy=self.policy, within_manual_anchor_scope=False
        )
        self.assertEqual(status, "candidate")
        self.assertIn("APPEARANCE_ALONE_INSUFFICIENT", reasons)
        self.assertNotEqual(status, "confirmed")

    def test_03_jersey_only_no_confirm(self) -> None:
        fx = get_fixture("jersey_only")
        status, reasons, _, _ = fuse_track_evidence(fx["evidence"], policy=self.policy)
        self.assertEqual(status, "candidate")
        self.assertIn("JERSEY_ALONE_INSUFFICIENT", reasons)

    def test_04_team_only_no_confirm(self) -> None:
        fx = get_fixture("team_only")
        status, reasons, _, _ = fuse_track_evidence(fx["evidence"], policy=self.policy)
        self.assertEqual(status, "candidate")
        self.assertIn("TEAM_ALONE_INSUFFICIENT", reasons)

    def test_05_two_auto_max_provisional(self) -> None:
        fx = get_fixture("two_auto_provisional")
        status, reasons, _, _ = fuse_track_evidence(fx["evidence"], policy=self.policy)
        self.assertEqual(status, "provisional")
        self.assertIn("MULTI_SUPPORTING_PROVISIONAL", reasons)
        self.assertNotEqual(status, "confirmed")

    def test_06_scoped_manual_confirm_and_no_propagation(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_manual"
            prep = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="scoped_manual_confirm",
            )
            self.assertTrue(prep.accepted)
            manifest = validate_review_manifest(
                json.loads(Path(prep.manifest_json).read_text(encoding="utf-8"))
            )
            by_track = {int(c["track_id"]): c for c in manifest["candidates"]}
            self.assertNotEqual(by_track[4]["proposed_status"], "confirmed")
            self.assertIn(by_track[5]["proposed_status"], {"candidate", "provisional"})
            dec = run_fixture_decision(
                output_dir=out,
                config=self.config,
                decision="confirm",
                track_id=4,
                start=0,
                end=40,
                contain_root=RUNTIME_ROOT,
            )
            self.assertTrue(dec.accepted, dec.error_code)
            res = resolve_fusion(output_dir=out, config=self.config, contain_root=RUNTIME_ROOT)
            self.assertTrue(res.accepted, res.error_code)
            import pyarrow.parquet as pq

            rows = pq.read_table(res.assignments_parquet).to_pylist()
            confirmed = [r for r in rows if r["assignment_status"] == "confirmed"]
            self.assertEqual(len(confirmed), 1)
            self.assertEqual(confirmed[0]["track_id"], 4)
            # Linked tracklet not auto-confirmed.
            track5 = [r for r in rows if r["track_id"] == 5]
            self.assertTrue(track5)
            self.assertTrue(all(r["assignment_status"] != "confirmed" for r in track5))

    def test_07_conflict_jersey_team(self) -> None:
        fx = get_fixture("conflict_jersey_team")
        status, reasons, _, _ = fuse_track_evidence(fx["evidence"], policy=self.policy)
        self.assertEqual(status, "rejected")
        self.assertIn("HARD_EVIDENCE_CONFLICT", reasons)

    def test_08_two_simultaneous_confirmed_rejected(self) -> None:
        rows = [
            {
                "assignment_id": "asn_a",
                "track_id": 1,
                "target_player_id": "target_player_01",
                "assignment_status": "confirmed",
                "start_frame_index": 0,
                "end_frame_index": 50,
            },
            {
                "assignment_id": "asn_b",
                "track_id": 2,
                "target_player_id": "target_player_01",
                "assignment_status": "confirmed",
                "start_frame_index": 25,
                "end_frame_index": 75,
            },
        ]
        findings = detect_confirmed_overlaps(rows)
        self.assertTrue(findings)
        self.assertEqual(findings[0]["code"], "DUPLICATE_CONFIRMED_IDENTITY")

    def test_09_cross_video_auto_link_rejected(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_xvid"
            prep = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="cross_video_forbidden",
            )
            self.assertFalse(prep.accepted)
            self.assertEqual(prep.error_code, "CROSS_VIDEO_AUTO_LINK_FORBIDDEN")

    def test_10_long_gap_new_track_candidate(self) -> None:
        fx = get_fixture("long_gap_candidate")
        status, reasons, _, _ = fuse_track_evidence(
            fx["evidence"], policy=self.policy, long_gap_after_cut=True
        )
        self.assertEqual(status, "candidate")
        self.assertIn("LONG_GAP_NEW_TRACK_CANDIDATE", reasons)

    def test_11_revocation_append_only(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_revoke"
            prep = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="scoped_manual_confirm",
            )
            self.assertTrue(prep.accepted)
            self.assertTrue(
                run_fixture_decision(
                    output_dir=out,
                    config=self.config,
                    decision="confirm",
                    track_id=4,
                    start=0,
                    end=40,
                    decision_id="dec_confirm_track4",
                    contain_root=RUNTIME_ROOT,
                ).accepted
            )
            self.assertTrue(
                run_fixture_decision(
                    output_dir=out,
                    config=self.config,
                    decision="revoke",
                    track_id=4,
                    start=0,
                    end=40,
                    decision_id="dec_revoke_track4",
                    contain_root=RUNTIME_ROOT,
                ).accepted
            )
            res = resolve_fusion(output_dir=out, config=self.config, contain_root=RUNTIME_ROOT)
            self.assertTrue(res.accepted, res.error_code)
            import pyarrow.parquet as pq

            rows = pq.read_table(res.assignments_parquet).to_pylist()
            self.assertTrue(any(r["assignment_status"] == "revoked" for r in rows))
            audit = Path(out) / "identity_manual_audit.jsonl"
            lines = [ln for ln in audit.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertGreaterEqual(len(lines), 2)

    def test_12_append_only_hash_chain_audit(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_audit"
            prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="two_auto_provisional",
            )
            d1 = run_fixture_decision(
                output_dir=out,
                config=self.config,
                decision="keep_provisional",
                track_id=3,
                start=0,
                end=30,
                decision_id="dec_keep_prov",
                contain_root=RUNTIME_ROOT,
            )
            self.assertTrue(d1.accepted)
            d2 = run_fixture_decision(
                output_dir=out,
                config=self.config,
                decision="reject",
                track_id=3,
                start=0,
                end=30,
                decision_id="dec_reject_later",
                contain_root=RUNTIME_ROOT,
            )
            self.assertTrue(d2.accepted)
            audit = Path(out) / "identity_manual_audit.jsonl"
            entries = [
                json.loads(ln) for ln in audit.read_text(encoding="utf-8").splitlines() if ln
            ]
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(e["provenance"]["append_only"] is True for e in entries))

    def test_13_stale_decision_rejection(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_stale"
            prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="two_auto_provisional",
            )
            manifest = validate_review_manifest(
                json.loads((out / "target_review_manifest.json").read_text(encoding="utf-8"))
            )
            payload = build_target_decision(
                decision_id="dec_stale_one",
                manifest=manifest,
                track_id=3,
                start_frame_index=0,
                end_frame_index=30,
                decision="reject",
                reviewer_id="synth_reviewer",
                reason="stale_test",
                expected_assignment_version=99,
                expected_previous_status="provisional",
                evidence_fingerprints=[],
                previous_audit_hash=None,
                synthetic_fixture=True,
            )
            with self.assertRaises(TargetDecisionError):
                assert_manifest_cas(payload, manifest=manifest, current_assignment_version=1)
            bad = apply_decision(
                output_dir=out,
                decision_payload=payload,
                config=self.config,
                contain_root=RUNTIME_ROOT,
            )
            self.assertFalse(bad.accepted)
            self.assertEqual(bad.error_code, "STALE_DECISION")

    def test_14_duplicate_decision_rejection(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_dup"
            prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="two_auto_provisional",
            )
            first = run_fixture_decision(
                output_dir=out,
                config=self.config,
                decision="reject",
                track_id=3,
                start=0,
                end=30,
                decision_id="dec_dup_same",
                contain_root=RUNTIME_ROOT,
            )
            self.assertTrue(first.accepted)
            second = run_fixture_decision(
                output_dir=out,
                config=self.config,
                decision="reject",
                track_id=3,
                start=0,
                end=30,
                decision_id="dec_dup_same",
                contain_root=RUNTIME_ROOT,
            )
            self.assertFalse(second.accepted)
            self.assertEqual(second.error_code, "DUPLICATE_DECISION")

    def test_15_predicted_not_eligible(self) -> None:
        elig = resolve_metric_eligibility(
            assignment_status="confirmed",
            target_scope="target",
            has_observed_tracking=True,
            sufficient_coverage=True,
            unresolved_hard_conflict=False,
            observation_state="predicted",
        )
        self.assertEqual(elig, "not_eligible")

    def test_16_confirmed_observed_eligible(self) -> None:
        elig = resolve_metric_eligibility(
            assignment_status="confirmed",
            target_scope="target",
            has_observed_tracking=True,
            sufficient_coverage=True,
            unresolved_hard_conflict=False,
            observation_state="observed",
        )
        self.assertEqual(elig, "eligible")

    def test_17_insufficient_coverage_not_evaluable(self) -> None:
        elig = resolve_metric_eligibility(
            assignment_status="confirmed",
            target_scope="target",
            has_observed_tracking=True,
            sufficient_coverage=False,
            unresolved_hard_conflict=False,
            observation_state="observed",
        )
        self.assertEqual(elig, "not_evaluable")
        timeline = build_eligibility_timeline(
            [
                {
                    "assignment_id": "asn_cov",
                    "track_id": 1,
                    "assignment_status": "confirmed",
                    "target_scope": "target",
                    "start_frame_index": 0,
                    "end_frame_index": 9,
                }
            ],
            timeline_id="telcoverage01",
            run_id="runid_coverage_test_01",
            video_id="video_cov",
            target_player_id="target_player_01",
            sufficient_coverage_by_assignment={"asn_cov": False},
        )
        self.assertGreater(timeline["summary"]["not_evaluable_frame_count"], 0)

    def test_18_false_target_attribution_synthetic_zero(self) -> None:
        report = evaluate_target_fusion(
            assignments=[
                {
                    "assignment_id": "asn_ok",
                    "track_id": 4,
                    "assignment_status": "confirmed",
                }
            ],
            has_reviewed_ground_truth=False,
            synthetic_expected_track_ids=[4],
        )
        self.assertEqual(report.metrics["false_target_attribution"], 0.0)

    def test_19_evaluation_leakage_hard_fail(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_leak"
            prep = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="evaluation_leakage",
            )
            self.assertFalse(prep.accepted)
            self.assertEqual(prep.error_code, "LEAKAGE_SEPARATION_VIOLATION")

    def test_20_deterministic_ranking(self) -> None:
        cands = [
            {
                "candidate_id": "cand_b",
                "track_id": 2,
                "rank_score": 1.0,
                "reason_codes": [],
                "ambiguous": False,
            },
            {
                "candidate_id": "cand_a",
                "track_id": 1,
                "rank_score": 1.0,
                "reason_codes": [],
                "ambiguous": False,
            },
        ]
        r1 = rank_candidates(cands, ambiguity_margin=0.05)
        r2 = rank_candidates(cands, ambiguity_margin=0.05)
        self.assertEqual([c["candidate_id"] for c in r1], [c["candidate_id"] for c in r2])
        self.assertTrue(r1[0]["ambiguous"])

    def test_21_receipt_recount_and_eval_code(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_receipt"
            prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="two_auto_provisional",
            )
            res = resolve_fusion(output_dir=out, config=self.config, contain_root=RUNTIME_ROOT)
            self.assertTrue(res.accepted, res.error_code)
            val = validate_fusion_outputs(
                output_dir=out, config=self.config, contain_root=RUNTIME_ROOT
            )
            self.assertTrue(val.accepted, val.error_code)
            receipt = json.loads(Path(res.receipt_json).read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["ground_truth_evaluation_status"], NOT_EVALUATED_TARGET_IDENTITY
            )
            eval_payload = json.loads(Path(res.evaluation_json).read_text(encoding="utf-8"))
            self.assertEqual(
                eval_payload["ground_truth_evaluation_status"],
                NOT_EVALUATED_TARGET_IDENTITY,
            )

    def test_22_atomic_no_overwrite(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_ow"
            first = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="appearance_only",
            )
            self.assertTrue(first.accepted)
            second = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="appearance_only",
            )
            self.assertFalse(second.accepted)
            self.assertEqual(second.error_code, "OVERWRITE_FORBIDDEN")

    def test_23_failure_cleanup(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_fail"
            result = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="appearance_only",
                inject_failure=True,
            )
            self.assertFalse(result.accepted)
            self.assertEqual(result.error_code, "INJECTED_FAILURE")
            self.assertFalse((out / "target_review_manifest.json").exists())

    def test_24_stage7_policy_regression(self) -> None:
        status, reasons = decide_assignment_status(
            get_fixture("jersey_only")["evidence"], policy=self.policy
        )
        self.assertEqual(status, "candidate")
        self.assertIn("JERSEY_ALONE_INSUFFICIENT", reasons)

    def test_25_no_reviewed_gt_code(self) -> None:
        self.assertEqual(
            NOT_EVALUATED_TARGET_IDENTITY,
            "NOT_EVALUATED_NO_REVIEWED_TARGET_IDENTITY_GROUND_TRUTH",
        )

    def test_26_e2e_confirm_reject_revoke_cleanup(self) -> None:
        with self._tmpdir() as td:
            out = Path(td) / "run_e2e"
            prep = prepare_review(
                output_dir=out,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                fixture_name="e2e_bundle",
            )
            self.assertTrue(prep.accepted)
            self.assertTrue(
                run_fixture_decision(
                    output_dir=out,
                    config=self.config,
                    decision="confirm",
                    track_id=4,
                    start=0,
                    end=40,
                    decision_id="dec_e2e_confirm",
                    contain_root=RUNTIME_ROOT,
                ).accepted
            )
            self.assertTrue(
                run_fixture_decision(
                    output_dir=out,
                    config=self.config,
                    decision="reject",
                    track_id=5,
                    start=50,
                    end=80,
                    decision_id="dec_e2e_reject",
                    contain_root=RUNTIME_ROOT,
                ).accepted
            )
            self.assertTrue(
                run_fixture_decision(
                    output_dir=out,
                    config=self.config,
                    decision="revoke",
                    track_id=4,
                    start=0,
                    end=40,
                    decision_id="dec_e2e_revoke",
                    contain_root=RUNTIME_ROOT,
                ).accepted
            )
            res = resolve_fusion(output_dir=out, config=self.config, contain_root=RUNTIME_ROOT)
            self.assertTrue(res.accepted, res.error_code)
            val = validate_fusion_outputs(
                output_dir=out, config=self.config, contain_root=RUNTIME_ROOT
            )
            self.assertTrue(val.accepted)
            # Cleanup runtime artifacts after successful e2e (test-owned).
            for p in out.rglob("*"):
                if p.is_file():
                    p.unlink()


if __name__ == "__main__":
    unittest.main()
