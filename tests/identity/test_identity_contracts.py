"""Stage 7A identity / ReID / target-player contract tests (synthetic only)."""

from __future__ import annotations

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
from football_analytics.identity import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_JERSEY_OBSERVATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    EXPECTED_TEAM_ASSIGNMENTS_FP,
    EXPECTED_TRACK_LIFECYCLE_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    NOT_EVALUATED_IDENTITY,
    assert_frozen_upstream_fingerprints,
    assert_identity_contracts_registered,
    build_revocation,
    decide_assignment_status,
    evaluate_identity,
    identity_schema_fingerprints,
    load_identity_json_schema,
    load_identity_policy,
    policy_fingerprint,
    resolve_metric_eligibility,
    validate_against_json_schema,
    validate_identity_bundle,
    validate_receipt_payload,
    validate_reid_link,
    validate_request_payload,
)
from football_analytics.identity.fixtures import (
    _cast,
    alone_insufficient_bundle,
    audit_entry,
    conflicting_jersey_team_bundle,
    cross_shot_link_bundle,
    cross_video_auto_link_rows,
    leakage_negative_bundle,
    manual_anchor_bundle,
    two_supporting_bundle,
    two_target_candidates_bundle,
    unknown_low_coverage_bundle,
)
from football_analytics.identity.receipt import (
    build_synthetic_receipt,
    build_synthetic_target_request,
)
from football_analytics.identity.review_audit import append_audit_log, read_audit_log
from football_analytics.identity.types import IdentityContractError

ROOT = default_project_root()


class IdentityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        self.policy = load_identity_policy()
        self.pol_fp = policy_fingerprint(self.policy)

    def test_01_registry_and_frozen_fingerprints(self) -> None:
        assert_identity_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_frozen_upstream_fingerprints(registry=self.reg)
        fps = identity_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["team_assignments"], EXPECTED_TEAM_ASSIGNMENTS_FP)
        self.assertEqual(fps["jersey_observations"], EXPECTED_JERSEY_OBSERVATIONS_FP)
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(fps["track_observations"], EXPECTED_TRACK_OBSERVATIONS_FP)
        self.assertEqual(fps["track_summaries"], EXPECTED_TRACK_SUMMARIES_FP)
        self.assertEqual(fps["track_lifecycle"], EXPECTED_TRACK_LIFECYCLE_FP)
        for name in (
            "identity_evidence",
            "reid_candidate_links",
            "track_identity_assignments",
        ):
            self.assertEqual(
                contract_fingerprint(get_contract(name, 1, registry=self.reg)),
                fps[name],
            )
            self.assertEqual(len(fps[name]), 64)

    def test_02_manual_anchor_confirmed(self) -> None:
        b = manual_anchor_bundle(self.pol_fp)
        vr = validate_identity_bundle(
            identity_evidence=b["identity_evidence"],
            track_identity_assignments=b["track_identity_assignments"],
            policy=self.policy,
        )
        self.assertNotEqual(vr.status, "FAIL", msg=vr.errors)
        st, reasons = decide_assignment_status(
            b["evidence_rows"], policy=self.policy, within_manual_anchor_scope=True
        )
        self.assertEqual(st, "confirmed")
        self.assertIn("MANUAL_VERIFIED_IN_SCOPE", reasons)

    def test_03_alone_insufficient(self) -> None:
        for et, code in (
            ("jersey_number", "JERSEY_ALONE_INSUFFICIENT"),
            ("team_assignment", "TEAM_ALONE_INSUFFICIENT"),
            ("appearance_similarity", "APPEARANCE_ALONE_INSUFFICIENT"),
        ):
            b = alone_insufficient_bundle(self.pol_fp, evidence_type=et)
            st, reasons = decide_assignment_status(b["evidence_rows"], policy=self.policy)
            self.assertEqual(st, "candidate")
            self.assertIn(code, reasons)

    def test_04_two_supporting_and_conflict(self) -> None:
        b = two_supporting_bundle(self.pol_fp)
        st, _ = decide_assignment_status(b["evidence_rows"], policy=self.policy)
        self.assertEqual(st, "provisional")
        c = conflicting_jersey_team_bundle(self.pol_fp)
        stc, reasons = decide_assignment_status(c["evidence_rows"], policy=self.policy)
        self.assertEqual(stc, "rejected")
        self.assertIn("HARD_EVIDENCE_CONFLICT", reasons)

    def test_05_cross_video_and_long_gap(self) -> None:
        with self.assertRaises(IdentityContractError):
            validate_reid_link(cross_video_auto_link_rows(self.pol_fp)[0])
        b = cross_shot_link_bundle(self.pol_fp, long_gap=True)
        self.assertEqual(b["link_rows"][0]["decision_status"], "review_required")

    def test_06_revoke_and_metric_eligibility(self) -> None:
        b = manual_anchor_bundle(self.pol_fp)
        revoked = build_revocation(
            b["assignment_rows"][0], new_assignment_id="asn_rev", reason="TEST"
        )
        self.assertEqual(revoked["assignment_status"], "revoked")
        self.assertEqual(revoked["metric_eligibility"], "not_eligible")
        self.assertEqual(
            resolve_metric_eligibility(
                assignment_status="confirmed",
                target_scope="target",
                has_observed_tracking=True,
                sufficient_coverage=True,
                unresolved_hard_conflict=False,
                observation_state="predicted",
            ),
            "not_eligible",
        )

    def test_07_duplicate_confirmed_and_leakage(self) -> None:
        b = manual_anchor_bundle(self.pol_fp)
        rows = b["assignment_rows"] + [
            {**b["assignment_rows"][0], "assignment_id": "asn_dup", "track_id": 2}
        ]
        vr = validate_identity_bundle(
            identity_evidence=b["identity_evidence"],
            track_identity_assignments=_cast("track_identity_assignments", rows),
            policy=self.policy,
        )
        self.assertEqual(vr.status, "FAIL")
        leak = leakage_negative_bundle(self.pol_fp)
        vr2 = validate_identity_bundle(
            identity_evidence=leak["identity_evidence"],
            track_identity_assignments=leak["track_identity_assignments"],
            policy=self.policy,
        )
        self.assertEqual(vr2.status, "FAIL")
        self.assertTrue(any("LEAKAGE" in e for e in vr2.errors))

    def test_08_request_receipt_audit_eval(self) -> None:
        b = manual_anchor_bundle(self.pol_fp)
        req = build_synthetic_target_request(
            run_id=b["run_id"],
            video_id=b["video_id"],
            policy_fingerprint=self.pol_fp,
            manual_anchors=[
                {
                    "anchor_id": "anchor_01",
                    "track_id": 0,
                    "frame_index": 0,
                    "start_frame_index": 0,
                    "end_frame_index": 20,
                    "reviewer_id": "reviewer_01",
                    "reviewed_at_utc": "2026-07-23T12:00:00.000000Z",
                    "notes": None,
                }
            ],
        )
        validate_request_payload(req)
        receipt = build_synthetic_receipt(
            run_id=b["run_id"],
            video_id=b["video_id"],
            policy_fingerprint=self.pol_fp,
            assignments=b["assignment_rows"],
            evidence=b["evidence_rows"],
        )
        validate_receipt_payload(receipt)
        ev = evaluate_identity()
        self.assertEqual(ev.ground_truth_evaluation_status, NOT_EVALUATED_IDENTITY)
        validate_against_json_schema(
            ev.to_dict(run_id=b["run_id"], video_id=b["video_id"]),
            load_identity_json_schema("identity_evaluation"),
        )
        # face forbidden in policy/docs path
        self.assertTrue(self.policy["safety"]["face_recognition_forbidden"])

    def test_09_append_only_and_no_overwrite(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.jsonl"
            b = two_target_candidates_bundle(self.pol_fp)
            e1 = audit_entry(
                run_id=b["run_id"],
                video_id=b["video_id"],
                target_player_id=b["target_player_id"],
            )
            append_audit_log(audit, e1, contain_root=root)
            e2 = audit_entry(
                run_id=b["run_id"],
                video_id=b["video_id"],
                target_player_id=b["target_player_id"],
                audit_id="audit_02",
                action="revoke",
                previous_decision="confirmed",
                new_decision="revoked",
            )
            append_audit_log(audit, e2, contain_root=root)
            self.assertEqual(len(read_audit_log(audit)), 2)
            out = root / "receipt.json"
            receipt = build_synthetic_receipt(
                run_id=b["run_id"],
                video_id=b["video_id"],
                policy_fingerprint=self.pol_fp,
                assignments=b["assignment_rows"],
                evidence=b["evidence_rows"],
            )
            write_json_record(out, receipt, overwrite=False)
            with self.assertRaises(RecordError):
                write_json_record(out, receipt, overwrite=False)

    def test_10_unknown_coverage_and_json_schemas(self) -> None:
        b = unknown_low_coverage_bundle(self.pol_fp)
        self.assertEqual(b["assignment_rows"][0]["metric_eligibility"], "not_evaluable")
        for name in (
            "target_player_request",
            "identity_manual_audit",
            "identity_run_receipt",
            "identity_evaluation",
        ):
            schema = load_identity_json_schema(name)
            self.assertEqual(schema["type"], "object")


if __name__ == "__main__":
    unittest.main()
