#!/usr/bin/env python3
"""Validate Stage 7A ReID / identity / target-player contracts.

Exit codes:
  0  success (PASS / PASS_WITH_WARNINGS)
  1  validation finding/failure
  2  configuration/schema failure
  3  integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/identity_contract_checks")
GATE_PASS = "PASS — REID AND TARGET IDENTITY CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — REID AND TARGET IDENTITY CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — IDENTITY CONTRACT FAILURE"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}
        self.scenarios: dict[str, str] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok_scenario(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail_scenario(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "scenarios": dict(self.scenarios),
            "extras": self.extras,
        }


def _expect_fail(vr: Any) -> bool:
    return getattr(vr, "status", None) == "FAIL" or bool(getattr(vr, "errors", None))


def _expect_pass(vr: Any) -> bool:
    return getattr(vr, "status", None) != "FAIL"


def run_checks(*, keep: bool, strict: bool) -> Result:
    from football_analytics.core.records import RecordError, write_json_record
    from football_analytics.data.compiler import list_contracts
    from football_analytics.identity.assignments import build_revocation
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
        compile_identity_schemas,
        identity_schema_fingerprints,
        load_identity_json_schema,
        validate_against_json_schema,
    )
    from football_analytics.identity.evaluation import NOT_EVALUATED_IDENTITY, evaluate_identity
    from football_analytics.identity.fixtures import (
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
    from football_analytics.identity.metric_eligibility import (
        customer_metric_allowed,
        resolve_metric_eligibility,
    )
    from football_analytics.identity.policy import (
        decide_assignment_status,
        load_identity_policy,
        policy_fingerprint,
    )
    from football_analytics.identity.receipt import (
        build_synthetic_receipt,
        build_synthetic_target_request,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.identity.reid_links import validate_reid_link
    from football_analytics.identity.review_audit import append_audit_log, read_audit_log
    from football_analytics.identity.types import IdentityContractError
    from football_analytics.identity.validation import validate_identity_bundle

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="id_", dir=str(RUNTIME_ROOT)))
    result.extras["session"] = str(session)

    try:
        assert_identity_contracts_registered()
        names = list_contracts()
        result.extras["contract_count"] = len(names)
        if len(names) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err(
                f"expected {EXPECTED_REGISTRY_CONTRACT_COUNT} contracts, got {len(names)}",
                config=True,
            )

        assert_frozen_upstream_fingerprints()
        fps = identity_schema_fingerprints()
        result.extras["schema_fingerprints"] = fps
        frozen = {
            "team_assignments": EXPECTED_TEAM_ASSIGNMENTS_FP,
            "jersey_observations": EXPECTED_JERSEY_OBSERVATIONS_FP,
            "detections": EXPECTED_DETECTIONS_FP,
            "track_observations": EXPECTED_TRACK_OBSERVATIONS_FP,
            "track_summaries": EXPECTED_TRACK_SUMMARIES_FP,
            "track_lifecycle": EXPECTED_TRACK_LIFECYCLE_FP,
        }
        for k, exp in frozen.items():
            if fps[k] != exp:
                result.err(f"{k} fingerprint regression", integrity=True)

        compile_identity_schemas()
        for js in (
            "target_player_request",
            "identity_manual_audit",
            "identity_run_receipt",
            "identity_evaluation",
        ):
            load_identity_json_schema(js)

        policy = load_identity_policy()
        pol_fp = policy_fingerprint(policy)
        result.extras["policy_fingerprint"] = pol_fp
        if policy["safety"]["face_recognition_forbidden"] is not True:
            result.err("face_recognition_forbidden must be true", integrity=True)

        # --- Scenarios 1–20 ---
        # 1 Manual anchor → confirmed target
        b1 = manual_anchor_bundle(pol_fp)
        vr1 = validate_identity_bundle(
            identity_evidence=b1["identity_evidence"],
            track_identity_assignments=b1["track_identity_assignments"],
            reid_candidate_links=b1["reid_candidate_links"],
            policy=policy,
        )
        if _expect_pass(vr1) and b1["assignment_rows"][0]["assignment_status"] == "confirmed":
            result.ok_scenario("01_manual_anchor_target")
        else:
            result.fail_scenario("01_manual_anchor_target", str(vr1.errors))

        # 2 Two supporting → provisional
        b2 = two_supporting_bundle(pol_fp)
        st, reasons = decide_assignment_status(b2["evidence_rows"], policy=policy)
        if st == "provisional" and _expect_pass(
            validate_identity_bundle(
                identity_evidence=b2["identity_evidence"],
                track_identity_assignments=b2["track_identity_assignments"],
                policy=policy,
            )
        ):
            result.ok_scenario("02_two_supporting_evidence")
        else:
            result.fail_scenario("02_two_supporting_evidence", f"{st}/{reasons}")

        # 3 Jersey alone
        b3 = alone_insufficient_bundle(pol_fp, evidence_type="jersey_number")
        st3, r3 = decide_assignment_status(b3["evidence_rows"], policy=policy)
        if st3 == "candidate" and "JERSEY_ALONE_INSUFFICIENT" in r3:
            result.ok_scenario("03_jersey_alone")
        else:
            result.fail_scenario("03_jersey_alone", f"{st3}/{r3}")

        # 4 Team alone
        b4 = alone_insufficient_bundle(pol_fp, evidence_type="team_assignment")
        st4, r4 = decide_assignment_status(b4["evidence_rows"], policy=policy)
        if st4 == "candidate" and "TEAM_ALONE_INSUFFICIENT" in r4:
            result.ok_scenario("04_team_alone")
        else:
            result.fail_scenario("04_team_alone", f"{st4}/{r4}")

        # 5 Appearance alone
        b5 = alone_insufficient_bundle(pol_fp, evidence_type="appearance_similarity")
        st5, r5 = decide_assignment_status(b5["evidence_rows"], policy=policy)
        if st5 == "candidate" and "APPEARANCE_ALONE_INSUFFICIENT" in r5:
            result.ok_scenario("05_appearance_alone")
        else:
            result.fail_scenario("05_appearance_alone", f"{st5}/{r5}")

        # 6 Conflicting jersey/team
        b6 = conflicting_jersey_team_bundle(pol_fp)
        st6, r6 = decide_assignment_status(b6["evidence_rows"], policy=policy)
        if st6 == "rejected" and "HARD_EVIDENCE_CONFLICT" in r6:
            result.ok_scenario("06_conflicting_jersey_team")
        else:
            result.fail_scenario("06_conflicting_jersey_team", f"{st6}/{r6}")

        # 7 Two target candidates → review
        b7 = two_target_candidates_bundle(pol_fp)
        if all(a["manual_review_required"] for a in b7["assignment_rows"]):
            result.ok_scenario("07_two_target_candidates")
        else:
            result.fail_scenario("07_two_target_candidates", "review not required")

        # 8 Cross-shot candidate link
        b8 = cross_shot_link_bundle(pol_fp, long_gap=False)
        if b8["link_rows"][0]["decision_status"] == "candidate" and _expect_pass(
            validate_identity_bundle(
                identity_evidence=b8["identity_evidence"],
                reid_candidate_links=b8["reid_candidate_links"],
                policy=policy,
            )
        ):
            result.ok_scenario("08_cross_shot_candidate_link")
        else:
            result.fail_scenario("08_cross_shot_candidate_link", "link invalid")

        # 9 Long gap
        b9 = cross_shot_link_bundle(pol_fp, long_gap=True)
        if b9["link_rows"][0]["decision_status"] == "review_required":
            result.ok_scenario("09_long_gap")
        else:
            result.fail_scenario("09_long_gap", "expected review_required")

        # 10 Cross-video auto link rejected
        try:
            validate_reid_link(cross_video_auto_link_rows(pol_fp)[0])
            result.fail_scenario("10_cross_video_auto_link", "should reject")
        except IdentityContractError as exc:
            if "CROSS_VIDEO" in str(exc):
                result.ok_scenario("10_cross_video_auto_link")
            else:
                result.fail_scenario("10_cross_video_auto_link", str(exc))

        # 11 Assignment revoke
        prev = b1["assignment_rows"][0]
        revoked = build_revocation(prev, new_assignment_id="asn_rev_01", reason="CONFLICT")
        if (
            revoked["assignment_status"] == "revoked"
            and revoked["metric_eligibility"] == "not_eligible"
            and not customer_metric_allowed(revoked["metric_eligibility"])
        ):
            result.ok_scenario("11_assignment_revoke")
        else:
            result.fail_scenario("11_assignment_revoke", "revoke ineligible failed")

        # 12 Duplicate confirmed identity
        dup = {
            "identity_evidence": b1["identity_evidence"],
            "track_identity_assignments": None,
            "assignment_rows": b1["assignment_rows"]
            + [
                {
                    **b1["assignment_rows"][0],
                    "assignment_id": "asn_dup",
                    "track_id": 1,
                }
            ],
        }
        from football_analytics.identity.fixtures import _cast

        dup_table = _cast("track_identity_assignments", dup["assignment_rows"])
        vr12 = validate_identity_bundle(
            identity_evidence=b1["identity_evidence"],
            track_identity_assignments=dup_table,
            policy=policy,
        )
        if _expect_fail(vr12) and any("DUPLICATE_CONFIRMED" in e for e in vr12.errors):
            result.ok_scenario("12_duplicate_confirmed_identity")
        else:
            result.fail_scenario("12_duplicate_confirmed_identity", str(vr12.errors))

        # 13 Unknown / low coverage
        b13 = unknown_low_coverage_bundle(pol_fp)
        if b13["assignment_rows"][0]["assignment_status"] == "unknown":
            result.ok_scenario("13_unknown_low_coverage")
        else:
            result.fail_scenario("13_unknown_low_coverage", "expected unknown")

        # 14 Metric eligibility
        elig = resolve_metric_eligibility(
            assignment_status="confirmed",
            target_scope="target",
            has_observed_tracking=True,
            sufficient_coverage=True,
            unresolved_hard_conflict=False,
        )
        not_elig = resolve_metric_eligibility(
            assignment_status="confirmed",
            target_scope="target",
            has_observed_tracking=True,
            sufficient_coverage=True,
            unresolved_hard_conflict=False,
            observation_state="predicted",
        )
        if elig == "eligible" and not_elig == "not_eligible":
            result.ok_scenario("14_metric_eligibility")
        else:
            result.fail_scenario("14_metric_eligibility", f"{elig}/{not_elig}")

        # 15 Evaluation leakage negative
        b15 = leakage_negative_bundle(pol_fp)
        vr15 = validate_identity_bundle(
            identity_evidence=b15["identity_evidence"],
            track_identity_assignments=b15["track_identity_assignments"],
            policy=policy,
        )
        if _expect_fail(vr15) and any("LEAKAGE" in e for e in vr15.errors):
            result.ok_scenario("15_evaluation_leakage_negative")
        else:
            result.fail_scenario("15_evaluation_leakage_negative", str(vr15.errors))

        # 16 Append-only manual audit
        audit_path = session / "identity_manual_audit.jsonl"
        entry = audit_entry(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            target_player_id=b1["target_player_id"],
        )
        append_audit_log(audit_path, entry, contain_root=session)
        entry2 = audit_entry(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            target_player_id=b1["target_player_id"],
            audit_id="audit_02",
            action="revoke",
            previous_decision="confirmed",
            new_decision="revoked",
        )
        append_audit_log(audit_path, entry2, contain_root=session)
        lines = read_audit_log(audit_path)
        if len(lines) == 2 and lines[0]["audit_id"] == "audit_01":
            result.ok_scenario("16_append_only_manual_audit")
        else:
            result.fail_scenario("16_append_only_manual_audit", f"lines={len(lines)}")

        # 17 Hash/FK/version mismatch
        bad_asn = [{**b1["assignment_rows"][0], "evidence_ids": ["missing_ev"]}]
        vr17 = validate_identity_bundle(
            identity_evidence=b1["identity_evidence"],
            track_identity_assignments=_cast("track_identity_assignments", bad_asn),
            policy=policy,
        )
        if _expect_fail(vr17):
            result.ok_scenario("17_hash_fk_version_mismatch")
        else:
            result.fail_scenario("17_hash_fk_version_mismatch", "expected FK fail")

        # 18 Deterministic repeat
        fp_a = policy_fingerprint(load_identity_policy())
        fp_b = policy_fingerprint(load_identity_policy())
        schemas_a = {k: v for k, v in identity_schema_fingerprints().items()}
        schemas_b = {k: v for k, v in identity_schema_fingerprints().items()}
        if fp_a == fp_b and schemas_a == schemas_b:
            result.ok_scenario("18_deterministic_repeat")
        else:
            result.fail_scenario("18_deterministic_repeat", "fingerprint drift")

        # 19 Atomic no-overwrite
        out = session / "receipt.json"
        req = build_synthetic_target_request(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            policy_fingerprint=pol_fp,
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
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            policy_fingerprint=pol_fp,
            assignments=b1["assignment_rows"],
            evidence=b1["evidence_rows"],
            metric_eligible_track_count=1,
            metric_eligible_frame_count=21,
        )
        validate_receipt_payload(receipt)
        write_json_record(out, receipt, overwrite=False)
        try:
            write_json_record(out, receipt, overwrite=False)
            result.fail_scenario("19_atomic_no_overwrite", "overwrite allowed")
        except RecordError:
            result.ok_scenario("19_atomic_no_overwrite")

        # 20 Failure cleanup
        junk = session / "partial_fail_dir"
        junk.mkdir()
        (junk / "tmp.bin").write_bytes(b"x")
        shutil.rmtree(junk)
        if not junk.exists():
            result.ok_scenario("20_failure_cleanup")
        else:
            result.fail_scenario("20_failure_cleanup", "dir remains")

        # Evaluation stub
        ev = evaluate_identity(has_reviewed_ground_truth=False)
        if ev.ground_truth_evaluation_status != NOT_EVALUATED_IDENTITY:
            result.err("evaluation stub status mismatch", config=True)
        result.extras["evaluation_status"] = ev.ground_truth_evaluation_status

        # Schema validate evaluation payload
        validate_against_json_schema(
            ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"], config_fingerprint=pol_fp),
            load_identity_json_schema("identity_evaluation"),
        )

        result.extras["gate"] = GATE_PASS if not result.errors else GATE_FAIL
        if result.warnings and not result.errors:
            result.extras["gate"] = GATE_FINDINGS

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator crash: {exc}", config=True)
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
            result.extras["cleaned"] = True
        else:
            result.extras["cleaned"] = False

    return result.finalize(strict=strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), strict=bool(args.strict))
    payload = result.to_dict()
    gate = payload["extras"].get("gate", GATE_FAIL if result.errors else GATE_PASS)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(gate)
        print(f"status={result.status} exit={result.exit_code}")
        for name, st in sorted(result.scenarios.items()):
            print(f"  {name}: {st}")
        for w in result.warnings:
            print(f"WARN: {w}")
        for e in result.errors:
            print(f"ERR: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
