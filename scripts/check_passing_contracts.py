#!/usr/bin/env python3
"""Validate Stage 11A passing / reception / progression contracts.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/passing_contract_checks")
GATE_PASS = "PASS — PASSING CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — PASSING CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — PASSING CONTRACT FAILURE"


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


def run_checks(*, keep: bool, strict: bool) -> Result:
    from football_analytics.core.records import write_json_record
    from football_analytics.data.compiler import list_contracts
    from football_analytics.passing.contracts import (
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_frozen_upstream_fingerprints,
        assert_passing_contracts_registered,
        load_passing_json_schema,
        passing_schema_fingerprints,
        validate_against_json_schema,
    )
    from football_analytics.passing.evaluation import (
        NOT_EVALUATED_PASSING,
        evaluate_passing,
    )
    from football_analytics.passing.fixtures import (
        coverage_example,
        cut_replay_rows,
        owner_change_alone_rows,
        penalty_presence_only_rows,
        single_target_pass_bundle,
    )
    from football_analytics.passing.policy import (
        assert_contract_only_policy,
        load_passing_policy,
        policy_fingerprint,
    )
    from football_analytics.passing.receipt import (
        build_attack_direction_evidence,
        build_synthetic_quality,
        build_synthetic_receipt,
        build_synthetic_request,
        build_synthetic_review_queue,
        recount_passing_counts,
        validate_quality_payload,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.passing.semantics import (
        attack_direction_unknown_blocks_directional,
        box_touch_eligible,
        cut_replay_gap_allows_pass,
        owner_change_alone_is_completed_pass,
        penalty_presence_is_box_touch,
    )
    from football_analytics.passing.validation import validate_passing_bundle

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pass11a_", dir=str(RUNTIME_ROOT)))

    try:
        assert_passing_contracts_registered()
        assert_frozen_upstream_fingerprints()
        if len(list_contracts()) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err("registry contract count mismatch", integrity=True)

        policy = load_passing_policy()
        assert_contract_only_policy(policy)
        pol_fp = policy_fingerprint(policy)
        result.extras["passing_policy_fp"] = pol_fp
        result.extras["schema_fps"] = passing_schema_fingerprints()

        b1 = single_target_pass_bundle(pol_fp)
        vr1 = validate_passing_bundle(
            passes=b1["pass_candidates"],
            receptions=b1["reception_candidates"],
            outcomes=b1["pass_outcomes"],
            progression=b1["ball_progression_segments"],
            touches=b1["target_ball_touches"],
            policy=policy,
            expected_run_id=b1["run_id"],
            expected_video_id=b1["video_id"],
        )
        if vr1.status == "PASS":
            result.ok_scenario("01_single_target_pass_bundle")
        else:
            result.fail_scenario("01_single_target_pass_bundle", str(vr1.errors))

        if not owner_change_alone_is_completed_pass(owner_changed=True):
            oc = owner_change_alone_rows(pol_fp)
            if oc["pass_rows"][0]["implies_completed_pass"] is False:
                result.ok_scenario("02_owner_change_alone_not_pass")
            else:
                result.fail_scenario("02_owner_change_alone_not_pass", "implies completed")
        else:
            result.fail_scenario("02_owner_change_alone_not_pass", "semantic broken")

        if not cut_replay_gap_allows_pass(cut_or_replay=True, hard_gap=False):
            cr = cut_replay_rows(pol_fp)
            if cr["pass_rows"][0]["candidate_state"] == "rejected":
                result.ok_scenario("03_cut_replay_no_pass")
            else:
                result.fail_scenario("03_cut_replay_no_pass", "not rejected")
        else:
            result.fail_scenario("03_cut_replay_no_pass", "semantic broken")

        if attack_direction_unknown_blocks_directional(attack_direction="unknown"):
            row = b1["outcome_rows"][0]
            if (
                row["attack_relative_evaluable"] is False
                and row["progression_1_to_2"] == "not_evaluable"
            ):
                result.ok_scenario("04_attack_direction_unknown_blocks")
            else:
                result.fail_scenario("04_attack_direction_unknown_blocks", "directional set")
        else:
            result.fail_scenario("04_attack_direction_unknown_blocks", "semantic broken")

        if not penalty_presence_is_box_touch(in_penalty=True):
            pr = penalty_presence_only_rows(pol_fp)
            t = pr["touch_rows"][0]
            if t["is_box_touch_candidate"] is False and t["penalty_presence_alone"] is True:
                result.ok_scenario("05_penalty_presence_not_box_touch")
            else:
                result.fail_scenario("05_penalty_presence_not_box_touch", "counted as box touch")
        else:
            result.fail_scenario("05_penalty_presence_not_box_touch", "semantic broken")

        if box_touch_eligible(
            in_penalty=True,
            has_possession_or_contact=True,
            has_pitch_mapping=True,
            playability_status="playable",
        ):
            result.ok_scenario("06_box_touch_requires_contact_pitch_playable")
        else:
            result.fail_scenario("06_box_touch_requires_contact_pitch_playable", "eligible false")

        req = build_synthetic_request(
            run_id=b1["run_id"], video_id=b1["video_id"], passing_policy_fingerprint=pol_fp
        )
        receipt = build_synthetic_receipt(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            passing_policy_fingerprint=pol_fp,
            passes=b1["pass_rows"],
            receptions=b1["reception_rows"],
            outcomes=b1["outcome_rows"],
            progression=b1["progression_rows"],
            touches=b1["touch_rows"],
            coverage_summary=coverage_example(),
        )
        quality = build_synthetic_quality(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            coverage=coverage_example(),
            passing_policy_fingerprint=pol_fp,
        )
        try:
            validate_request_payload(req)
            validate_receipt_payload(receipt)
            validate_quality_payload(quality)
            result.ok_scenario("07_request_receipt_quality_schemas")
        except Exception as exc:  # noqa: BLE001
            result.fail_scenario("07_request_receipt_quality_schemas", str(exc))

        mismatches = recount_passing_counts(
            passes=b1["pass_rows"],
            receptions=b1["reception_rows"],
            outcomes=b1["outcome_rows"],
            progression=b1["progression_rows"],
            touches=b1["touch_rows"],
            receipt=receipt,
        )
        if not mismatches:
            result.ok_scenario("08_receipt_recount")
        else:
            result.fail_scenario("08_receipt_recount", str(mismatches))

        ev = evaluate_passing(
            passes=b1["pass_rows"],
            receptions=b1["reception_rows"],
            outcomes=b1["outcome_rows"],
        )
        if ev.ground_truth_evaluation_status == NOT_EVALUATED_PASSING:
            result.ok_scenario("09_not_evaluated_without_gt")
        else:
            result.fail_scenario("09_not_evaluated_without_gt", ev.ground_truth_evaluation_status)

        for name in (
            "passing_request",
            "passing_run_receipt",
            "passing_evaluation",
            "passing_quality",
            "attack_direction_evidence",
            "manual_review_queue",
        ):
            try:
                schema = load_passing_json_schema(name)
                if name == "attack_direction_evidence":
                    payload = build_attack_direction_evidence(
                        run_id=b1["run_id"], video_id=b1["video_id"]
                    )
                    validate_against_json_schema(payload, schema)
                elif name == "manual_review_queue":
                    payload = build_synthetic_review_queue(
                        run_id=b1["run_id"], video_id=b1["video_id"]
                    )
                    validate_against_json_schema(payload, schema)
                elif name == "passing_evaluation":
                    validate_against_json_schema(
                        ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"]), schema
                    )
            except Exception as exc:  # noqa: BLE001
                result.fail_scenario(f"10_json_schema_{name}", str(exc))
                break
        else:
            result.ok_scenario("10_json_schemas_load")

        # Atomic no-overwrite smoke
        out = session / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        p = out / "receipt.json"
        write_json_record(p, receipt, overwrite=False)
        try:
            write_json_record(p, receipt, overwrite=False)
            result.fail_scenario("11_no_overwrite", "overwrite allowed")
        except Exception:
            result.ok_scenario("11_no_overwrite")

        # Evidence JSON
        evidence_dir = REPO_ROOT / "artifacts" / "evidence" / "stage_11"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence_dir / "stage_11a_validator_summary.json",
            result.to_dict(),
            overwrite=True,
        )
        write_json_record(
            evidence_dir / "stage_11a_schema_fingerprints.json",
            {"schema_version": 1, "fingerprints": result.extras.get("schema_fps", {})},
            overwrite=True,
        )
        write_json_record(evidence_dir / "stage_11a_request.json", req, overwrite=True)
        write_json_record(evidence_dir / "stage_11a_receipt.json", receipt, overwrite=True)
        write_json_record(evidence_dir / "stage_11a_quality.json", quality, overwrite=True)
        write_json_record(
            evidence_dir / "stage_11a_evaluation.json",
            ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"]),
            overwrite=True,
        )

    except Exception as exc:  # noqa: BLE001
        result.err(f"config/runtime failure: {exc}", config=True)
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)

    return result.finalize(strict=strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), strict=bool(args.strict))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if result.status == "PASS":
        print(GATE_PASS)
    elif result.status == "PASS_WITH_WARNINGS":
        print(GATE_FINDINGS)
    else:
        print(GATE_FAIL)
        for e in result.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
