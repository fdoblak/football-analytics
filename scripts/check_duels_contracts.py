#!/usr/bin/env python3
"""Validate Stage 12A duels / competitive-events contracts.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/duels_contract_checks")
GATE_PASS = "PASS — DUELS CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — DUELS CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — DUELS CONTRACT FAILURE"


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
    from football_analytics.duels.contracts import (
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_duels_contracts_registered,
        assert_frozen_upstream_fingerprints,
        duels_schema_fingerprints,
        load_duels_json_schema,
        validate_against_json_schema,
    )
    from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS, evaluate_duels
    from football_analytics.duels.fixtures import (
        coverage_example,
        long_ball_alone_rows,
        monocular_aerial_rows,
        nearby_opponent_alone_rows,
        nearest_switch_alone_rows,
        single_target_duels_bundle,
    )
    from football_analytics.duels.policy import (
        assert_contract_only_policy,
        load_duels_policy,
        policy_fingerprint,
    )
    from football_analytics.duels.receipt import (
        build_synthetic_quality,
        build_synthetic_receipt,
        build_synthetic_request,
        build_synthetic_review_queue,
        validate_quality_payload,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.duels.semantics import (
        long_ball_alone_is_clearance,
        monocular_aerial_allows_exact_height,
        nearby_opponent_alone_is_take_on,
        nearest_switch_alone_is_duel_outcome,
    )
    from football_analytics.duels.validation import recount_duels_counts, validate_duels_bundle

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="duels12a_", dir=str(RUNTIME_ROOT)))

    try:
        assert_duels_contracts_registered()
        assert_frozen_upstream_fingerprints()
        if len(list_contracts()) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err("registry contract count mismatch", integrity=True)

        policy = load_duels_policy()
        assert_contract_only_policy(policy)
        pol_fp = policy_fingerprint(policy)
        result.extras["duels_policy_fp"] = pol_fp
        result.extras["schema_fps"] = duels_schema_fingerprints()

        b1 = single_target_duels_bundle(pol_fp)
        vr1 = validate_duels_bundle(
            take_ons=b1["take_on_attempts"],
            ground_duels=b1["ground_duel_candidates"],
            aerial_duels=b1["aerial_duel_candidates"],
            tackles=b1["tackle_events"],
            recoveries=b1["recovery_events"],
            turnovers=b1["turnover_events"],
            clearances=b1["clearance_events"],
            policy=policy,
            expected_run_id=b1["run_id"],
            expected_video_id=b1["video_id"],
        )
        if vr1.status == "PASS":
            result.ok_scenario("01_single_target_duels_bundle")
        else:
            result.fail_scenario("01_single_target_duels_bundle", str(vr1.errors))

        if not nearby_opponent_alone_is_take_on(nearby_opponent_alone=True):
            near = nearby_opponent_alone_rows(pol_fp)
            if near["take_on_rows"][0]["implies_take_on"] is False:
                result.ok_scenario("02_nearby_opponent_alone_not_take_on")
            else:
                result.fail_scenario("02_nearby_opponent_alone_not_take_on", "implies take-on")
        else:
            result.fail_scenario("02_nearby_opponent_alone_not_take_on", "semantic broken")

        if not nearest_switch_alone_is_duel_outcome(nearest_switch_alone=True):
            sw = nearest_switch_alone_rows(pol_fp)
            if sw["ground_rows"][0]["implies_duel_outcome"] is False:
                result.ok_scenario("03_nearest_switch_alone_not_duel_outcome")
            else:
                result.fail_scenario("03_nearest_switch_alone_not_duel_outcome", "implies outcome")
        else:
            result.fail_scenario("03_nearest_switch_alone_not_duel_outcome", "semantic broken")

        if not monocular_aerial_allows_exact_height(monocular_only=True):
            air = monocular_aerial_rows(pol_fp)
            row = air["aerial_rows"][0]
            if row["exact_3d_height_m"] is None and row["exact_3d_height_claimed"] is False:
                result.ok_scenario("04_monocular_aerial_no_exact_height")
            else:
                result.fail_scenario("04_monocular_aerial_no_exact_height", "height claimed")
        else:
            result.fail_scenario("04_monocular_aerial_no_exact_height", "semantic broken")

        if not long_ball_alone_is_clearance(long_ball_alone=True):
            clr = long_ball_alone_rows(pol_fp)
            if clr["clearance_rows"][0]["implies_clearance"] is False:
                result.ok_scenario("05_long_ball_alone_not_clearance")
            else:
                result.fail_scenario("05_long_ball_alone_not_clearance", "implies clearance")
        else:
            result.fail_scenario("05_long_ball_alone_not_clearance", "semantic broken")

        req = build_synthetic_request(
            run_id=b1["run_id"], video_id=b1["video_id"], duels_policy_fingerprint=pol_fp
        )
        receipt = build_synthetic_receipt(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            duels_policy_fingerprint=pol_fp,
            take_ons=b1["take_on_rows"],
            ground_duels=b1["ground_rows"],
            aerial_duels=b1["aerial_rows"],
            tackles=b1["tackle_rows"],
            recoveries=b1["recovery_rows"],
            turnovers=b1["turnover_rows"],
            clearances=b1["clearance_rows"],
            coverage_summary=coverage_example(),
        )
        quality = build_synthetic_quality(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            coverage=coverage_example(),
            duels_policy_fingerprint=pol_fp,
        )
        try:
            validate_request_payload(req)
            validate_receipt_payload(receipt)
            validate_quality_payload(quality)
            result.ok_scenario("06_request_receipt_quality_schemas")
        except Exception as exc:  # noqa: BLE001
            result.fail_scenario("06_request_receipt_quality_schemas", str(exc))

        mismatches = recount_duels_counts(
            take_ons=b1["take_on_rows"],
            ground_duels=b1["ground_rows"],
            aerial_duels=b1["aerial_rows"],
            tackles=b1["tackle_rows"],
            recoveries=b1["recovery_rows"],
            turnovers=b1["turnover_rows"],
            clearances=b1["clearance_rows"],
            receipt=receipt,
        )
        if not mismatches:
            result.ok_scenario("07_receipt_recount")
        else:
            result.fail_scenario("07_receipt_recount", str(mismatches))

        ev = evaluate_duels(
            take_ons=b1["take_on_rows"],
            ground_duels=b1["ground_rows"],
            aerial_duels=b1["aerial_rows"],
        )
        if ev.ground_truth_evaluation_status == NOT_EVALUATED_DUELS:
            result.ok_scenario("08_not_evaluated_without_gt")
        else:
            result.fail_scenario("08_not_evaluated_without_gt", ev.ground_truth_evaluation_status)

        for name in (
            "duels_request",
            "duels_run_receipt",
            "duels_evaluation",
            "duels_quality",
            "manual_review_queue",
        ):
            try:
                schema = load_duels_json_schema(name)
                if name == "manual_review_queue":
                    payload = build_synthetic_review_queue(
                        run_id=b1["run_id"], video_id=b1["video_id"]
                    )
                    validate_against_json_schema(payload, schema)
                elif name == "duels_evaluation":
                    validate_against_json_schema(
                        ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"]), schema
                    )
            except Exception as exc:  # noqa: BLE001
                result.fail_scenario(f"09_json_schema_{name}", str(exc))
                break
        else:
            result.ok_scenario("09_json_schemas_load")

        out = session / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        p = out / "receipt.json"
        write_json_record(p, receipt, overwrite=False)
        try:
            write_json_record(p, receipt, overwrite=False)
            result.fail_scenario("10_no_overwrite", "overwrite allowed")
        except Exception:
            result.ok_scenario("10_no_overwrite")

        evidence_dir = REPO_ROOT / "artifacts" / "evidence" / "stage_12"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence_dir / "stage_12a_validator_summary.json",
            result.to_dict(),
            overwrite=True,
        )
        write_json_record(
            evidence_dir / "stage_12a_schema_fingerprints.json",
            {"schema_version": 1, "fingerprints": result.extras.get("schema_fps", {})},
            overwrite=True,
        )
        write_json_record(evidence_dir / "stage_12a_request.json", req, overwrite=True)
        write_json_record(evidence_dir / "stage_12a_receipt.json", receipt, overwrite=True)
        write_json_record(evidence_dir / "stage_12a_quality.json", quality, overwrite=True)
        write_json_record(
            evidence_dir / "stage_12a_evaluation.json",
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
