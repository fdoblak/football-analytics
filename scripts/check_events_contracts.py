#!/usr/bin/env python3
"""Validate Stage 13 contracts (replay/ledger/revisions)."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/events_contract_checks")


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.scenarios: dict[str, str] = {}
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, config: bool = False) -> None:
        self.errors.append(msg)
        if config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def ok(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self) -> Result:
        if self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
        else:
            self.status = "PASS"
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


GATE_PASS = "PASS — TARGET EVENTS CONTRACTS ACTIVE"


def run_checks(*, keep: bool) -> Result:
    from football_analytics.core.records import write_json_record
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.compiler import list_contracts
    from football_analytics.events.contracts import (
        EVENTS_ARROW_CONTRACTS,
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_events_contracts_registered,
        assert_frozen_upstream_fingerprints,
        compile_events_schemas,
        events_schema_fingerprints,
        load_events_json_schema,
    )
    from football_analytics.events.eligibility import live_event_eligible
    from football_analytics.events.fixtures import source_events_fixture
    from football_analytics.events.ledger_service import build_ledger_rows
    from football_analytics.events.policy import (
        assert_contract_only_policy,
        load_events_policy,
        policy_fingerprint,
    )
    from football_analytics.events.receipt import (
        build_synthetic_quality,
        build_synthetic_receipt,
        build_synthetic_request,
        validate_quality_payload,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.events.validation import validate_events_bundle

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="evt13a_", dir=str(RUNTIME_ROOT)))
    try:
        names = list_contracts()
        if len(names) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.fail(
                "00_registry_count",
                f"expected {EXPECTED_REGISTRY_CONTRACT_COUNT}, got {len(names)}",
            )
        else:
            result.ok("00_registry_count")
        try:
            assert_events_contracts_registered()
            assert_frozen_upstream_fingerprints()
            result.ok("01_contracts_registered_frozen")
        except Exception as exc:  # noqa: BLE001
            result.fail("01_contracts_registered_frozen", str(exc))

        fps = events_schema_fingerprints()
        compile_events_schemas()
        result.extras["fingerprints"] = fps
        result.extras["new_contracts"] = list(EVENTS_ARROW_CONTRACTS)
        result.ok("02_compile_schemas")

        policy = load_events_policy()
        assert_contract_only_policy(policy)
        result.extras["policy_fp"] = policy_fingerprint(policy)
        result.ok("03_policy")

        for name in (
            "events_request",
            "events_run_receipt",
            "events_evaluation",
            "events_quality",
            "manual_review_queue",
            "attack_direction_evidence",
        ):
            load_events_json_schema(name)
        result.ok("04_json_schemas")

        rid = generate_run_id()
        req = build_synthetic_request(run_id=rid, video_id="v13")
        validate_request_payload(req)
        rec = build_synthetic_receipt(run_id=rid, video_id="v13")
        validate_receipt_payload(rec)
        qual = build_synthetic_quality(run_id=rid, video_id="v13")
        validate_quality_payload(qual)
        result.ok("05_receipt_request_quality")

        if live_event_eligible(replay_status="unknown"):
            result.fail("06_unknown_blocks_live", "unknown marked eligible")
        else:
            result.ok("06_unknown_blocks_live")

        sources = source_events_fixture("full_package")
        # normalize run ids
        for rows in sources.values():
            for r in rows:
                r["run_id"] = rid
                r["video_id"] = "v13"
        ledger, revisions = build_ledger_rows(
            sources=sources, policy_fp=result.extras["policy_fp"], run_id=rid, video_id="v13"
        )
        validate_events_bundle(ledger=ledger, revisions=revisions)
        if not any(r.get("suppressed_duplicate") is False for r in ledger):
            result.fail("07_ledger_rows", "no active rows")
        else:
            result.ok("07_ledger_rows")

        write_json_record(
            session / "validator_summary.json", result.finalize().to_dict(), overwrite=True
        )
        write_json_record(session / "schema_fingerprints.json", fps, overwrite=True)
        write_json_record(session / "request.json", req, overwrite=True)
        write_json_record(session / "receipt.json", rec, overwrite=True)
        write_json_record(session / "quality.json", qual, overwrite=True)
    except Exception as exc:  # noqa: BLE001
        result.fail("99_unexpected", str(exc))
        result.finalize()
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep))
    payload = result.to_dict()
    payload["gate"] = (
        GATE_PASS if result.exit_code == EXIT_PASS else "NO-GO — TARGET EVENTS CONTRACTS FAILURE"
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["gate"])
        for k, v in result.scenarios.items():
            print(f"  {k}: {v}")
        for e in result.errors:
            print(f"ERROR: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
