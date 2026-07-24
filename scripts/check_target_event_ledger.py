#!/usr/bin/env python3
"""Validate Stage 13C target event ledger."""

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
RUNTIME_ROOT = Path("/home/fdoblak/workspace/target_event_ledger_checks")
GATE = "PASS — TARGET EVENT LEDGER ACTIVE"
GATE_FAIL = "NO-GO — TARGET EVENT LEDGER FAILURE"


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


def run_checks(*, keep: bool) -> Result:
    from football_analytics.events.fixtures import source_events_fixture
    from football_analytics.events.ledger_service import build_target_event_ledger

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="led13c_", dir=str(RUNTIME_ROOT)))
    try:
        out = session / "full"
        r = build_target_event_ledger(output_dir=out, sources=source_events_fixture("full_package"))
        if not r.accepted:
            result.fail("01_full", str(r.error_code))
        else:
            result.ok("01_full")
        out2 = session / "dup"
        r2 = build_target_event_ledger(
            output_dir=out2, sources=source_events_fixture("duplicate_overlap")
        )
        if r2.summary.get("suppressed_count", 0) < 1:
            result.fail("02_dedup", str(r2.summary))
        else:
            result.ok("02_dedup")
        # source preservation
        if any(not row.get("source_event_id") for row in r.ledger):
            result.fail("03_source_preservation", "missing source ids")
        else:
            result.ok("03_source_preservation")
        # no destructive merge / append only
        if (
            r.summary.get("append_only") is not True
            or r.summary.get("no_destructive_merge") is not True
        ):
            result.fail("04_append_only", str(r.summary))
        else:
            result.ok("04_append_only")
        out3 = session / "replay"
        r3 = build_target_event_ledger(
            output_dir=out3, sources=source_events_fixture("replay_blocked")
        )
        if any(row.get("live_event_eligible") for row in r3.ledger):
            result.fail("05_replay_blocks_live", "eligible on cut/replay")
        else:
            result.ok("05_replay_blocks_live")
    except Exception as exc:  # noqa: BLE001
        result.fail("99_unexpected", str(exc))
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
    payload["gate"] = GATE if result.exit_code == EXIT_PASS else GATE_FAIL
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
