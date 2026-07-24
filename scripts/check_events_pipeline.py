#!/usr/bin/env python3
"""Validate Stage 13E target events fusion + Stage 13 close."""

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
RUNTIME_ROOT = Path("/home/fdoblak/workspace/events_pipeline_checks")
GATE = (
    "PASS_WITH_FINDINGS — TARGET EVENTS PIPELINE ACTIVE; "
    "STAGE 13 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — TARGET EVENTS PIPELINE FAILURE"


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
    from football_analytics.events.evaluation import NOT_EVALUATED
    from football_analytics.events.fixtures import pipeline_fixture
    from football_analytics.events.pipeline_config import (
        events_pipeline_config_fingerprint,
        load_events_pipeline_config,
    )
    from football_analytics.events.pipeline_service import integrate_target_events

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pipe13e_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_events_pipeline_config()
        fp = events_pipeline_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        result.ok("00_config")
        for name in ("full_package", "passes_only", "duplicate_overlap"):
            fx = pipeline_fixture(name)
            out = session / name
            r = integrate_target_events(
                output_dir=out,
                sources=fx["sources"],
                replay_contexts=fx["replay_contexts"],
                attack_periods=fx["attack_periods"],
                run_id=fx["run_id"],
                video_id=fx["video_id"],
                interaction_coverage=fx["interaction_coverage"],
            )
            if not r.accepted:
                result.fail(f"scenario_{name}", str(r.error_code))
                continue
            summary = json.loads(Path(str(r.summary_json)).read_text(encoding="utf-8"))
            if summary.get("real_football_accuracy_validated") is not False:
                result.fail(f"scenario_{name}", "accuracy claimed")
                continue
            if summary.get("evaluation_status") != NOT_EVALUATED:
                result.fail(f"scenario_{name}", "evaluation status")
                continue
            if "TARGET EVENTS PIPELINE ACTIVE" not in str(summary.get("gate_hint")):
                result.fail(f"scenario_{name}", "gate_hint")
                continue
            if "REAL FOOTBALL ACCURACY NOT YET VALIDATED" not in str(summary.get("gate_hint")):
                result.fail(f"scenario_{name}", "accuracy disclaimer")
                continue
            result.ok(f"scenario_{name}")
        # recount consistency: ledger_count matches
        fx = pipeline_fixture("full_package")
        r = integrate_target_events(
            output_dir=session / "recount",
            sources=fx["sources"],
            replay_contexts=fx["replay_contexts"],
            attack_periods=fx["attack_periods"],
            run_id=fx["run_id"],
            video_id=fx["video_id"],
        )
        import pyarrow.parquet as pq

        if r.ledger_parquet:
            n = pq.read_table(r.ledger_parquet).num_rows
            if n != r.summary.get("ledger_count"):
                result.fail("recount", f"{n} vs {r.summary.get('ledger_count')}")
            else:
                result.ok("recount")
        else:
            result.fail("recount", "missing ledger")
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
