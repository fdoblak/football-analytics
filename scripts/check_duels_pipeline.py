#!/usr/bin/env python3
"""Validate Stage 12E duels fusion + Stage 12 close."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/duels_pipeline_checks")
GATE_PASS = "PASS — DUELS EVENTS PIPELINE ACTIVE; STAGE 12 CLOSED"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — DUELS EVENTS PIPELINE ACTIVE; "
    "STAGE 12 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — DUELS EVENTS PIPELINE FAILURE"


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


def run_checks(*, keep: bool) -> Result:
    from football_analytics.core.records import write_json_record
    from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS
    from football_analytics.duels.pipeline_config import (
        duels_pipeline_config_fingerprint,
        load_duels_pipeline_config,
    )
    from football_analytics.duels.pipeline_fixtures import load_pipeline_fixture
    from football_analytics.duels.pipeline_service import integrate_duels

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pipe12e_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_duels_pipeline_config()
        fp = duels_pipeline_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if duels_pipeline_config_fingerprint(load_duels_pipeline_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        scenarios = [
            "full_package",
            "nearby_and_switch",
            "take_on_only",
            "ground_only",
            "aerial_only",
        ]
        for fixture_name in scenarios:
            name = f"scenario_{fixture_name}"
            out = session / name
            fx = load_pipeline_fixture(fixture_name)
            r = integrate_duels(
                output_dir=out,
                take_on_contexts=fx.get("take_on_contexts"),
                ground_contexts=fx.get("ground_contexts"),
                aerial_contexts=fx.get("aerial_contexts"),
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            if not r.accepted:
                result.fail(name, str(r.error_code))
                continue
            summary = json.loads(Path(str(r.summary_json)).read_text(encoding="utf-8"))
            if summary.get("real_football_accuracy_validated") is not False:
                result.fail(name, "accuracy claimed")
                continue
            if summary.get("opta_accuracy_validated") is not False:
                result.fail(name, "opta claimed")
                continue
            if summary.get("evaluation_status") != NOT_EVALUATED_DUELS:
                result.fail(name, "evaluation status")
                continue
            if "REAL FOOTBALL ACCURACY NOT YET VALIDATED" not in str(summary.get("gate_hint")):
                result.fail(name, "gate_hint missing")
                continue
            if fixture_name == "nearby_and_switch":
                import pyarrow.parquet as pq

                if r.take_on_parquet:
                    takes = pq.read_table(r.take_on_parquet).to_pylist()
                    if any(t.get("implies_take_on") for t in takes):
                        result.fail(name, "nearby counted as take-on")
                        continue
                if r.clearance_parquet:
                    clrs = pq.read_table(r.clearance_parquet).to_pylist()
                    if any(c.get("implies_clearance") for c in clrs):
                        result.fail(name, "long ball counted as clearance")
                        continue
            result.ok(name)

        # Intentional finding so close gate is PASS_WITH_FINDINGS
        result.warn("real_football_accuracy_not_yet_validated")

        evidence = REPO_ROOT / "artifacts" / "evidence" / "stage_12"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence / "stage_12e_pipeline_summary.json",
            result.to_dict(),
            overwrite=True,
        )
        write_json_record(
            evidence / "stage_12_close_summary.json",
            {
                "schema_version": 1,
                "stage": "12",
                "status": "CLOSED",
                "gate": GATE_FINDINGS,
                "evaluation_status": NOT_EVALUATED_DUELS,
                "real_football_accuracy_validated": False,
                "opta_accuracy_validated": False,
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            },
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001
        result.err(f"config/runtime failure: {exc}", config=True)
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
