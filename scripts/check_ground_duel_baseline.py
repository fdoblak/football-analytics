#!/usr/bin/env python3
"""Validate Stage 12C ground duel / tackle / recovery / turnover baseline (synthetic)."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/ground_duel_checks")
GATE_PASS = "PASS — GROUND DUEL BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — GROUND DUEL BASELINE ACTIVE"
GATE_FAIL = "NO-GO — GROUND DUEL BASELINE FAILURE"


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
    from football_analytics.core.records import write_json_record
    from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS
    from football_analytics.duels.ground_config import (
        ground_config_fingerprint,
        load_ground_config,
    )
    from football_analytics.duels.ground_fixtures import load_fixture
    from football_analytics.duels.ground_service import compute_ground_family

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="ground12c_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_ground_config()
        fp = ground_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        result.ok("00_deterministic_config")

        fx = load_fixture("contested_ground")
        r = compute_ground_family(
            output_dir=session / "contested",
            contexts=fx["contexts"],
            run_id=fx.get("run_id"),
            video_id=fx.get("video_id"),
        )
        if not r.accepted:
            result.fail("01_contested_ground", str(r.error_code))
        elif not (r.ground_duels and r.tackles and r.recoveries and r.turnovers):
            result.fail("01_contested_ground", "missing family outputs")
        elif r.summary.get("evaluation_status") != NOT_EVALUATED_DUELS:
            result.fail("01_contested_ground", "evaluation status")
        else:
            result.ok("01_contested_ground")

        fx2 = load_fixture("nearest_switch_alone")
        r2 = compute_ground_family(
            output_dir=session / "switch",
            contexts=fx2["contexts"],
            run_id=fx2.get("run_id"),
            video_id=fx2.get("video_id"),
        )
        if not r2.accepted:
            result.fail("02_nearest_switch_alone", str(r2.error_code))
        elif r2.ground_duels[0].get("implies_duel_outcome") is not False:
            result.fail("02_nearest_switch_alone", "implies outcome")
        else:
            result.ok("02_nearest_switch_alone")

        evidence = REPO_ROOT / "artifacts" / "evidence" / "stage_12"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence / "stage_12c_ground_summary.json",
            result.to_dict(),
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
