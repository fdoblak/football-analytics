#!/usr/bin/env python3
"""Validate Stage 12D aerial duel / clearance baseline (synthetic)."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/aerial_clearance_checks")
GATE_PASS = "PASS — AERIAL CLEARANCE BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — AERIAL CLEARANCE BASELINE ACTIVE"
GATE_FAIL = "NO-GO — AERIAL CLEARANCE BASELINE FAILURE"


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
    from football_analytics.duels.aerial_config import (
        aerial_config_fingerprint,
        load_aerial_config,
    )
    from football_analytics.duels.aerial_fixtures import load_fixture
    from football_analytics.duels.aerial_service import compute_aerial_clearance
    from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="aerial12d_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_aerial_config()
        result.extras["config_fp"] = aerial_config_fingerprint(cfg)
        result.ok("00_deterministic_config")

        fx = load_fixture("monocular_aerial")
        r = compute_aerial_clearance(
            output_dir=session / "mono",
            contexts=fx["contexts"],
            run_id=fx.get("run_id"),
            video_id=fx.get("video_id"),
        )
        if not r.accepted:
            result.fail("01_monocular_aerial", str(r.error_code))
        elif r.aerial_duels[0].get("exact_3d_height_m") is not None:
            result.fail("01_monocular_aerial", "exact height present")
        elif r.aerial_duels[0].get("aerial_evaluability") not in {
            "candidate",
            "unknown",
            "not_evaluable",
        }:
            result.fail("01_monocular_aerial", "evaluability too strong")
        else:
            result.ok("01_monocular_aerial")

        fx2 = load_fixture("clearance_with_evidence")
        r2 = compute_aerial_clearance(
            output_dir=session / "clr",
            contexts=fx2["contexts"],
            run_id=fx2.get("run_id"),
            video_id=fx2.get("video_id"),
        )
        if not r2.accepted:
            result.fail("02_clearance_with_evidence", str(r2.error_code))
        elif r2.clearances[0].get("implies_clearance") is not True:
            result.fail("02_clearance_with_evidence", "not implied")
        elif r2.summary.get("evaluation_status") != NOT_EVALUATED_DUELS:
            result.fail("02_clearance_with_evidence", "evaluation status")
        else:
            result.ok("02_clearance_with_evidence")

        fx3 = load_fixture("long_ball_alone")
        r3 = compute_aerial_clearance(
            output_dir=session / "long",
            contexts=fx3["contexts"],
            run_id=fx3.get("run_id"),
            video_id=fx3.get("video_id"),
        )
        if not r3.accepted:
            result.fail("03_long_ball_alone", str(r3.error_code))
        elif r3.clearances[0].get("implies_clearance") is not False:
            result.fail("03_long_ball_alone", "long ball counted as clearance")
        else:
            result.ok("03_long_ball_alone")

        evidence = REPO_ROOT / "artifacts" / "evidence" / "stage_12"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence / "stage_12d_aerial_clearance_summary.json",
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
