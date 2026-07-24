#!/usr/bin/env python3
"""Validate Stage 12B take-on / dribble baseline (synthetic)."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/take_on_checks")
GATE_PASS = "PASS — TAKE-ON BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — TAKE-ON BASELINE ACTIVE"
GATE_FAIL = "NO-GO — TAKE-ON BASELINE FAILURE"


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
    from football_analytics.duels.take_on_config import (
        load_take_on_config,
        take_on_config_fingerprint,
    )
    from football_analytics.duels.take_on_fixtures import load_fixture
    from football_analytics.duels.take_on_service import compute_take_ons

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="take12b_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_take_on_config()
        fp = take_on_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if take_on_config_fingerprint(load_take_on_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        for fixture_name, check in (
            ("successful_take_on", lambda r: r.take_ons[0]["implies_take_on"] is True),
            (
                "nearby_opponent_alone",
                lambda r: r.take_ons[0]["implies_take_on"] is False
                and r.take_ons[0]["nearby_opponent_alone"] is True,
            ),
            ("cut_replay", lambda r: r.take_ons[0]["event_state"] == "rejected"),
        ):
            name = f"scenario_{fixture_name}"
            out = session / name
            fx = load_fixture(fixture_name)
            r = compute_take_ons(
                output_dir=out,
                contexts=fx["contexts"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            if not r.accepted:
                result.fail(name, str(r.error_code))
                continue
            if r.summary.get("evaluation_status") != NOT_EVALUATED_DUELS:
                result.fail(name, "evaluation status")
                continue
            if r.take_ons and r.take_ons[0].get("event_state") == "confirmed":
                result.fail(name, "automatic confirmed")
                continue
            if not check(r):
                result.fail(name, "semantic check failed")
                continue
            result.ok(name)

        evidence = REPO_ROOT / "artifacts" / "evidence" / "stage_12"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence / "stage_12b_take_on_summary.json",
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
