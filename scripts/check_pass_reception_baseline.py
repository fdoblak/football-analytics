#!/usr/bin/env python3
"""Validate Stage 11B pass / reception baseline (synthetic)."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/passing_reception_checks")
GATE_PASS = "PASS — PASS RECEPTION BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — PASS RECEPTION BASELINE ACTIVE"
GATE_FAIL = "NO-GO — PASS RECEPTION BASELINE FAILURE"


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
    from football_analytics.passing.evaluation import NOT_EVALUATED_PASSING
    from football_analytics.passing.pass_config import (
        load_pass_reception_config,
        pass_reception_config_fingerprint,
    )
    from football_analytics.passing.pass_fixtures import load_fixture
    from football_analytics.passing.pass_service import compute_pass_reception

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pass11b_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_pass_reception_config()
        fp = pass_reception_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if pass_reception_config_fingerprint(load_pass_reception_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        scenarios = [
            ("01_completed_pass", "completed_pass", "completed"),
            ("02_incomplete_pass", "incomplete_pass", "incomplete"),
            ("03_owner_change_alone", "owner_change_alone", "not_evaluable"),
            ("04_cut_replay", "cut_replay", "not_evaluable"),
            ("05_long_pass", "long_pass", "completed"),
            ("06_hard_gap", "hard_gap", "not_evaluable"),
        ]
        for name, fixture_name, expected_outcome in scenarios:
            out = session / name
            fx = load_fixture(fixture_name)
            r = compute_pass_reception(
                output_dir=out,
                transitions=fx["transitions"],
                run_id=fx.get("run_id"),
                video_id=fx.get("video_id"),
            )
            if not r.accepted:
                result.fail(name, str(r.error_code))
                continue
            if not r.outcomes:
                result.fail(name, "no outcomes")
                continue
            outcome = str(r.outcomes[0].get("outcome"))
            if outcome != expected_outcome:
                result.fail(name, f"expected {expected_outcome} got {outcome}")
                continue
            if any(o.get("outcome_state") == "confirmed" for o in r.outcomes):
                result.fail(name, "automatic confirmed")
                continue
            if r.summary.get("evaluation_status") != NOT_EVALUATED_PASSING:
                result.fail(name, "evaluation status wrong")
                continue
            if (
                name == "03_owner_change_alone"
                and r.passes[0].get("implies_completed_pass") is True
            ):
                result.fail(name, "owner change implied completed")
                continue
            if name == "05_long_pass" and r.outcomes[0].get("is_long_pass") is not True:
                result.fail(name, "long pass not flagged")
                continue
            result.ok(name)

        evidence = REPO_ROOT / "artifacts" / "evidence" / "stage_11"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence / "stage_11b_pass_reception_summary.json",
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
