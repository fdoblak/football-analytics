#!/usr/bin/env python3
"""Validate Stage 13D target event metrics aggregation."""

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
RUNTIME_ROOT = Path("/home/fdoblak/workspace/target_event_metrics_checks")
GATE = "PASS — TARGET EVENT METRICS ACTIVE"
GATE_FAIL = "NO-GO — TARGET EVENT METRICS FAILURE"


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
    from football_analytics.events.metrics_service import compute_target_event_metrics

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="met13d_", dir=str(RUNTIME_ROOT)))
    try:
        led = build_target_event_ledger(
            output_dir=session / "led", sources=source_events_fixture("full_package")
        )
        m = compute_target_event_metrics(
            output_dir=session / "met",
            ledger=led.ledger,
            attack_direction_manual="toward_goal_b",
            interaction_coverage=0.9,
        )
        metrics = m.metrics["metrics"]
        required = [
            "pass_attempts",
            "pass_completion_rate",
            "receptions",
            "progressive_passes_def_to_mid",
            "progressive_passes_mid_to_att",
            "long_pass_attempts",
            "long_pass_completion_rate",
            "dribbles_successful",
            "dribbles_failed",
            "take_on_success_rate",
            "duels_won",
            "duel_win_rate",
            "aerial_duels",
            "aerial_win_rate",
            "tackles_interceptions_recoveries",
            "ball_losses",
            "clearances",
            "penalty_area_ball_touches",
            "interaction_coverage",
        ]
        missing = [k for k in required if k not in metrics]
        if missing:
            result.fail("01_requested_metrics", str(missing))
        else:
            result.ok("01_requested_metrics")
        # each metric has required fields
        bad = []
        for mid, row in metrics.items():
            for field in (
                "value",
                "status",
                "numerator",
                "denominator",
                "definition",
                "definition_version",
                "source_events",
                "coverage",
                "confidence",
                "provenance",
                "review_status",
                "warnings",
            ):
                if field not in row:
                    bad.append(f"{mid}.{field}")
        if bad:
            result.fail("02_metric_fields", str(bad[:8]))
        else:
            result.ok("02_metric_fields")
        # unknown attack → progressive not_evaluable
        m2 = compute_target_event_metrics(
            output_dir=session / "met2",
            ledger=led.ledger,
            attack_direction_manual=None,
            interaction_coverage=0.9,
        )
        prog = m2.metrics["metrics"]["progressive_passes_def_to_mid"]
        if prog.get("status") != "not_evaluable":
            result.fail("03_attack_unknown_blocks", str(prog))
        else:
            result.ok("03_attack_unknown_blocks")
        if metrics["pass_attempts"]["status"] == "not_evaluable":
            result.fail("04_pass_attempts", str(metrics["pass_attempts"]))
        else:
            result.ok("04_pass_attempts")
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
