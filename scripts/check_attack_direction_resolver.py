#!/usr/bin/env python3
"""Validate Stage 13B attack direction resolver."""

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
RUNTIME_ROOT = Path("/home/fdoblak/workspace/attack_direction_checks")
GATE = "PASS — ATTACK DIRECTION RESOLVER ACTIVE"
GATE_FAIL = "NO-GO — ATTACK DIRECTION RESOLVER FAILURE"


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
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.events.attack_config import (
        attack_config_fingerprint,
        load_attack_config,
    )
    from football_analytics.events.attack_direction import (
        resolve_match_attack_directions,
        resolve_period_attack_direction,
    )

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="atk13b_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_attack_config()
        result.extras["config_fp"] = attack_config_fingerprint(cfg)
        result.ok("00_config")
        rid = generate_run_id()
        conflict = resolve_period_attack_direction(
            run_id=rid,
            video_id="v",
            config_direction="toward_goal_a",
            manual_direction="toward_goal_b",
        )
        if conflict["attack_direction"] != "unknown" or not conflict["conflict"]:
            result.fail("01_conflict_unknown", str(conflict))
        else:
            result.ok("01_conflict_unknown")

        named = resolve_period_attack_direction(
            run_id=rid,
            video_id="v",
            config_direction="toward_goal_a",
            team_display_name="Real Madrid",
        )
        if named["attack_direction"] != "unknown":
            result.fail("02_no_team_names", str(named))
        else:
            result.ok("02_no_team_names")

        periods = resolve_match_attack_directions(
            run_id=rid,
            video_id="v",
            periods=[
                {
                    "period_id": "period_1",
                    "half_id": "first_half",
                    "anonymous_team_id": "anon_a",
                    "config_direction": "toward_goal_b",
                    "apply_half_boundary_flip": False,
                },
                {
                    "period_id": "period_2",
                    "half_id": "second_half",
                    "anonymous_team_id": "anon_a",
                    "config_direction": "toward_goal_b",
                    "apply_half_boundary_flip": True,
                },
            ],
        )
        if periods[0]["attack_direction"] != "toward_goal_b":
            result.fail("03_half_boundary", "first half")
        elif periods[1]["attack_direction"] != "toward_goal_a":
            result.fail("03_half_boundary", str(periods[1]))
        elif not periods[1].get("half_boundary_flipped"):
            result.fail("03_half_boundary", "flip flag")
        else:
            result.ok("03_half_boundary")

        # manual override without conflict
        manual = resolve_period_attack_direction(
            run_id=rid,
            video_id="v",
            config_direction="unknown",
            manual_direction="toward_goal_a",
            anonymous_team_id="anon_a",
        )
        if manual["attack_direction"] != "toward_goal_a" or not manual["manual_override"]:
            result.fail("04_manual_override", str(manual))
        else:
            result.ok("04_manual_override")
        (session / "attack_periods.json").write_text(
            json.dumps(periods, indent=2), encoding="utf-8"
        )
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
