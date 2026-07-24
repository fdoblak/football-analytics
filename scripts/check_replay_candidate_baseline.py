#!/usr/bin/env python3
"""Validate Stage 13A replay candidate baseline."""

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
RUNTIME_ROOT = Path("/home/fdoblak/workspace/replay_candidate_checks")
GATE = "PASS — REPLAY CANDIDATE BASELINE ACTIVE"
GATE_FAIL = "NO-GO — REPLAY CANDIDATE BASELINE FAILURE"


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
    from football_analytics.events.fixtures import replay_contexts_fixture
    from football_analytics.events.replay_config import (
        load_replay_config,
        replay_config_fingerprint,
    )
    from football_analytics.events.replay_service import compute_replay_candidates

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="rep13a_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_replay_config()
        fp = replay_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        result.ok("00_config")

        out = session / "mixed"
        r = compute_replay_candidates(output_dir=out, contexts=replay_contexts_fixture("mixed"))
        if not r.accepted:
            result.fail("01_mixed", str(r.error_code))
        else:
            live = [x for x in r.replays if x["live_event_eligible"]]
            unk = [x for x in r.replays if x["replay_status"] == "unknown"]
            if not live:
                result.fail("01_mixed", "no live eligible")
            elif not unk:
                result.fail("01_mixed", "expected unknown segment")
            else:
                result.ok("01_mixed")

        out2 = session / "uncertain"
        r2 = compute_replay_candidates(
            output_dir=out2, contexts=replay_contexts_fixture("uncertain_blocks_live")
        )
        if any(x.get("implies_live") or x.get("live_event_eligible") for x in r2.replays):
            result.fail("02_uncertain_blocks_live", "invented live")
        else:
            result.ok("02_uncertain_blocks_live")

        out3 = session / "cam"
        r3 = compute_replay_candidates(
            output_dir=out3, contexts=replay_contexts_fixture("supported_camera")
        )
        by_id = {x["replay_candidate_id"]: x for x in r3.replays}
        if by_id["rep_cam_01"]["camera_position"] != "sideline":
            result.fail("03_supported_camera", str(by_id["rep_cam_01"]["camera_position"]))
        elif by_id["rep_cam_02"]["camera_position"] != "unknown":
            result.fail("03_supported_camera", "crowd should be unknown")
        else:
            result.ok("03_supported_camera")
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
