#!/usr/bin/env python3
"""Validate Stage 10C possession / control baseline."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/possession_control_checks")
GATE_PASS = "PASS — POSSESSION CONTROL BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — POSSESSION CONTROL BASELINE ACTIVE"
GATE_FAIL = "NO-GO — POSSESSION CONTROL BASELINE FAILURE"


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


def _read_poss(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def run_checks(*, keep: bool) -> Result:
    from football_analytics.interaction.evaluation import NOT_EVALUATED_INTERACTION
    from football_analytics.interaction.possession_config import (
        load_possession_baseline_config,
        possession_baseline_config_fingerprint,
    )
    from football_analytics.interaction.possession_fixtures import load_fixture
    from football_analytics.interaction.possession_service import compute_possession_control

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="poss10c_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_possession_baseline_config()
        fp = possession_baseline_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if possession_baseline_config_fingerprint(load_possession_baseline_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        # provisional control — never confirmed
        out1 = session / "prov"
        fx1 = load_fixture("provisional_control")
        r1 = compute_possession_control(output_dir=out1, points=fx1["points"])
        if not r1.accepted:
            result.fail("01_provisional_control", str(r1.error_code))
        else:
            poss = _read_poss(r1.possession_parquet)
            states = {p.get("possession_state") for p in poss}
            if "confirmed" in states:
                result.fail("01_provisional_control", "auto confirmed")
            elif "provisional" in states or "candidate" in states:
                if all(p.get("automatic_ceiling") == "provisional" for p in poss):
                    result.ok("01_provisional_control")
                else:
                    result.fail("01_provisional_control", "bad ceiling")
            else:
                result.fail("01_provisional_control", f"states={states}")

        # contested multi-player
        out2 = session / "contested"
        fx2 = load_fixture("contested")
        r2 = compute_possession_control(output_dir=out2, points=fx2["points"])
        if r2.accepted:
            poss = _read_poss(r2.possession_parquet)
            if any(p.get("possession_state") == "contested" for p in poss):
                result.ok("02_contested")
            else:
                result.fail("02_contested", str([p.get("possession_state") for p in poss]))
        else:
            result.fail("02_contested", str(r2.error_code))

        # loose ball → unknown + LOOSE_BALL (not missing)
        out3 = session / "loose"
        fx3 = load_fixture("loose_ball")
        r3 = compute_possession_control(output_dir=out3, points=fx3["points"])
        if r3.accepted:
            poss = _read_poss(r3.possession_parquet)
            if any(
                p.get("possession_state") == "unknown"
                and "LOOSE_BALL" in list(p.get("reason_codes") or [])
                for p in poss
            ):
                result.ok("03_loose_ball")
            else:
                result.fail("03_loose_ball", str(poss))
        else:
            result.fail("03_loose_ball", str(r3.error_code))

        # missing ball ≠ loose / ≠ no possession
        out4 = session / "missing"
        fx4 = load_fixture("missing_ball")
        r4 = compute_possession_control(output_dir=out4, points=fx4["points"])
        if r4.accepted:
            poss = _read_poss(r4.possession_parquet)
            bad = any("LOOSE_BALL" in list(p.get("reason_codes") or []) for p in poss)
            good = any(
                p.get("possession_state") == "not_evaluable"
                and "MISSING_BALL_NOT_NO_POSSESSION" in list(p.get("reason_codes") or [])
                for p in poss
            )
            if good and not bad:
                result.ok("04_missing_ball_not_loose")
            else:
                result.fail("04_missing_ball_not_loose", str(poss))
        else:
            result.fail("04_missing_ball_not_loose", str(r4.error_code))

        # hard gap terminates / splits
        out5 = session / "gap"
        fx5 = load_fixture("hard_gap")
        r5 = compute_possession_control(output_dir=out5, points=fx5["points"])
        if r5.accepted:
            poss = _read_poss(r5.possession_parquet)
            if any(p.get("termination_reason") == "hard_gap" for p in poss) and len(poss) >= 2:
                result.ok("05_hard_gap_terminate")
            else:
                result.fail(
                    "05_hard_gap_terminate",
                    str([(p.get("termination_reason"), p.get("possession_state")) for p in poss]),
                )
        else:
            result.fail("05_hard_gap_terminate", str(r5.error_code))

        # nearest alone ≠ owner
        out6 = session / "nearest"
        fx6 = load_fixture("nearest_not_owner")
        r6 = compute_possession_control(output_dir=out6, points=fx6["points"])
        if r6.accepted:
            poss = _read_poss(r6.possession_parquet)
            owned = [
                p
                for p in poss
                if p.get("possession_state") in {"candidate", "provisional", "confirmed"}
                and p.get("owner_human_track_id") is not None
            ]
            if not owned and any(
                "NEAREST_PLAYER_NOT_POSSESSION" in list(p.get("reason_codes") or []) for p in poss
            ):
                result.ok("06_nearest_not_owner")
            else:
                result.fail("06_nearest_not_owner", str(poss))
        else:
            result.fail("06_nearest_not_owner", str(r6.error_code))

        # replay terminates
        out7 = session / "replay"
        fx7 = load_fixture("replay")
        r7 = compute_possession_control(output_dir=out7, points=fx7["points"])
        if r7.accepted:
            poss = _read_poss(r7.possession_parquet)
            if all(
                p.get("possession_state") == "not_evaluable"
                or p.get("termination_reason") in {"replay", "non_playable"}
                for p in poss
            ):
                result.ok("07_replay_terminate")
            else:
                result.fail("07_replay_terminate", str(poss))
        else:
            result.fail("07_replay_terminate", str(r7.error_code))

        # predicted ball rejected
        out8 = session / "pred"
        fx8 = load_fixture("predicted_ball")
        r8 = compute_possession_control(output_dir=out8, points=fx8["points"])
        if r8.accepted:
            poss = _read_poss(r8.possession_parquet)
            if any(
                p.get("possession_state") == "rejected"
                and "PREDICTED_SOLE_EVIDENCE" in list(p.get("reason_codes") or [])
                for p in poss
            ):
                result.ok("08_predicted_ball")
            else:
                result.fail("08_predicted_ball", str(poss))
        else:
            result.fail("08_predicted_ball", str(r8.error_code))

        # NOT_EVALUATED without GT
        if r1.accepted and r1.evaluation_json:
            ev = json.loads(Path(r1.evaluation_json).read_text(encoding="utf-8"))
            if ev.get("ground_truth_evaluation_status") == NOT_EVALUATED_INTERACTION:
                result.ok("09_not_evaluated_gt")
            else:
                result.fail("09_not_evaluated_gt", str(ev.get("ground_truth_evaluation_status")))
        else:
            result.fail("09_not_evaluated_gt", "no eval")

        # no event implies
        if r1.accepted:
            poss = _read_poss(r1.possession_parquet)
            if all(
                p.get("implies_completed_pass") is False
                and p.get("implies_dribble_or_take_on") is False
                and p.get("implies_duel_or_aerial") is False
                and p.get("implies_box_touch") is False
                and p.get("implies_turnover") is False
                for p in poss
            ):
                result.ok("10_no_event_implies")
            else:
                result.fail("10_no_event_implies", "event flag set")
        else:
            result.fail("10_no_event_implies", "no r1")

        # deterministic repeat
        out9 = session / "rep1"
        out10 = session / "rep2"
        fx = load_fixture("provisional_control", run_id=fx1["run_id"])
        a = compute_possession_control(output_dir=out9, points=fx["points"])
        b = compute_possession_control(output_dir=out10, points=fx["points"])
        if a.accepted and b.accepted and a.config_fingerprint == b.config_fingerprint:
            sa = json.loads(Path(a.summary_json).read_text(encoding="utf-8"))
            sb = json.loads(Path(b.summary_json).read_text(encoding="utf-8"))
            if sa["possession_hypothesis_count"] == sb["possession_hypothesis_count"]:
                result.ok("11_deterministic_repeat")
            else:
                result.fail("11_deterministic_repeat", "count mismatch")
        else:
            result.fail("11_deterministic_repeat", "compute fail")

        result.extras["gate"] = GATE_PASS if not result.errors else GATE_FAIL
        if result.warnings and not result.errors:
            result.extras["gate"] = GATE_FINDINGS
    except Exception as exc:  # noqa: BLE001
        result.err(f"validator crash: {exc}", config=True)
        result.extras["gate"] = GATE_FAIL
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
        else:
            result.extras["session"] = str(session)
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["extras"].get("gate", GATE_FAIL))
        print(f"status={payload['status']} scenarios={len(payload['scenarios'])}")
        for e in payload["errors"]:
            print(f"  - {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
