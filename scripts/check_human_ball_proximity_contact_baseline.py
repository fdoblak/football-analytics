#!/usr/bin/env python3
"""Validate Stage 10B human-ball proximity / contact-candidate baseline."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_ball_proximity_contact_checks")
GATE_PASS = "PASS — HUMAN BALL PROXIMITY CONTACT BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — HUMAN BALL PROXIMITY CONTACT BASELINE ACTIVE"
GATE_FAIL = "NO-GO — HUMAN BALL PROXIMITY CONTACT BASELINE FAILURE"


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
    from football_analytics.interaction.evaluation import NOT_EVALUATED_INTERACTION
    from football_analytics.interaction.proximity_config import (
        load_proximity_baseline_config,
        proximity_baseline_config_fingerprint,
    )
    from football_analytics.interaction.proximity_fixtures import load_fixture
    from football_analytics.interaction.proximity_service import (
        compute_human_ball_proximity_contact,
    )
    from football_analytics.interaction.semantics import nearest_player_is_possession

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="prox10b_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_proximity_baseline_config()
        fp = proximity_baseline_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if proximity_baseline_config_fingerprint(load_proximity_baseline_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        # controlled carry → multi-frame contact candidate
        out1 = session / "carry"
        fx1 = load_fixture("controlled_carry")
        r1 = compute_human_ball_proximity_contact(output_dir=out1, points=fx1["points"])
        if not r1.accepted:
            result.fail("01_controlled_carry", str(r1.error_code))
        else:
            import pyarrow.parquet as pq

            contacts = pq.read_table(r1.contact_parquet).to_pylist()
            multi = [c for c in contacts if c.get("multi_frame_support")]
            if multi and all(c.get("implies_controlled_possession") is False for c in contacts):
                result.ok("01_controlled_carry")
            else:
                result.fail("01_controlled_carry", "no multi-frame or implies possession")

        # nearest ≠ possession
        out2 = session / "nearest"
        fx2 = load_fixture("false_nearest")
        r2 = compute_human_ball_proximity_contact(output_dir=out2, points=fx2["points"])
        if r2.accepted:
            import pyarrow.parquet as pq

            prox = pq.read_table(r2.proximity_parquet).to_pylist()
            nearest = [p for p in prox if p.get("is_nearest_human")]
            if (
                nearest
                and all(p.get("nearest_implies_possession") is False for p in nearest)
                and not nearest_player_is_possession(is_nearest=True)
            ):
                result.ok("02_nearest_not_possession")
            else:
                result.fail("02_nearest_not_possession", "nearest implies possession")
        else:
            result.fail("02_nearest_not_possession", str(r2.error_code))

        # single frame ≠ contact
        out3 = session / "single"
        fx3 = load_fixture("single_frame")
        r3 = compute_human_ball_proximity_contact(output_dir=out3, points=fx3["points"])
        if r3.accepted:
            import pyarrow.parquet as pq

            contacts = pq.read_table(r3.contact_parquet).to_pylist()
            if (
                contacts
                and all(c.get("contact_state") == "rejected" for c in contacts)
                or not contacts
            ):
                result.ok("03_single_frame_rejected")
            else:
                result.fail(
                    "03_single_frame_rejected", str([c.get("contact_state") for c in contacts])
                )
        else:
            result.fail("03_single_frame_rejected", str(r3.error_code))

        # missing ball
        out4 = session / "missing"
        fx4 = load_fixture("missing_ball")
        r4 = compute_human_ball_proximity_contact(output_dir=out4, points=fx4["points"])
        if r4.accepted:
            import pyarrow.parquet as pq

            prox = pq.read_table(r4.proximity_parquet).to_pylist()
            if any("MISSING_BALL_NOT_NO_POSSESSION" in (p.get("reason_codes") or []) for p in prox):
                result.ok("04_missing_ball")
            else:
                result.fail("04_missing_ball", "missing reason absent")
        else:
            result.fail("04_missing_ball", str(r4.error_code))

        # replay excluded
        out5 = session / "replay"
        fx5 = load_fixture("replay")
        r5 = compute_human_ball_proximity_contact(output_dir=out5, points=fx5["points"])
        if r5.accepted:
            import pyarrow.parquet as pq

            prox = pq.read_table(r5.proximity_parquet).to_pylist()
            if all(str(p.get("eligibility_status")) == "excluded" for p in prox):
                result.ok("05_replay_excluded")
            else:
                result.fail("05_replay_excluded", "replay not excluded")
        else:
            result.fail("05_replay_excluded", str(r5.error_code))

        # airborne unknown blocks pitch
        out6 = session / "air"
        fx6 = load_fixture("airborne_unknown")
        r6 = compute_human_ball_proximity_contact(output_dir=out6, points=fx6["points"])
        if r6.accepted:
            import pyarrow.parquet as pq

            prox = pq.read_table(r6.proximity_parquet).to_pylist()
            if all(p.get("pitch_distance_usable") is False for p in prox):
                result.ok("06_airborne_blocks_pitch")
            else:
                result.fail("06_airborne_blocks_pitch", "pitch usable")
        else:
            result.fail("06_airborne_blocks_pitch", str(r6.error_code))

        # evaluation not claimed
        if r1.accepted and r1.evaluation_json:
            ev = json.loads(Path(r1.evaluation_json).read_text(encoding="utf-8"))
            if ev.get("ground_truth_evaluation_status") == NOT_EVALUATED_INTERACTION:
                result.ok("07_not_evaluated_gt")
            else:
                result.fail("07_not_evaluated_gt", str(ev.get("ground_truth_evaluation_status")))
        else:
            result.fail("07_not_evaluated_gt", "no eval")

        # deterministic repeat
        out7 = session / "rep1"
        out8 = session / "rep2"
        fx = load_fixture("controlled_carry", run_id=fx1["run_id"])
        a = compute_human_ball_proximity_contact(output_dir=out7, points=fx["points"])
        b = compute_human_ball_proximity_contact(output_dir=out8, points=fx["points"])
        if a.accepted and b.accepted and a.config_fingerprint == b.config_fingerprint:
            sa = json.loads(Path(a.summary_json).read_text(encoding="utf-8"))
            sb = json.loads(Path(b.summary_json).read_text(encoding="utf-8"))
            if sa["proximity_row_count"] == sb["proximity_row_count"]:
                result.ok("08_deterministic_repeat")
            else:
                result.fail("08_deterministic_repeat", "count mismatch")
        else:
            result.fail("08_deterministic_repeat", "compute fail")

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
