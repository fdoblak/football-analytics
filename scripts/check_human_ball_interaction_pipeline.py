#!/usr/bin/env python3
"""Validate Stage 10D human-ball interaction fusion + Stage 10 close."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_ball_interaction_pipeline_checks")
GATE_PASS = "PASS — HUMAN BALL INTERACTION PIPELINE ACTIVE; STAGE 10 CLOSED"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — HUMAN BALL INTERACTION PIPELINE ACTIVE; "
    "STAGE 10 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — HUMAN BALL INTERACTION PIPELINE FAILURE"


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


def _poss(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def run_checks(*, keep: bool) -> Result:
    from football_analytics.interaction.evaluation import NOT_EVALUATED_INTERACTION
    from football_analytics.interaction.pipeline_config import (
        interaction_pipeline_config_fingerprint,
        load_interaction_pipeline_config,
    )
    from football_analytics.interaction.pipeline_fixtures import load_pipeline_fixture
    from football_analytics.interaction.pipeline_service import (
        integrate_human_ball_interaction,
    )

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pipe10d_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_interaction_pipeline_config()
        fp = interaction_pipeline_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if interaction_pipeline_config_fingerprint(load_interaction_pipeline_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        scenarios = [
            ("01_controlled_carry", "controlled_carry", "confirmed"),
            ("02_loose_ball", "loose_ball", "confirmed"),
            ("03_contested_ball", "contested_ball", "confirmed"),
            ("04_two_player_ambiguity", "two_player_ambiguity", "confirmed"),
            ("05_missing_ball", "missing_ball", "confirmed"),
            ("06_predicted_ball", "predicted_ball", "confirmed"),
            ("07_cut_replay", "cut_replay", "confirmed"),
            ("08_rapid_owner_change", "rapid_owner_change", "confirmed"),
            ("09_false_nearest", "false_nearest", "confirmed"),
            ("10_target_confirmed", "target_confirmed", "confirmed"),
            ("11_target_revoked", "target_revoked", "revoked"),
        ]

        first_ok = None
        for name, fixture_name, identity in scenarios:
            out = session / name
            fx = load_pipeline_fixture(fixture_name)
            r = integrate_human_ball_interaction(
                output_dir=out,
                points=fx["points"],
                identity_status=identity,
            )
            if not r.accepted:
                result.fail(name, str(r.error_code))
                continue
            poss = _poss(r.possession_parquet)
            summary = json.loads(Path(r.summary_json).read_text(encoding="utf-8"))
            if summary.get("event_metrics_produced") is not False:
                result.fail(name, "event metrics claimed")
                continue
            if summary.get("event_accuracy_validated") is not False:
                result.fail(name, "event accuracy claimed")
                continue

            if name == "01_controlled_carry":
                if not (r.proximity_parquet and r.contact_parquet and r.possession_parquet):
                    result.fail(name, "missing fused outputs")
                elif "confirmed" in {p.get("possession_state") for p in poss}:
                    result.fail(name, "auto confirmed")
                else:
                    result.ok(name)
                    first_ok = r
            elif name == "02_loose_ball":
                if any(
                    p.get("possession_state") == "unknown"
                    and "LOOSE_BALL" in list(p.get("reason_codes") or [])
                    for p in poss
                ):
                    result.ok(name)
                else:
                    result.fail(name, str([p.get("reason_codes") for p in poss]))
            elif name in {"03_contested_ball", "04_two_player_ambiguity"}:
                if any(p.get("possession_state") == "contested" for p in poss):
                    result.ok(name)
                else:
                    result.fail(name, str([p.get("possession_state") for p in poss]))
            elif name == "05_missing_ball":
                if any(
                    "MISSING_BALL_NOT_NO_POSSESSION" in list(p.get("reason_codes") or [])
                    for p in poss
                ) and not any("LOOSE_BALL" in list(p.get("reason_codes") or []) for p in poss):
                    result.ok(name)
                else:
                    result.fail(name, str(poss))
            elif name == "06_predicted_ball":
                if any(p.get("possession_state") == "rejected" for p in poss):
                    result.ok(name)
                else:
                    result.fail(name, str([p.get("possession_state") for p in poss]))
            elif name == "07_cut_replay":
                if all(
                    p.get("possession_state") == "not_evaluable"
                    or p.get("termination_reason") in {"replay", "non_playable"}
                    for p in poss
                ):
                    result.ok(name)
                else:
                    result.fail(name, str(poss))
            elif name == "08_rapid_owner_change":
                owners = {
                    p.get("owner_human_track_id")
                    for p in poss
                    if p.get("owner_human_track_id") is not None
                }
                if len(owners) >= 2 or any(
                    p.get("termination_reason") == "owner_transition" for p in poss
                ):
                    result.ok(name)
                else:
                    result.fail(
                        name,
                        str(
                            [
                                (p.get("owner_human_track_id"), p.get("termination_reason"))
                                for p in poss
                            ]
                        ),
                    )
            elif name == "09_false_nearest":
                # multi-player near ball → contested or nearest not owner; never auto confirmed
                if "confirmed" not in {p.get("possession_state") for p in poss}:
                    result.ok(name)
                else:
                    result.fail(name, "confirmed ownership")
            elif name == "10_target_confirmed":
                if (
                    summary.get("identity_status") == "confirmed"
                    and summary.get("overall_status") == "succeeded"
                ):
                    result.ok(name)
                else:
                    result.fail(name, str(summary.get("overall_status")))
            elif name == "11_target_revoked":
                if summary.get("overall_status") == "not_evaluable":
                    result.ok(name)
                else:
                    result.fail(name, str(summary.get("overall_status")))

        # deterministic repeat
        out_a = session / "rep_a"
        out_b = session / "rep_b"
        fx = load_pipeline_fixture("controlled_carry")
        a = integrate_human_ball_interaction(output_dir=out_a, points=fx["points"])
        b = integrate_human_ball_interaction(
            output_dir=out_b,
            points=load_pipeline_fixture("controlled_carry", run_id=fx["run_id"])["points"],
        )
        if a.accepted and b.accepted and a.config_fingerprint == b.config_fingerprint:
            sa = json.loads(Path(a.summary_json).read_text(encoding="utf-8"))
            sb = json.loads(Path(b.summary_json).read_text(encoding="utf-8"))
            if sa["possession_hypothesis_count"] == sb["possession_hypothesis_count"]:
                result.ok("12_deterministic_repeat")
            else:
                result.fail("12_deterministic_repeat", "count mismatch")
        else:
            result.fail("12_deterministic_repeat", "compute fail")

        # NOT_EVALUATED + stage close gate hint
        if first_ok and first_ok.evaluation_json:
            ev = json.loads(Path(first_ok.evaluation_json).read_text(encoding="utf-8"))
            sm = json.loads(Path(first_ok.summary_json).read_text(encoding="utf-8"))
            if ev.get("ground_truth_evaluation_status") == NOT_EVALUATED_INTERACTION:
                result.ok("13_not_evaluated_gt")
            else:
                result.fail("13_not_evaluated_gt", str(ev.get("ground_truth_evaluation_status")))
            if "STAGE 10 CLOSED" in str(sm.get("gate_hint", "")):
                result.ok("14_stage_10_close_hint")
            else:
                result.fail("14_stage_10_close_hint", str(sm.get("gate_hint")))
            if first_ok.review_queue_json and first_ok.receipt_json and first_ok.quality_json:
                result.ok("15_receipt_quality_review")
            else:
                result.fail("15_receipt_quality_review", "missing package files")
        else:
            result.fail("13_not_evaluated_gt", "no first_ok")
            result.fail("14_stage_10_close_hint", "no first_ok")
            result.fail("15_receipt_quality_review", "no first_ok")

        # Real accuracy not validated is an expected finding, not a failure
        result.warn("real_football_accuracy_not_yet_validated")
        result.extras["gate"] = GATE_FINDINGS if not result.errors else GATE_FAIL
        if not result.errors and not result.warnings:
            result.extras["gate"] = GATE_PASS
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
