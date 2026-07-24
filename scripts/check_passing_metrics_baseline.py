#!/usr/bin/env python3
"""Validate Stage 11C passing metrics baseline (synthetic)."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/passing_metrics_checks")
GATE_PASS = "PASS — PASSING METRICS BASELINE ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — PASSING METRICS BASELINE ACTIVE"
GATE_FAIL = "NO-GO — PASSING METRICS BASELINE FAILURE"


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
    from football_analytics.passing.attack_direction import resolve_attack_direction
    from football_analytics.passing.evaluation import NOT_EVALUATED_PASSING
    from football_analytics.passing.metrics_config import (
        load_metrics_config,
        metrics_config_fingerprint,
    )
    from football_analytics.passing.metrics_service import compute_passing_metrics
    from football_analytics.passing.pass_fixtures import load_fixture
    from football_analytics.passing.pass_service import compute_pass_reception

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pass11c_", dir=str(RUNTIME_ROOT)))
    try:
        cfg = load_metrics_config()
        fp = metrics_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        if metrics_config_fingerprint(load_metrics_config()) != fp:
            result.fail("00_deterministic_config", "fingerprint drift")
        else:
            result.ok("00_deterministic_config")

        # conflict → unknown
        conflict = resolve_attack_direction(
            run_id="run_conflict_01xxxxxxxx",
            video_id="video_synth_01",
            config_direction="toward_goal_a",
            manual_direction="toward_goal_b",
        )
        if (
            conflict.get("attack_direction") == "unknown"
            and conflict.get("conflict") is True
            and conflict.get("invented") is False
        ):
            result.ok("01_attack_direction_conflict_unknown")
        else:
            result.fail("01_attack_direction_conflict_unknown", str(conflict))

        # completed pass metrics with unknown attack direction
        out_b = session / "base_pass"
        fx = load_fixture("completed_pass")
        pr = compute_pass_reception(
            output_dir=out_b,
            transitions=fx["transitions"],
            run_id=fx.get("run_id"),
            video_id=fx.get("video_id"),
        )
        if not pr.accepted:
            result.fail("02_pass_accuracy", str(pr.error_code))
        else:
            out_m = session / "metrics_unknown"
            mr = compute_passing_metrics(
                output_dir=out_m,
                passes=pr.passes,
                receptions=pr.receptions,
                outcomes=pr.outcomes,
                touch_inputs=[
                    {
                        "human_track_id": 1,
                        "touch_time_us": 3_000_000,
                        "touch_x_m": 100.0,
                        "touch_y_m": 34.0,
                        "in_penalty_area": True,
                        "has_possession_or_contact": True,
                        "has_pitch_mapping": True,
                        "playability_status": "playable",
                        "contact_candidate_ids": ["c1"],
                        "evidence_refs": ["e1"],
                    }
                ],
                run_id=pr.summary.get("run_id"),
                video_id=pr.summary.get("video_id"),
            )
            if not mr.accepted:
                result.fail("02_pass_accuracy", str(mr.error_code))
            else:
                m = mr.metrics
                if m["pass_attempts"]["value"] != 1:
                    result.fail("02_pass_accuracy", "attempts")
                elif m["pass_completed"]["value"] != 1:
                    result.fail("02_pass_accuracy", "completed")
                elif m["pass_accuracy"]["status"] != "provisional":
                    result.fail("02_pass_accuracy", "accuracy status")
                elif m["progression_1_to_2"]["status"] != "not_evaluable":
                    result.fail("02_pass_accuracy", "1to2 should be not_evaluable")
                elif m["progression_2_to_3"]["status"] != "not_evaluable":
                    result.fail("02_pass_accuracy", "2to3 should be not_evaluable")
                else:
                    result.ok("02_pass_accuracy_and_directional_not_evaluable")

                if m["box_contact_candidates"]["value"] != 1:
                    result.fail("03_box_touch_eligible", str(m["box_contact_candidates"]))
                else:
                    result.ok("03_box_touch_eligible")

                if m.get("evaluation_status") == NOT_EVALUATED_PASSING:
                    result.ok("04_not_evaluated")
                else:
                    result.fail("04_not_evaluated", str(m.get("evaluation_status")))

        # presence-only not counted
        out_pres = session / "presence"
        mr2 = compute_passing_metrics(
            output_dir=out_pres,
            passes=pr.passes,
            receptions=pr.receptions,
            outcomes=pr.outcomes,
            touch_inputs=[
                {
                    "human_track_id": 1,
                    "touch_time_us": 3_000_000,
                    "in_penalty_area": True,
                    "has_possession_or_contact": False,
                    "has_pitch_mapping": True,
                    "playability_status": "playable",
                }
            ],
            run_id=pr.summary.get("run_id"),
            video_id=pr.summary.get("video_id"),
        )
        if mr2.accepted and mr2.metrics["box_contact_candidates"]["value"] == 0:
            result.ok("05_penalty_presence_not_box_touch")
        else:
            result.fail("05_penalty_presence_not_box_touch", "counted")

        # resolved attack direction enables directional
        out_dir = session / "directional"
        mr3 = compute_passing_metrics(
            output_dir=out_dir,
            passes=pr.passes,
            receptions=pr.receptions,
            outcomes=pr.outcomes,
            run_id=pr.summary.get("run_id"),
            video_id=pr.summary.get("video_id"),
            attack_direction_manual="toward_goal_b",
        )
        if (
            mr3.accepted
            and mr3.metrics["progression_1_to_2"]["status"] == "provisional"
            and mr3.metrics["attack_relative_evaluable"] is True
        ):
            result.ok("06_directional_when_resolved")
        else:
            result.fail("06_directional_when_resolved", str(mr3.metrics.get("progression_1_to_2")))

        # long pass
        out_long_b = session / "long_b"
        fx_long = load_fixture("long_pass")
        pr_long = compute_pass_reception(
            output_dir=out_long_b,
            transitions=fx_long["transitions"],
            run_id=fx_long.get("run_id"),
            video_id=fx_long.get("video_id"),
        )
        out_long_m = session / "long_m"
        mr_long = compute_passing_metrics(
            output_dir=out_long_m,
            passes=pr_long.passes,
            receptions=pr_long.receptions,
            outcomes=pr_long.outcomes,
            run_id=pr_long.summary.get("run_id"),
            video_id=pr_long.summary.get("video_id"),
        )
        if mr_long.accepted and mr_long.metrics["long_pass_attempts"]["value"] == 1:
            result.ok("07_long_pass_metrics")
        else:
            result.fail("07_long_pass_metrics", str(mr_long.metrics.get("long_pass_attempts")))

        evidence = REPO_ROOT / "artifacts" / "evidence" / "stage_11"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_record(
            evidence / "stage_11c_metrics_summary.json",
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
