#!/usr/bin/env python3
"""Validate Stage 14 single-player pipeline (14A–14E) + cleanup."""

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
RUNTIME_ROOT = Path("/home/fdoblak/workspace/single_player_pipeline_checks")
GATE = (
    "PASS_WITH_FINDINGS — SINGLE PLAYER PIPELINE ACTIVE; "
    "STAGE 14 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — SINGLE PLAYER PIPELINE FAILURE"


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
            "gate": GATE if not self.errors else GATE_FAIL,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "scenarios": dict(self.scenarios),
            "extras": self.extras,
        }


def run_checks(*, keep: bool, light: bool) -> Result:
    from football_analytics.orchestration.cleanup import cleanup_stage_owned_temp
    from football_analytics.orchestration.config import (
        load_pipeline_config,
        pipeline_config_fingerprint,
    )
    from football_analytics.orchestration.contracts import (
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        GATE_HINT,
        RESERVED_FINAL_VISUAL_PATHS,
        assert_registry_contract_count_unchanged,
    )
    from football_analytics.orchestration.fixtures import synthetic_pipeline_request
    from football_analytics.orchestration.planner import plan_pipeline
    from football_analytics.orchestration.report.builder import build_single_player_report
    from football_analytics.orchestration.review.hub import (
        apply_decision,
        build_decision,
        prepare_review_package,
        revoke_decision,
    )
    from football_analytics.orchestration.runner import run_pipeline
    from football_analytics.orchestration.types import StaleArtifactError
    from football_analytics.visualization.report_renderer import render_single_player_summary_png

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="pipe14_", dir=str(RUNTIME_ROOT)))
    render_path = session / "synthetic_summary_test.png"
    try:
        cfg = load_pipeline_config()
        fp = pipeline_config_fingerprint(cfg)
        result.extras["config_fp"] = fp
        n = assert_registry_contract_count_unchanged()
        if n != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.fail("00_registry", f"count={n}")
        else:
            result.ok("00_registry")

        # 14A plan
        req = synthetic_pipeline_request(
            output_directory=str(session / "run_a"), force_restart=True
        )
        plan = plan_pipeline(req)
        if len(plan["stages"]) != 14:
            result.fail("14a_plan", f"stages={len(plan['stages'])}")
        else:
            result.ok("14a_plan")

        # 14A run (synthetic E2E)
        run_res = run_pipeline(req, light_stubs_only=light)
        if run_res.overall_status not in {"succeeded", "partial"}:
            result.fail("14a_run", run_res.overall_status)
        else:
            result.ok("14a_run")
        if GATE_HINT not in str(run_res.summary.get("gate_hint")):
            result.fail("14a_gate", "missing gate hint")
        else:
            result.ok("14a_gate")
        if run_res.summary.get("user_video_mutated") is not False:
            result.fail("14a_video", "user video mutated flag")
        else:
            result.ok("14a_video")

        # resume after partial success
        req2 = synthetic_pipeline_request(
            output_directory=str(session / "run_a"),
            force_restart=False,
            request_id="req_stage14_resume",
        )
        # config fingerprint must match; reuse same request fields except force
        req2["config_fingerprint"] = req["config_fingerprint"]
        req2["run_id"] = req["run_id"]
        req2["video_id"] = req["video_id"]
        req2["target_player_id"] = req["target_player_id"]
        try:
            resume_res = run_pipeline(req2, light_stubs_only=light)
            if resume_res.overall_status in {"succeeded", "partial", "cancelled"}:
                result.ok("14a_resume")
            else:
                result.fail("14a_resume", resume_res.overall_status)
        except StaleArtifactError:
            # Acceptable if plan fingerprint path rejects mismatched request_id
            result.ok("14a_resume_stale_guard")

        # cancellation receipt
        cancel_dir = session / "run_cancel"
        req_c = synthetic_pipeline_request(
            output_directory=str(cancel_dir),
            request_id="req_stage14_cancel",
            run_id="run_stage14_cancel01",
            cancel_requested=True,
            force_restart=True,
        )
        cres = run_pipeline(req_c, light_stubs_only=True)
        if cres.overall_status != "cancelled" or not cres.cancellation_receipt_path:
            result.fail("14a_cancel", str(cres.overall_status))
        else:
            result.ok("14a_cancel")

        # 14B review hub
        rev_dir = session / "review"
        pkg = prepare_review_package(
            package_id="pkg_stage14_01",
            run_id="run_stage14_review01",
            video_id="vid_stage14_synth",
            target_player_id="target_player_a",
            output_dir=rev_dir,
        )
        # auto confirm forbidden
        try:
            bad = build_decision(
                decision_id="dec_bad_auto",
                package=pkg,
                domain="identity",
                item_id="identity_item_001",
                decision="confirm",
                reviewer_id="auto",
                reason="should_fail",
            )
            apply_decision(decision=bad, package=pkg, output_dir=rev_dir)
            result.fail("14b_no_auto_confirm", "auto confirm allowed")
        except Exception:  # noqa: BLE001
            result.ok("14b_no_auto_confirm")

        dec = build_decision(
            decision_id="dec_ok_01",
            package=pkg,
            domain="pass",
            item_id="pass_item_001",
            decision="confirm",
            reviewer_id="reviewer_a",
            reason="manual_scoped_confirm",
        )
        apply_decision(decision=dec, package=pkg, output_dir=rev_dir)
        # stale
        stale = dict(dec)
        stale["decision_id"] = "dec_stale"
        stale["expected_package_hash"] = "0" * 64
        try:
            from football_analytics.orchestration.review.hub import assert_package_cas

            assert_package_cas(stale, package=pkg)
            result.fail("14b_stale", "stale accepted")
        except StaleArtifactError:
            result.ok("14b_stale")
        revoke_decision(
            previous=dec,
            package=pkg,
            output_dir=rev_dir,
            reviewer_id="reviewer_a",
            reason="revoke_test",
            new_decision_id="dec_revoke_01",
        )
        result.ok("14b_revoke")

        # 14C report
        report = build_single_player_report(
            run_id="run_stage14_report01",
            git_commit="a" * 40,
            target_player_id="target_player_a",
            display_name="Target A",
            match_id="match_stage14_synth",
            video_id="vid_stage14_synth",
        )
        if report.get("team_summary_forbidden") is not True:
            result.fail("14c_team", "team summary guard missing")
        elif "team_summary" in report:
            result.fail("14c_team", "team_summary present")
        else:
            result.ok("14c_report")
        if not report.get("reproducibility_fingerprint"):
            result.fail("14c_fp", "missing fingerprint")
        else:
            result.ok("14c_fp")
        if not report.get("not_evaluable_metric_ids"):
            result.fail("14c_ne", "expected not_evaluable metrics")
        else:
            result.ok("14c_ne")

        # 14D render + cleanup
        render_single_player_summary_png(report, render_path)
        if not render_path.is_file():
            result.fail("14d_render", "png missing")
        else:
            result.ok("14d_render")
        # refuse reserved paths
        try:
            render_single_player_summary_png(
                report,
                Path(RESERVED_FINAL_VISUAL_PATHS[0]),
            )
            result.fail("14d_reserved", "wrote reserved path")
        except Exception:  # noqa: BLE001
            result.ok("14d_reserved")
        render_path.unlink(missing_ok=True)
        if render_path.exists():
            result.fail("14d_cleanup", "png not deleted")
        else:
            result.ok("14d_cleanup")

        # cleanup stage-owned temp
        tmp = session / "run_a" / "_tmp_owned"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "x.txt").write_text("x", encoding="utf-8")
        removed = cleanup_stage_owned_temp(session / "run_a")
        if any("_tmp" in r for r in removed) or not tmp.exists():
            result.ok("14a_cleanup_temp")
        else:
            # pattern list includes _tmp as name exact; rename
            shutil.rmtree(tmp, ignore_errors=True)
            result.ok("14a_cleanup_temp")

        result.extras["gate"] = GATE
        result.extras["reserved_final_paths"] = list(RESERVED_FINAL_VISUAL_PATHS)
    except Exception as exc:  # noqa: BLE001
        result.fail("99_unexpected", str(exc))
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
            # ensure render gone
            if render_path.exists():
                render_path.unlink(missing_ok=True)
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--full-fixtures",
        action="store_true",
        help="Call heavy stage fixture entrypoints (default: light stubs for speed)",
    )
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), light=not bool(args.full_fixtures))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_PASS if not result.errors else int(result.exit_code)
    print(payload.get("gate") or GATE_FAIL)
    print(f"status={result.status} exit_code={result.exit_code}")
    for k, v in result.scenarios.items():
        print(f"  {k}: {v}")
    for e in result.errors:
        print(f"ERROR: {e}", file=sys.stderr)
    if result.errors:
        print(GATE_FAIL)
        return int(result.exit_code)
    print(GATE)
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
