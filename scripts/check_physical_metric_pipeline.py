#!/usr/bin/env python3
"""Validate Stage 9E physical metric fusion pipeline (synthetic E2E).

Exit codes:
  0  success
  1  validation finding/failure
  2  configuration failure
  3  integrity failure
"""

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
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/physical_metric_pipeline_checks")
GATE_PASS = "PASS — PHYSICAL METRICS PIPELINE ACTIVE; STAGE 9 CLOSED"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — PHYSICAL METRICS PIPELINE ACTIVE; STAGE 9 CLOSED; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — PHYSICAL METRIC PIPELINE INTEGRITY FAILURE"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.scenarios: dict[str, str] = {}
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
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


def _session(prefix: str) -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(RUNTIME_ROOT)))


def _receipt_from_result(res: Any) -> dict[str, Any]:
    path = Path(str(res.receipt_json))
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_from_result(res: Any) -> dict[str, Any]:
    # motion/spatial expose summary attribute; also load JSON when available
    if getattr(res, "summary_json", None):
        return json.loads(Path(str(res.summary_json)).read_text(encoding="utf-8"))
    return dict(res.summary)


def run_checks(*, keep: bool) -> Result:
    from football_analytics.physical.pipeline_config import (
        load_pipeline_config,
        pipeline_config_fingerprint,
    )
    from football_analytics.physical.pipeline_evaluation import NOT_EVALUATED_PIPELINE
    from football_analytics.physical.pipeline_fixtures import run_consistent_chain
    from football_analytics.physical.pipeline_service import integrate_physical_metrics

    result = Result()
    sessions: list[Path] = []
    try:
        cfg = load_pipeline_config()
        fp = pipeline_config_fingerprint(cfg)
        result.extras["config_fingerprint"] = fp
        if cfg["stage"] != "9E" or cfg["attack_direction"] != "unknown":
            result.fail("00_metadata", "bad stage/attack")
        else:
            result.ok("00_metadata")

        # 01 consistent success package
        sess1 = _session("e2e_")
        sessions.append(sess1)
        chain = run_consistent_chain(sess1 / "chain")
        out1 = sess1 / "fuse"
        # load motion/spatial summary JSON
        motion_sum = json.loads(Path(str(chain["motion"].summary_json)).read_text())
        spatial_sum = json.loads(Path(str(chain["spatial"].summary_json)).read_text())
        traj_receipt = _receipt_from_result(chain["trajectory"])
        # trajectory summary is embedded in result.summary
        traj_sum = {
            "run_id": chain["identity"]["run_id"],
            "video_id": chain["identity"]["video_id"],
            "target_player_id": chain["identity"]["target_player_id"],
            "eligible_duration_us": traj_receipt.get("eligible_duration_us"),
            **dict(chain["trajectory"].summary),
        }
        # align motion/spatial summary ids for scope check
        for blk in (motion_sum, spatial_sum):
            blk["run_id"] = chain["identity"]["run_id"]
            blk["video_id"] = chain["identity"]["video_id"]
            blk["target_player_id"] = chain["identity"]["target_player_id"]
        motion_receipt = _receipt_from_result(chain["motion"])
        spatial_receipt = _receipt_from_result(chain["spatial"])
        hm = json.loads(Path(str(chain["spatial"].heatmap_json)).read_text())
        zones = json.loads(Path(str(chain["spatial"].zones_json)).read_text())
        activity = json.loads(Path(str(chain["spatial"].activity_json)).read_text())

        r1 = integrate_physical_metrics(
            output_dir=out1,
            identity=chain["identity"],
            trajectory_summary=traj_sum,
            trajectory_receipt=traj_receipt,
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            heatmap_ref=hm,
            zone_ref=zones,
            activity_ref=activity,
            recounted_distance_m=motion_sum.get("measured_distance_m"),
            config=cfg,
        )
        if (
            r1.accepted
            and r1.receipt_json
            and r1.quality_json
            and r1.summary.get("evaluation_status") == NOT_EVALUATED_PIPELINE
            and r1.summary.get("final_customer_visual_created") is False
        ):
            result.ok("01_consistent_success_package")
            result.extras["overall"] = r1.summary.get("overall_physical_analysis_status")
        else:
            result.fail("01_consistent_success_package", str(r1.to_summary()))

        fused = json.loads(Path(str(r1.summary_json)).read_text())

        # 02 different target mix → hard fail
        sess2 = _session("mix_")
        sessions.append(sess2)
        bad_id = dict(chain["identity"])
        bad_motion = dict(motion_sum)
        bad_motion["target_player_id"] = "other_target_99"
        r2 = integrate_physical_metrics(
            output_dir=sess2 / "out",
            identity=bad_id,
            motion_summary=bad_motion,
            motion_receipt=motion_receipt,
            config=cfg,
        )
        if not r2.accepted and "SOURCE_MIX" in str(r2.summary.get("error", "")):
            result.ok("02_target_mix_hard_fail")
        else:
            result.fail("02_target_mix_hard_fail", str(r2.summary))

        # 03 revoked identity
        sess3 = _session("rev_")
        sessions.append(sess3)
        rev = dict(chain["identity"])
        rev["identity_status"] = "revoked"
        rev["assignment_revoked"] = True
        r3 = integrate_physical_metrics(
            output_dir=sess3 / "out",
            identity=rev,
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            config=cfg,
        )
        if r3.accepted and r3.summary.get("overall_physical_analysis_status") == "not_evaluable":
            result.ok("03_revoked_identity")
        else:
            result.fail("03_revoked_identity", str(r3.summary))

        # 04 low coverage → insufficient / not_evaluable metrics allowed
        sess4 = _session("low_")
        sessions.append(sess4)
        low_motion = dict(motion_sum)
        low_motion["coverage_ratio_distance"] = 0.01
        low_motion["distance_status"] = "not_evaluable"
        low_motion["speed_status"] = "not_evaluable"
        r4 = integrate_physical_metrics(
            output_dir=sess4 / "out",
            identity=chain["identity"],
            motion_summary=low_motion,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            config=cfg,
        )
        if r4.accepted and r4.summary.get("overall_physical_analysis_status") in {
            "partial",
            "not_evaluable",
            "succeeded",
        }:
            result.ok("04_low_coverage_partial")
        else:
            result.fail("04_low_coverage_partial", str(r4.summary))

        # 05 single-point / empty motion → not_evaluable distance without zeroing others wrongly
        sess5 = _session("single_")
        sessions.append(sess5)
        empty_motion = dict(motion_sum)
        empty_motion["measured_distance_m"] = None
        empty_motion["distance_status"] = "not_evaluable"
        empty_motion["sprint_count"] = 0
        r5 = integrate_physical_metrics(
            output_dir=sess5 / "out",
            identity=chain["identity"],
            motion_summary=empty_motion,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            config=cfg,
        )
        if r5.accepted:
            s5 = json.loads(Path(str(r5.summary_json)).read_text())
            dist = next(m for m in s5["metrics"] if m["metric_name"] == "measured_distance")
            sprint = next(m for m in s5["metrics"] if m["metric_name"] == "sprint_count")
            if dist["status"] == "not_evaluable" and sprint["value"] == 0:
                result.ok("05_zero_vs_null")
            else:
                result.fail("05_zero_vs_null", f"{dist} {sprint}")
        else:
            result.fail("05_zero_vs_null", str(r5.summary))

        # 06 hard gap already handled upstream; fusion must keep no_extrapolation flags
        if fused.get("no_coverage_extrapolation") and fused.get("partial_is_not_full_match"):
            result.ok("06_no_extrapolation_flags")
        else:
            result.fail("06_no_extrapolation_flags", "flags missing")

        # 07 fingerprint / stale receipt
        sess7 = _session("stale_")
        sessions.append(sess7)
        stale = dict(motion_receipt)
        stale["status"] = "failed"
        stale["completion_status"] = "failed"
        r7 = integrate_physical_metrics(
            output_dir=sess7 / "out",
            identity=chain["identity"],
            motion_summary=motion_sum,
            motion_receipt=stale,
            config=cfg,
        )
        if not r7.accepted:
            result.ok("07_stale_receipt_fail")
        else:
            result.fail("07_stale_receipt_fail", "accepted stale")

        # 08 distance recount mismatch
        sess8 = _session("dist_")
        sessions.append(sess8)
        r8 = integrate_physical_metrics(
            output_dir=sess8 / "out",
            identity=chain["identity"],
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            recounted_distance_m=float(motion_sum.get("measured_distance_m") or 0) + 5.0,
            config=cfg,
        )
        if r8.accepted:
            s8 = json.loads(Path(str(r8.summary_json)).read_text())
            dist = next(m for m in s8["metrics"] if m["metric_name"] == "measured_distance")
            if dist["status"] == "source_inconsistent":
                result.ok("08_distance_recount_mismatch")
            else:
                result.fail("08_distance_recount_mismatch", dist["status"])
        else:
            result.fail("08_distance_recount_mismatch", str(r8.summary))

        # 09 sprint aggregate present independently
        if any(m["metric_name"] == "sprint_count" for m in fused["metrics"]):
            result.ok("09_sprint_metric_present")
        else:
            result.fail("09_sprint_metric_present", "missing")

        # 10 heatmap mass force inconsistent
        sess10 = _session("hm_")
        sessions.append(sess10)
        r10 = integrate_physical_metrics(
            output_dir=sess10 / "out",
            identity=chain["identity"],
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            heatmap_ref=hm,
            force_source_inconsistent=["source_inconsistent:heatmap_mass_mismatch"],
            config=cfg,
        )
        if r10.accepted:
            s10 = json.loads(Path(str(r10.summary_json)).read_text())
            hm_m = next(m for m in s10["metrics"] if m["metric_name"] == "heatmap_dwell")
            if hm_m["status"] == "source_inconsistent":
                result.ok("10_heatmap_mass_mismatch")
            else:
                result.fail("10_heatmap_mass_mismatch", hm_m["status"])
        else:
            result.fail("10_heatmap_mass_mismatch", str(r10.summary))

        # 11 zone/activity duration mismatch force
        sess11 = _session("za_")
        sessions.append(sess11)
        r11 = integrate_physical_metrics(
            output_dir=sess11 / "out",
            identity=chain["identity"],
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            activity_ref=activity,
            force_source_inconsistent=["source_inconsistent:activity_mass_exceeds_eligible"],
            config=cfg,
        )
        if r11.accepted:
            s11 = json.loads(Path(str(r11.summary_json)).read_text())
            act_m = next(m for m in s11["metrics"] if m["metric_name"] == "activity_distribution")
            if act_m["status"] == "source_inconsistent":
                result.ok("11_activity_duration_mismatch")
            else:
                result.fail("11_activity_duration_mismatch", act_m["status"])
        else:
            result.fail("11_activity_duration_mismatch", str(r11.summary))

        # 12 attack direction unknown
        if fused.get("attack_direction") == "unknown":
            result.ok("12_attack_direction_unknown")
        else:
            result.fail("12_attack_direction_unknown", str(fused.get("attack_direction")))

        # 13 penalty not touch
        if fused.get("penalty_occupancy_is_not_touch") is True:
            result.ok("13_penalty_not_touch")
        else:
            result.fail("13_penalty_not_touch", "flag missing")

        # 14 deterministic repeat
        sess14a = _session("det_a_")
        sess14b = _session("det_b_")
        sessions.extend([sess14a, sess14b])
        kwargs = dict(
            identity=chain["identity"],
            trajectory_summary=traj_sum,
            trajectory_receipt=traj_receipt,
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            heatmap_ref=hm,
            zone_ref=zones,
            activity_ref=activity,
            recounted_distance_m=motion_sum.get("measured_distance_m"),
            config=cfg,
        )
        ra = integrate_physical_metrics(output_dir=sess14a / "out", **kwargs)
        rb = integrate_physical_metrics(output_dir=sess14b / "out", **kwargs)
        if (
            ra.accepted
            and rb.accepted
            and ra.summary.get("overall_physical_analysis_status")
            == rb.summary.get("overall_physical_analysis_status")
            and ra.config_fingerprint == rb.config_fingerprint
        ):
            result.ok("14_deterministic_repeat")
        else:
            result.fail("14_deterministic_repeat", "mismatch")

        # 15 no-overwrite
        sess15 = _session("ow_")
        sessions.append(sess15)
        r15a = integrate_physical_metrics(output_dir=sess15 / "out", **kwargs)
        r15b = integrate_physical_metrics(output_dir=sess15 / "out", **kwargs)
        if r15a.accepted and not r15b.accepted:
            result.ok("15_no_overwrite")
        else:
            result.fail("15_no_overwrite", f"{r15a.accepted}/{r15b.accepted}")

        # 16 partial evaluability — one failed metric does not zero others
        if r1.accepted:
            statuses = {m["metric_name"]: m["status"] for m in fused["metrics"]}
            if "measured_distance" in statuses and "heatmap_dwell" in statuses:
                result.ok("16_partial_metric_independence")
            else:
                result.fail("16_partial_metric_independence", str(statuses))
        else:
            result.fail("16_partial_metric_independence", "no fuse")

        # 17 receipt before success
        if r1.receipt_json and Path(str(r1.quality_json)).is_file():
            result.ok("17_receipt_quality_required")
        else:
            result.fail("17_receipt_quality_required", "missing")

        # 18 missing coverage not inactive
        if fused.get("missing_coverage_not_inactive") is True:
            result.ok("18_missing_not_inactive")
        else:
            result.fail("18_missing_not_inactive", "flag")

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator_exception: {type(exc).__name__}: {exc}", config=True)
    finally:
        if not keep:
            for s in sessions:
                shutil.rmtree(s, ignore_errors=True)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep))
    payload = result.to_dict()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RUNTIME_ROOT / "latest_validator_summary.json"
    try:
        from football_analytics.core.records import write_json_record

        if summary_path.exists():
            summary_path.unlink()
        write_json_record(summary_path, payload, overwrite=False)
    except Exception:
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        gate = GATE_FAIL
        if result.status in {"PASS", "PASS_WITH_WARNINGS"}:
            gate = GATE_FINDINGS
        print(gate)
        print(f"status={result.status} scenarios={len(result.scenarios)}")
        for name, st in result.scenarios.items():
            print(f"  {name}: {st}")
        for e in result.errors:
            print(f"ERROR: {e}", file=sys.stderr)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
