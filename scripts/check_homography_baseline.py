#!/usr/bin/env python3
"""Validate Stage 8C homography solve + calibration segment baseline.

Exit codes:
  0 PASS / PASS_WITH_FINDINGS
  1 validation finding / NO-GO content
  2 configuration failure
  3 integrity/security failure
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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/homography_checks")
GATE = (
    "PASS_WITH_FINDINGS — HOMOGRAPHY AND CALIBRATION SEGMENT BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.findings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finding(self, msg: str) -> None:
        self.findings.append(msg)

    def finalize(self) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "NO-GO" if self.exit_code == EXIT_INTEGRITY else "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.findings or self.warnings:
            self.status = "PASS_WITH_FINDINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
            "overall_status": self.status,
            "gate": GATE if self.status in {"PASS", "PASS_WITH_FINDINGS"} else self.status,
        }
        body.update(self.extras)
        return body


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.calibration.contracts import (
        EXPECTED_CALIBRATIONS_FP,
        assert_calibrations_fingerprint_frozen,
        calibration_schema_fingerprints,
    )
    from football_analytics.calibration.correspondence import build_correspondences_from_features
    from football_analytics.calibration.homography_config import (
        homography_config_fingerprint,
        load_homography_config,
    )
    from football_analytics.calibration.homography_evaluation import (
        NOT_EVALUATED_HOMOGRAPHY,
        evaluate_homography,
    )
    from football_analytics.calibration.homography_fixtures import (
        assert_runtime_root,
        known_perspective_H,
        multi_frame_stable_features,
        synthetic_feature_rows_for_H,
    )
    from football_analytics.calibration.homography_service import run_homography_solve
    from football_analytics.calibration.homography_solve import solve_frame_homography
    from football_analytics.calibration.pitch_template import build_pitch_template
    from football_analytics.core.run_id import generate_run_id

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_homography_config(cfg_path)
        result.extras["config_fingerprint"] = homography_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    if config["attack_direction"] != "unknown":
        result.err("attack_direction must be unknown", integrity=True)
    if config["auto_project_positions"] is not False:
        result.err("auto_project_positions must be false", integrity=True)
    if config["network_sources_allowed"] is not False:
        result.err("network_sources_allowed must be false", integrity=True)
    if config["segments"]["silent_gap_fill"] is not False:
        result.err("silent_gap_fill must be false", integrity=True)
    if config["quality"]["degraded_physical_eligible"] is not False:
        result.err("degraded_physical_eligible must be false", integrity=True)

    try:
        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        result.extras["calibrations_fp"] = fps["calibrations"]
        if fps["calibrations"] != EXPECTED_CALIBRATIONS_FP:
            result.err("calibrations fingerprint drift", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"fingerprint check failed: {exc}", integrity=True)

    # Lazy: importing calibration must not load HRNet
    import sys as _sys

    before = {k for k in _sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
    import football_analytics.calibration as cal  # noqa: F401

    after = {k for k in _sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
    if after - before:
        result.err("import football_analytics.calibration loaded HRNet", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    eval_report = evaluate_homography(has_reviewed_ground_truth=False)
    if eval_report.ground_truth_evaluation_status != NOT_EVALUATED_HOMOGRAPHY:
        result.err("expected NOT_EVALUATED_NO_REVIEWED_HOMOGRAPHY_GROUND_TRUTH")
    result.extras["evaluation_status"] = eval_report.ground_truth_evaluation_status

    # Synthetic known-H solve (no SV inference)
    session = Path(tempfile.mkdtemp(prefix="homo_val_", dir=str(RUNTIME_ROOT)))
    try:
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_homo_val"
        rows = multi_frame_stable_features(H, run_id=rid, video_id=vid, n_frames=3)
        out = session / "solve"
        svc = run_homography_solve(
            output_dir=out,
            config=config,
            contain_root=RUNTIME_ROOT,
            features_rows=rows,
            correspondence_mode="keypoint_only",
        )
        if not svc.accepted:
            result.err(f"solve failed: {svc.error_code}")
        else:
            result.extras["solve_ok"] = True
            result.extras["calibration_count"] = svc.summary.get("calibration_count")
            result.extras["segment_count"] = svc.summary.get("segment_count")
            if Path(str(svc.calibrations_parquet)).is_file() is False:
                result.err("calibrations parquet missing")
            if "projected_positions" in (svc.summary or {}):
                result.err("projected_positions must not be produced", integrity=True)
            # no-overwrite
            again = run_homography_solve(
                output_dir=out,
                config=config,
                contain_root=RUNTIME_ROOT,
                features_rows=rows,
            )
            if again.accepted or again.error_code != "NO_OVERWRITE":
                result.err("no-overwrite gate failed")

        # Direct geometry smoke
        template = build_pitch_template()
        one = synthetic_feature_rows_for_H(H, run_id=rid, video_id=vid, n=4)
        built = build_correspondences_from_features(
            one, template=template, config=config, mode="keypoint_only"
        )
        sol = solve_frame_homography(built.accepted, config=config)
        result.extras["direct_quality"] = sol.quality.value
        if sol.H is None and sol.quality.value not in {"invalid", "not_available"}:
            result.err("unexpected empty H")
    except Exception as exc:  # noqa: BLE001
        result.err(f"synthetic validator failed: {type(exc).__name__}: {exc}")
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)

    # Preserve known Stage 8B GPL finding (adapter unchanged)
    result.finding(
        "NBJW/SV adapter remains evaluation_only / GPL-2.0 linking risk (Stage 8B; not vendored)"
    )
    result.finding("Real football homography accuracy not validated — no reviewed ground truth")
    result.finding("Homography is pitch-plane only; projected positions deferred to Stage 8D")
    result.warn("Synthetic known-H metrics are not match accuracy")

    result.extras["gate"] = GATE
    result.extras["runtime_root"] = str(RUNTIME_ROOT)
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/calibration/homography_baseline.yaml",
        help="Homography baseline config path",
    )
    parser.add_argument("--keep", action="store_true", help="Keep validator session dir")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)
    result = run_checks(args)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload['status']}")
        print(f"gate: {payload.get('gate')}")
        print(f"exit_code: {payload['exit_code']}")
        for e in payload["errors"]:
            print(f"error: {e}")
        for w in payload["warnings"]:
            print(f"warning: {w}")
        for f in payload["findings"]:
            print(f"finding: {f}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
