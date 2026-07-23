#!/usr/bin/env python3
"""Validate Stage 6C ball tracking baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/ball_tracking_checks")
GATE_PASS = "PASS — BALL TRACKING BASELINE ACTIVE"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — BALL TRACKING BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — BALL TRACKING BASELINE FAILURE"


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
        gate = GATE_FAIL
        if self.status in {"PASS", "PASS_WITH_FINDINGS"}:
            gate = GATE_FINDINGS if self.findings or self.warnings else GATE_PASS
        body = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
            "overall_status": self.status,
            "gate": gate,
        }
        body.update(self.extras)
        return body


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.hashing import hash_canonical_json
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.fingerprint import contract_fingerprint
    from football_analytics.tracking.ball_tracker import run_ball_tracker
    from football_analytics.tracking.ball_tracking_config import (
        ball_tracking_config_fingerprint,
        load_ball_tracking_config,
    )
    from football_analytics.tracking.ball_tracking_evaluation import NOT_EVALUATED_BALL_TRACKING
    from football_analytics.tracking.ball_tracking_fixtures import (
        assert_runtime_root,
        frozen_ambiguity,
        frozen_constant_velocity,
        frozen_fast_zero_iou,
        frozen_long_gap,
        frozen_reject_human,
        frozen_short_gap,
        frozen_shot_cut,
    )
    from football_analytics.tracking.ball_tracking_service import run_ball_tracking
    from football_analytics.tracking.contracts import (
        EXPECTED_DETECTIONS_FP,
        EXPECTED_TRACK_OBSERVATIONS_FP,
        EXPECTED_TRACK_SUMMARIES_FP,
        assert_v1_track_fingerprints_unchanged,
    )
    from football_analytics.tracking.policy import load_tracking_policy

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_ball_tracking_config(cfg_path)
        result.extras["config_fingerprint"] = ball_tracking_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    try:
        assert_v1_track_fingerprints_unchanged()
        fps = {
            "track_observations": contract_fingerprint(get_contract("track_observations", 1)),
            "track_summaries": contract_fingerprint(get_contract("track_summaries", 1)),
            "detections": contract_fingerprint(get_contract("detections", 1)),
            "track_lifecycle": contract_fingerprint(get_contract("track_lifecycle", 1)),
        }
        result.extras["schema_fingerprints"] = fps
        if fps["track_observations"] != EXPECTED_TRACK_OBSERVATIONS_FP:
            result.err("track_observations fingerprint drift", integrity=True)
        if fps["track_summaries"] != EXPECTED_TRACK_SUMMARIES_FP:
            result.err("track_summaries fingerprint drift", integrity=True)
        if fps["detections"] != EXPECTED_DETECTIONS_FP:
            result.err("detections fingerprint drift", integrity=True)
        if not fps["track_lifecycle"].startswith("613cd81e"):
            result.err("track_lifecycle fingerprint drift", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"fingerprint check failed: {exc}", integrity=True)

    policy = load_tracking_policy()
    session = Path(tempfile.mkdtemp(prefix="bt6c_", dir=RUNTIME_ROOT))
    result.extras["session_dir"] = str(session)

    try:
        b = frozen_constant_velocity()
        r1 = run_ball_tracker(
            run_id=b["run_id"],
            video_id=b["video_id"],
            frames=b["frames"].to_pylist(),
            detections=b["detections"].to_pylist(),
            analysis_windows=b["analysis_windows"].to_pylist(),
            detection_attributes=b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        r2 = run_ball_tracker(
            run_id=b["run_id"],
            video_id=b["video_id"],
            frames=b["frames"].to_pylist(),
            detections=b["detections"].to_pylist(),
            analysis_windows=b["analysis_windows"].to_pylist(),
            detection_attributes=b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        fp1 = hash_canonical_json(
            {"o": r1.observations, "l": r1.lifecycle, "p": r1.primary_sidecar}
        )
        fp2 = hash_canonical_json(
            {"o": r2.observations, "l": r2.lifecycle, "p": r2.primary_sidecar}
        )
        if fp1 != fp2:
            result.err("deterministic repeat mismatch", integrity=True)
        result.extras["deterministic_output_fingerprint"] = fp1

        fast_b = frozen_fast_zero_iou()
        fast = run_ball_tracker(
            run_id=fast_b["run_id"],
            video_id=fast_b["video_id"],
            frames=fast_b["frames"].to_pylist(),
            detections=fast_b["detections"].to_pylist(),
            analysis_windows=fast_b["analysis_windows"].to_pylist(),
            detection_attributes=fast_b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        if (
            len({o["track_id"] for o in fast.observations if o["observation_state"] == "observed"})
            != 1
        ):
            result.err("fast zero-IoU association failed to keep single track")

        short_b = frozen_short_gap()
        short = run_ball_tracker(
            run_id=short_b["run_id"],
            video_id=short_b["video_id"],
            frames=short_b["frames"].to_pylist(),
            detections=short_b["detections"].to_pylist(),
            analysis_windows=short_b["analysis_windows"].to_pylist(),
            detection_attributes=short_b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        if not any(o["observation_state"] == "predicted" for o in short.observations):
            result.err("short gap prediction missing")

        long_b = frozen_long_gap()
        long_r = run_ball_tracker(
            run_id=long_b["run_id"],
            video_id=long_b["video_id"],
            frames=long_b["frames"].to_pylist(),
            detections=long_b["detections"].to_pylist(),
            analysis_windows=long_b["analysis_windows"].to_pylist(),
            detection_attributes=long_b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        if len({o["track_id"] for o in long_r.observations}) < 2:
            result.err("long occlusion did not create new track")

        shot_b = frozen_shot_cut()
        shot_r = run_ball_tracker(
            run_id=shot_b["run_id"],
            video_id=shot_b["video_id"],
            frames=shot_b["frames"].to_pylist(),
            detections=shot_b["detections"].to_pylist(),
            analysis_windows=shot_b["analysis_windows"].to_pylist(),
            detection_attributes=shot_b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        if not any("SHOT_CUT" in f for f in shot_r.findings):
            result.err("shot cut not detected")

        amb_b = frozen_ambiguity()
        amb_r = run_ball_tracker(
            run_id=amb_b["run_id"],
            video_id=amb_b["video_id"],
            frames=amb_b["frames"].to_pylist(),
            detections=amb_b["detections"].to_pylist(),
            analysis_windows=amb_b["analysis_windows"].to_pylist(),
            detection_attributes=amb_b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        if amb_r.stats["ambiguous_frames"] < 1:
            result.err("ambiguity scenario did not mark ambiguous frames")

        hum_b = frozen_reject_human()
        hum_r = run_ball_tracker(
            run_id=hum_b["run_id"],
            video_id=hum_b["video_id"],
            frames=hum_b["frames"].to_pylist(),
            detections=hum_b["detections"].to_pylist(),
            analysis_windows=hum_b["analysis_windows"].to_pylist(),
            detection_attributes=hum_b["detection_attributes"].to_pylist(),
            config=config,
            policy=policy,
        )
        if hum_r.stats["rejected_non_ball"] < 1:
            result.err("human detection not rejected")

        out = session / "e2e"
        out.mkdir(parents=True, exist_ok=True)
        svc = run_ball_tracking(
            detections="mem",
            frames="mem",
            analysis_windows="mem",
            output_dir=out,
            config=config,
            in_memory_bundle=b,
            contain_root=RUNTIME_ROOT,
            run_id=b["run_id"],
            video_id=b["video_id"],
        )
        if not svc.accepted:
            result.err(f"service e2e failed: {svc.error_code}")
        else:
            eval_payload = json.loads(Path(svc.evaluation_json).read_text(encoding="utf-8"))
            if eval_payload.get("ground_truth_evaluation_status") != NOT_EVALUATED_BALL_TRACKING:
                result.err("expected ball-tracking not-evaluated status code")
            result.finding(NOT_EVALUATED_BALL_TRACKING)
            result.finding("REAL FOOTBALL ACCURACY NOT YET VALIDATED — synthetic fixtures only")
            result.finding("primary_ball_candidate is not a guarantee of true ball identity")

        svc2 = run_ball_tracking(
            detections="mem",
            frames="mem",
            analysis_windows="mem",
            output_dir=out,
            config=config,
            in_memory_bundle=b,
            contain_root=RUNTIME_ROOT,
            run_id=b["run_id"],
            video_id=b["video_id"],
        )
        if svc2.accepted or svc2.error_code != "OVERWRITE_FORBIDDEN":
            result.err("overwrite negative failed")

        result.extras["tracker_algorithm"] = config["tracker_algorithm"]
        result.extras["association_method"] = config["association_method"]
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)
            result.extras.pop("session_dir", None)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/tracking/ball_tracking_baseline.yaml",
        help="Ball tracking baseline config path",
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
        print(f"gate: {payload['gate']}")
        for e in payload["errors"]:
            print(f"error: {e}")
        for w in payload["warnings"]:
            print(f"warning: {w}")
        for f in payload["findings"]:
            print(f"finding: {f}")
        if "config_fingerprint" in payload:
            print(f"config_fingerprint: {payload['config_fingerprint']}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
