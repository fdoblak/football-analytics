#!/usr/bin/env python3
"""Validate Stage 6D tracking fusion + quality pipeline (synthetic E2E).

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

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/tracking_pipeline_checks")
EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"
EXPECTED_OBS_FP = "9ca2f7af56e69b47ec8db8d644164c84aa7fe3a62da40e247ed6db4f2c4c5f01"
EXPECTED_SUM_FP = "7b04e31d641c49e66ad06baec53e1075e2bc286b9f08f1497aa0571bf7c1c168"
GATE = (
    "PASS_WITH_FINDINGS — TRACKING PIPELINE ACTIVE; STAGE 6 CLOSED; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
NOT_EVAL = "NOT_EVALUATED_NO_REVIEWED_TRACKING_GROUND_TRUTH"


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


def _tbl(contract_name: str, rows: list[dict[str, Any]]) -> Any:
    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    schema = compile_arrow_schema(get_contract(contract_name, 1))
    return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.hashing import sha256_file
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import write_contract_parquet
    from football_analytics.tracking.contracts import (
        load_tracking_json_schema,
        tracking_schema_fingerprints,
        validate_against_json_schema,
    )
    from football_analytics.tracking.evaluation import NOT_EVALUATED_TRACKING
    from football_analytics.tracking.tracking_pipeline import run_tracking_integrate
    from football_analytics.tracking.tracking_pipeline_config import (
        load_tracking_pipeline_config,
        tracking_pipeline_config_fingerprint,
    )
    from football_analytics.tracking.tracking_pipeline_fixtures import (
        DETECTION_FP_A,
        SOURCE_SHA_A,
        TIMELINE_FP_A,
        WINDOW_FP_A,
        assert_runtime_root,
        build_minimal_tracking_fusion_inputs,
        write_json,
    )

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_tracking_pipeline_config(cfg_path)
        result.extras["config_fingerprint"] = tracking_pipeline_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    fps = tracking_schema_fingerprints()
    result.extras["detections_fingerprint"] = fps["detections"]
    result.extras["track_observations_fingerprint"] = fps["track_observations"]
    result.extras["track_summaries_fingerprint"] = fps["track_summaries"]
    result.extras["track_lifecycle_fingerprint"] = fps["track_lifecycle"]
    if fps["detections"] != EXPECTED_DETECTIONS_FP:
        result.err("detections v1 fingerprint changed", integrity=True)
    if fps["track_observations"] != EXPECTED_OBS_FP:
        result.err("track_observations v1 fingerprint changed", integrity=True)
    if fps["track_summaries"] != EXPECTED_SUM_FP:
        result.err("track_summaries v1 fingerprint changed", integrity=True)
    if not str(fps["track_lifecycle"]).startswith("613cd81e"):
        result.err("track_lifecycle v1 fingerprint changed", integrity=True)

    session = Path(tempfile.mkdtemp(prefix="trk_pipe_val_", dir=str(RUNTIME_ROOT)))
    try:
        rid = generate_run_id()
        inputs = build_minimal_tracking_fusion_inputs(
            rid, n_frames=min(int(args.frames), 20), collide_track_ids=True
        )
        for key, contract in (
            ("detections", "detections"),
            ("detection_attributes", "detection_attributes"),
            ("frames", "frames"),
            ("analysis_windows", "analysis_windows"),
            ("human_observations", "track_observations"),
            ("human_summaries", "track_summaries"),
            ("human_lifecycle", "track_lifecycle"),
            ("ball_observations", "track_observations"),
            ("ball_summaries", "track_summaries"),
            ("ball_lifecycle", "track_lifecycle"),
        ):
            rows = list(inputs[key])
            write_contract_parquet(
                _tbl(contract, rows),
                session / f"{key}.parquet",
                get_contract(contract, 1),
                contain_root=RUNTIME_ROOT,
            )
        write_json(session / "detection_receipt.json", inputs["detection_receipt"])
        write_json(session / "human_receipt.json", inputs["human_receipt"])
        write_json(session / "ball_receipt.json", inputs["ball_receipt"])
        write_json(
            session / "ball_primary_candidates.json",
            {
                "schema_version": 1,
                "run_id": rid,
                "video_id": "video_01",
                "frames": inputs["primary_sidecar"],
            },
        )

        # Stamp detection fingerprint to actual detections hash for alignment.
        det_fp = sha256_file(session / "detections.parquet")
        aw_fp = sha256_file(session / "analysis_windows.parquet")
        tl_fp = sha256_file(session / "frames.parquet")
        for name in ("detection_receipt", "human_receipt", "ball_receipt"):
            rec = json.loads((session / f"{name}.json").read_text(encoding="utf-8"))
            rec["source_video_sha256"] = SOURCE_SHA_A
            rec["timeline_fingerprint"] = tl_fp
            rec["detection_bundle_fingerprint"] = det_fp
            rec["analysis_window_fingerprint"] = aw_fp
            if "artifacts" in rec and isinstance(rec["artifacts"], dict):
                rec["artifacts"]["source_video_sha256"] = SOURCE_SHA_A
                rec["artifacts"]["timeline_fingerprint"] = tl_fp
                rec["artifacts"]["detection_bundle_fingerprint"] = det_fp
                rec["artifacts"]["analysis_window_fingerprint"] = aw_fp
            write_json(session / f"{name}.json", rec)

        out1 = session / "out1"
        out1.mkdir()
        res1 = run_tracking_integrate(
            detections=str(session / "detections.parquet"),
            detection_attributes=str(session / "detection_attributes.parquet"),
            detection_receipt=str(session / "detection_receipt.json"),
            human_observations=str(session / "human_observations.parquet"),
            human_summaries=str(session / "human_summaries.parquet"),
            human_lifecycle=str(session / "human_lifecycle.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_observations=str(session / "ball_observations.parquet"),
            ball_summaries=str(session / "ball_summaries.parquet"),
            ball_lifecycle=str(session / "ball_lifecycle.parquet"),
            ball_receipt=str(session / "ball_receipt.json"),
            output_dir=str(out1),
            config=config,
            contain_root=RUNTIME_ROOT,
            frames=str(session / "frames.parquet"),
            analysis_windows=str(session / "analysis_windows.parquet"),
            ball_primary_sidecar=str(session / "ball_primary_candidates.json"),
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=tl_fp,
            expected_detection_fp=det_fp,
            expected_analysis_window_fp=aw_fp,
        )
        if not res1.accepted:
            result.err(f"integrate failed: {res1.error_code}")
            return result.finalize()

        receipt = json.loads(Path(str(res1.pipeline_receipt_json)).read_text(encoding="utf-8"))
        quality = json.loads(Path(str(res1.quality_report_json)).read_text(encoding="utf-8"))
        try:
            validate_against_json_schema(
                receipt, load_tracking_json_schema("tracking_pipeline_receipt")
            )
            validate_against_json_schema(
                quality, load_tracking_json_schema("tracking_quality_report")
            )
            if res1.bundle_manifest_json:
                manifest = json.loads(
                    Path(str(res1.bundle_manifest_json)).read_text(encoding="utf-8")
                )
                validate_against_json_schema(
                    manifest, load_tracking_json_schema("tracking_bundle_manifest")
                )
            result.extras["schema_validation"] = "ok"
        except Exception as exc:  # noqa: BLE001
            result.err(f"schema validation failed: {exc}", integrity=True)

        if receipt.get("ground_truth_evaluation_status") != NOT_EVALUATED_TRACKING:
            result.err("expected NOT_EVALUATED tracking GT status")
        if quality.get("status") not in {"pass", "pass_with_findings"}:
            result.err(f"unexpected quality status: {quality.get('status')}")
        if receipt.get("provenance", {}).get("track_id_is_player_identity") is not False:
            result.err("track_id must not be claimed as player identity")

        # Deterministic repeat
        out2 = session / "out2"
        out2.mkdir()
        res2 = run_tracking_integrate(
            detections=str(session / "detections.parquet"),
            detection_attributes=str(session / "detection_attributes.parquet"),
            detection_receipt=str(session / "detection_receipt.json"),
            human_observations=str(session / "human_observations.parquet"),
            human_summaries=str(session / "human_summaries.parquet"),
            human_lifecycle=str(session / "human_lifecycle.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_observations=str(session / "ball_observations.parquet"),
            ball_summaries=str(session / "ball_summaries.parquet"),
            ball_lifecycle=str(session / "ball_lifecycle.parquet"),
            ball_receipt=str(session / "ball_receipt.json"),
            output_dir=str(out2),
            config=config,
            contain_root=RUNTIME_ROOT,
            frames=str(session / "frames.parquet"),
            analysis_windows=str(session / "analysis_windows.parquet"),
            ball_primary_sidecar=str(session / "ball_primary_candidates.json"),
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=tl_fp,
            expected_detection_fp=det_fp,
            expected_analysis_window_fp=aw_fp,
        )
        if not res2.accepted:
            result.err(f"repeat integrate failed: {res2.error_code}")
        elif res1.total_track_count != res2.total_track_count:
            result.err("non-deterministic track counts")
        else:
            result.extras["deterministic"] = True

        # Negative: overwrite forbidden
        res_ow = run_tracking_integrate(
            detections=str(session / "detections.parquet"),
            detection_attributes=str(session / "detection_attributes.parquet"),
            detection_receipt=str(session / "detection_receipt.json"),
            human_observations=str(session / "human_observations.parquet"),
            human_summaries=str(session / "human_summaries.parquet"),
            human_lifecycle=str(session / "human_lifecycle.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_observations=str(session / "ball_observations.parquet"),
            ball_summaries=str(session / "ball_summaries.parquet"),
            ball_lifecycle=str(session / "ball_lifecycle.parquet"),
            ball_receipt=str(session / "ball_receipt.json"),
            output_dir=str(out1),
            config=config,
            contain_root=RUNTIME_ROOT,
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=tl_fp,
            expected_detection_fp=det_fp,
            expected_analysis_window_fp=aw_fp,
        )
        if res_ow.accepted or res_ow.error_code != "OVERWRITE_FORBIDDEN":
            result.err("overwrite control failed")

        # Negative: source SHA mismatch
        bad_rec = json.loads((session / "ball_receipt.json").read_text(encoding="utf-8"))
        bad_rec["source_video_sha256"] = "b" * 64
        bad_rec["artifacts"]["source_video_sha256"] = "b" * 64
        write_json(session / "ball_receipt_bad.json", bad_rec)
        out_neg = session / "out_neg"
        out_neg.mkdir()
        res_neg = run_tracking_integrate(
            detections=str(session / "detections.parquet"),
            detection_attributes=str(session / "detection_attributes.parquet"),
            detection_receipt=str(session / "detection_receipt.json"),
            human_observations=str(session / "human_observations.parquet"),
            human_summaries=str(session / "human_summaries.parquet"),
            human_lifecycle=str(session / "human_lifecycle.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_observations=str(session / "ball_observations.parquet"),
            ball_summaries=str(session / "ball_summaries.parquet"),
            ball_lifecycle=str(session / "ball_lifecycle.parquet"),
            ball_receipt=str(session / "ball_receipt_bad.json"),
            output_dir=str(out_neg),
            config=config,
            contain_root=RUNTIME_ROOT,
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=tl_fp,
            expected_detection_fp=det_fp,
            expected_analysis_window_fp=aw_fp,
        )
        if res_neg.accepted or res_neg.error_code != "SOURCE_SHA_MISMATCH":
            result.err(f"source sha negative failed: {res_neg.error_code}")
        elif any(out_neg.iterdir()):
            # Partial outputs must not remain on failure (except empty dir)
            leftover = [p for p in out_neg.iterdir() if p.is_file()]
            if leftover:
                result.err("partial outputs left after failure")

        result.finding(NOT_EVAL)
        result.finding("real football tracking accuracy not validated")
        result.extras["total_track_count"] = res1.total_track_count
        result.extras["quality_status"] = res1.quality_status
        result.extras["review_count"] = res1.review_count
        result.extras["fixture_fps"] = {
            "SOURCE_SHA_A": SOURCE_SHA_A[:8],
            "TIMELINE_FP_A": TIMELINE_FP_A[:8],
            "DETECTION_FP_A": DETECTION_FP_A[:8],
            "WINDOW_FP_A": WINDOW_FP_A[:8],
        }
        result.extras["cuda_required"] = False
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)
            result.extras["cleanup"] = "removed_session"
        else:
            result.extras["session"] = str(session)
            result.extras["cleanup"] = "kept"

    return result.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/tracking/tracking_pipeline.yaml",
        help="Pipeline config path",
    )
    parser.add_argument("--frames", type=int, default=8, help="Synthetic frame count (≤20)")
    parser.add_argument("--keep", action="store_true", help="Keep session directory")
    parser.add_argument("--json-out", type=str, default=None, help="Write report JSON path")
    args = parser.parse_args()
    if args.frames > 20:
        print("frames must be <= 20", file=sys.stderr)
        return EXIT_CONFIG

    result = run_checks(args)
    payload = result.to_dict()
    out_path = args.json_out
    if out_path is None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = str(RUNTIME_ROOT / f"tracking_pipeline_validation_{stamp}.json")
    Path(out_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status: {payload['status']}")
    print(f"gate: {payload.get('gate')}")
    print(f"report: {out_path}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
