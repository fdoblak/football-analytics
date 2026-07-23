#!/usr/bin/env python3
"""Validate Stage 5A player/official/ball detection contracts.

Exit codes:
  0  success (PASS / PASS_WITH_WARNINGS)
  1  validation finding/failure
  2  configuration/schema failure
  3  integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/detection_contract_checks")
DETECTIONS_JSON = REPO_ROOT / "schemas" / "data" / "v1" / "detections.json"
DETECTIONS_SHA256 = "957a41ca2ded9580bc18d39bc7902e133b34ec866077ccc944ab334b9e2681fd"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
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

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "extras": self.extras,
        }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id() -> str:
    from football_analytics.core.run_id import generate_run_id

    return generate_run_id()


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    spec = get_contract(name, 1)
    schema = compile_arrow_schema(spec)
    return pa.Table.from_pylist(rows, schema=schema)


def _build_synthetic(run_id: str) -> dict[str, Any]:
    video_id = "clip_demo_01"
    videos = _cast(
        "videos",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "source_sha256": "a" * 64,
                "container": "mp4",
                "codec": "h264",
                "width_px": 1280,
                "height_px": 720,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "time_base_numerator": 1,
                "time_base_denominator": 25,
                "frame_count": 8,
                "duration_us": 320000,
                "has_audio": False,
                "source_ref": "logical_clip_demo_01",
            }
        ],
    )
    frames = _cast(
        "frames",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": i,
                "pts": i,
                "video_time_us": i * 40000,
                "duration_us": 40000,
                "is_key_frame": i % 4 == 0,
                "decode_status": "ok",
            }
            for i in range(8)
        ],
    )
    windows = _cast(
        "analysis_windows",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "analysis_window_id": "aw_play_001",
                "start_time_us": 0,
                "end_time_us": 160000,
                "start_frame_index": 0,
                "end_frame_index_exclusive": 4,
                "shot_id": "shot_001",
                "camera_segment_ids": ["cam_001"],
                "view_family": "main_broadcast",
                "framing_scale": "wide",
                "replay_status": "live",
                "graphics_status": "none",
                "playability": "playable",
                "tracking_eligibility": "eligible",
                "calibration_eligibility": "eligible",
                "identity_eligibility": "conditionally_eligible",
                "ball_analysis_eligibility": "eligible",
                "live_event_eligibility": "unknown",
                "physical_metric_eligibility": "eligible",
                "decision_codes": ["PLAYABLE_WIDE_VIEW"],
                "manual_review_required": False,
                "coverage": 1.0,
                "confidence": 0.95,
                "timeline_mapping_quality": "exact_identity",
                "source_refs": ["shot_001", "cam_001"],
                "policy_version": "1",
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "analysis_window_id": "aw_graphics_001",
                "start_time_us": 160000,
                "end_time_us": 240000,
                "start_frame_index": 4,
                "end_frame_index_exclusive": 6,
                "shot_id": "shot_002",
                "camera_segment_ids": ["cam_002"],
                "view_family": "graphics",
                "framing_scale": "unknown",
                "replay_status": "live",
                "graphics_status": "full_screen",
                "playability": "non_playable",
                "tracking_eligibility": "ineligible",
                "calibration_eligibility": "ineligible",
                "identity_eligibility": "ineligible",
                "ball_analysis_eligibility": "ineligible",
                "live_event_eligibility": "ineligible",
                "physical_metric_eligibility": "ineligible",
                "decision_codes": ["GRAPHICS_NON_PLAYABLE"],
                "manual_review_required": False,
                "coverage": 1.0,
                "confidence": 0.99,
                "timeline_mapping_quality": "exact_identity",
                "source_refs": ["shot_002", "cam_002"],
                "policy_version": "1",
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "analysis_window_id": "aw_id_close_001",
                "start_time_us": 240000,
                "end_time_us": 320000,
                "start_frame_index": 6,
                "end_frame_index_exclusive": 8,
                "shot_id": "shot_003",
                "camera_segment_ids": ["cam_003"],
                "view_family": "player_isolation",
                "framing_scale": "close_up",
                "replay_status": "live",
                "graphics_status": "none",
                "playability": "partially_playable",
                "tracking_eligibility": "ineligible",
                "calibration_eligibility": "ineligible",
                "identity_eligibility": "eligible",
                "ball_analysis_eligibility": "ineligible",
                "live_event_eligibility": "unknown",
                "physical_metric_eligibility": "ineligible",
                "decision_codes": ["IDENTITY_ONLY_CLOSEUP"],
                "manual_review_required": False,
                "coverage": 0.9,
                "confidence": 0.8,
                "timeline_mapping_quality": "exact_identity",
                "source_refs": ["shot_003", "cam_003"],
                "policy_version": "1",
                "provenance_json": None,
                "contract_version": 1,
            },
        ],
    )
    detections = _cast(
        "detections",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "detection_id": 0,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.91,
                "bbox_x1": 10.0,
                "bbox_y1": 20.0,
                "bbox_x2": 40.0,
                "bbox_y2": 80.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "detection_id": 1,
                "class_id": 32,
                "class_name": "sports_ball",
                "confidence": 0.55,
                "bbox_x1": 100.0,
                "bbox_y1": 200.0,
                "bbox_x2": 108.0,
                "bbox_y2": 208.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 6,
                "detection_id": 0,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.88,
                "bbox_x1": 50.0,
                "bbox_y1": 40.0,
                "bbox_x2": 200.0,
                "bbox_y2": 400.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": [],
            },
        ],
    )
    attributes = _cast(
        "detection_attributes",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "detection_id": 0,
                "entity_type": "human",
                "role_label": "unknown",
                "role_source": "unknown",
                "role_score": None,
                "occlusion": 0.0,
                "truncation": 0.0,
                "visibility": 1.0,
                "review_status": "unreviewed",
                "attribute_source_ref": None,
                "provenance_json": '{"mapping":"person"}',
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "detection_id": 1,
                "entity_type": "ball",
                "role_label": "unknown",
                "role_source": "unknown",
                "role_score": None,
                "occlusion": None,
                "truncation": None,
                "visibility": 0.9,
                "review_status": "unreviewed",
                "attribute_source_ref": None,
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 6,
                "detection_id": 0,
                "entity_type": "human",
                "role_label": "player",
                "role_source": "downstream_classifier",
                "role_score": 0.7,
                "occlusion": 0.1,
                "truncation": 0.0,
                "visibility": 0.95,
                "review_status": "needs_review",
                "attribute_source_ref": "role_clf_v0",
                "provenance_json": None,
                "contract_version": 1,
            },
        ],
    )
    frame_status = _cast(
        "detection_frame_status",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "video_time_us": 0,
                "analysis_window_id": "aw_play_001",
                "processing_status": "processed",
                "eligibility": "eligible",
                "detector_id": "det_dummy_v1",
                "input_artifact_ref": None,
                "detection_count": 2,
                "human_count": 1,
                "ball_count": 1,
                "skip_reason": None,
                "error_code": None,
                "coverage": 1.0,
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 1,
                "video_time_us": 40000,
                "analysis_window_id": "aw_play_001",
                "processing_status": "processed_no_detections",
                "eligibility": "eligible",
                "detector_id": "det_dummy_v1",
                "input_artifact_ref": None,
                "detection_count": 0,
                "human_count": 0,
                "ball_count": 0,
                "skip_reason": None,
                "error_code": None,
                "coverage": 1.0,
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 4,
                "video_time_us": 160000,
                "analysis_window_id": "aw_graphics_001",
                "processing_status": "not_eligible",
                "eligibility": "ineligible",
                "detector_id": "det_dummy_v1",
                "input_artifact_ref": None,
                "detection_count": 0,
                "human_count": 0,
                "ball_count": 0,
                "skip_reason": "FRAME_NOT_ELIGIBLE",
                "error_code": "FRAME_NOT_ELIGIBLE",
                "coverage": 1.0,
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 5,
                "video_time_us": 200000,
                "analysis_window_id": "aw_graphics_001",
                "processing_status": "failed",
                "eligibility": "ineligible",
                "detector_id": "det_dummy_v1",
                "input_artifact_ref": None,
                "detection_count": 0,
                "human_count": 0,
                "ball_count": 0,
                "skip_reason": None,
                "error_code": "FRAME_DECODE_FAILED",
                "coverage": 0.0,
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 6,
                "video_time_us": 240000,
                "analysis_window_id": "aw_id_close_001",
                "processing_status": "processed",
                "eligibility": "conditionally_eligible",
                "detector_id": "det_dummy_v1",
                "input_artifact_ref": None,
                "detection_count": 1,
                "human_count": 1,
                "ball_count": 0,
                "skip_reason": "BALL_ANALYSIS_NOT_ELIGIBLE",
                "error_code": None,
                "coverage": 0.9,
                "provenance_json": '{"identity_only":true}',
                "contract_version": 1,
            },
        ],
    )
    return {
        "videos": videos,
        "frames": frames,
        "analysis_windows": windows,
        "detections": detections,
        "detection_attributes": attributes,
        "detection_frame_status": frame_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    args = parser.parse_args(argv)
    result = Result()

    runtime_root: Path = args.runtime_root
    work: Path | None = None
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work = runtime_root / f"work_{stamp}"
        work.mkdir(parents=True, exist_ok=False)

        # detections v1 SHA unchanged
        got_sha = _sha256(DETECTIONS_JSON)
        result.extras["detections_v1_sha256"] = got_sha
        if got_sha != DETECTIONS_SHA256:
            result.err("detections.json SHA changed", integrity=True)

        from football_analytics.data.compiler import list_contracts
        from football_analytics.perception import (
            DetectionAttributes,
            DetectionFrameStatus,
            DetectionRunReceipt,
            assert_detection_contracts_registered,
            build_preprocessing_transform,
            compile_detection_schemas,
            detection_schema_fingerprints,
            inverse_bbox,
            load_detection_policy,
            load_detection_taxonomy,
            load_perception_json_schema,
            map_model_class,
            policy_fingerprint,
            resolve_frame_routing,
            roundtrip_bbox,
            taxonomy_fingerprint,
            validate_against_json_schema,
            validate_bbox_xyxy,
            validate_detection_bundle,
        )
        from football_analytics.perception.transforms import forward_bbox

        assert_detection_contracts_registered()
        names = list_contracts()
        result.extras["contract_count"] = len(names)
        if len(names) != 20:
            result.err(f"expected 20 contracts, got {len(names)}", config=True)

        schemas = compile_detection_schemas()
        fps = detection_schema_fingerprints()
        result.extras["schema_fingerprints"] = fps
        if set(schemas) != {"detections", "detection_frame_status", "detection_attributes"}:
            result.err("compile_detection_schemas set mismatch", config=True)

        tax = load_detection_taxonomy()
        pol = load_detection_policy()
        tax_fp = taxonomy_fingerprint(tax)
        pol_fp = policy_fingerprint(pol)
        result.extras["taxonomy_fingerprint"] = tax_fp
        result.extras["policy_fingerprint"] = pol_fp
        if taxonomy_fingerprint(tax) != tax_fp or policy_fingerprint(pol) != pol_fp:
            result.err("fingerprint unstable", integrity=True)

        person = map_model_class(0, "person", taxonomy=tax)
        if person.role_label.value != "unknown" or person.entity_type.value != "human":
            result.err("person must map to human+role unknown")
        ball = map_model_class(32, "sports_ball", taxonomy=tax)
        if ball.entity_type.value != "ball" or ball.role_label.value != "unknown":
            result.err("ball mapping failed")
        unmapped = map_model_class(99, "spaceship", taxonomy=tax)
        if not unmapped.rejected:
            result.err("unmapped class should reject under default policy")

        transform = build_preprocessing_transform(
            source_width=1280,
            source_height=720,
            model_input_width=640,
            model_input_height=640,
            resize_mode="letterbox",
        )
        bbox = (10.0, 20.0, 40.0, 80.0)
        validate_bbox_xyxy(bbox, frame_width=1280, frame_height=720)
        back = roundtrip_bbox(bbox, transform)
        result.extras["bbox_roundtrip"] = list(back)
        fwd = forward_bbox(bbox, transform)
        inv = inverse_bbox(fwd, transform)
        if any(
            abs(a - b) > transform.roundtrip_tolerance_px for a, b in zip(bbox, inv, strict=True)
        ):
            result.err("inverse bbox roundtrip failed")

        try:
            validate_bbox_xyxy((0.0, 0.0, 0.0, 10.0))
            result.err("zero-area bbox should fail")
        except Exception:
            pass
        try:
            validate_bbox_xyxy((float("nan"), 0.0, 1.0, 1.0))
            result.err("NaN bbox should fail")
        except Exception:
            pass

        run_id = _run_id()
        bundle = _build_synthetic(run_id)
        from football_analytics.data.compiler import get_contract

        specs = {
            n: get_contract(n, 1)
            for n in (
                "videos",
                "frames",
                "detections",
                "detection_frame_status",
                "detection_attributes",
                "analysis_windows",
            )
        }
        win0 = bundle["analysis_windows"].to_pylist()[0]
        route = resolve_frame_routing(win0, policy=pol)
        if not route["process_human"] or not route["process_ball"]:
            result.err("playable window should allow human+ball")
        win_g = bundle["analysis_windows"].to_pylist()[1]
        route_g = resolve_frame_routing(win_g, policy=pol)
        if route_g["process_human"] or route_g["process_ball"]:
            result.err("graphics window should skip detection")

        # Typed models
        DetectionFrameStatus.from_dict(bundle["detection_frame_status"].to_pylist()[0])
        DetectionAttributes.from_dict(bundle["detection_attributes"].to_pylist()[0])

        receipt = DetectionRunReceipt.from_dict(
            {
                "schema_version": 1,
                "receipt_id": "det_receipt_01",
                "run_id": run_id,
                "detector_id": "det_dummy_v1",
                "model_registry_id": None,
                "model_sha256": None,
                "adapter_id": "synthetic_adapter",
                "adapter_version": "0.0.0",
                "config_fingerprint": pol_fp,
                "taxonomy_version": str(tax["taxonomy_version"]),
                "source_video_ref": "logical_clip_demo_01",
                "frames_ref": "frames.parquet",
                "analysis_windows_ref": "analysis_windows.parquet",
                "eligible_frame_count": 3,
                "processed_frame_count": 3,
                "skipped_frame_count": 0,
                "failed_frame_count": 1,
                "processed_no_detection_count": 1,
                "total_detection_count": 3,
                "human_detection_count": 2,
                "ball_detection_count": 1,
                "pre_nms_count": 5,
                "post_nms_count": 3,
                "started_at_utc": "2026-07-23T00:00:00.000000Z",
                "completed_at_utc": "2026-07-23T00:00:01.000000Z",
                "status": "succeeded",
                "warnings": [],
                "errors": [],
                "artifacts": {"detections": "detections.parquet"},
                "environment_ref": None,
                "transform_fingerprint": transform.transform_fingerprint,
                "threshold_config_fingerprint": pol_fp,
                "provenance": {"stage": "5A", "label": "synthetic_check", "notes": None},
            }
        )
        validate_against_json_schema(
            receipt.to_dict(), load_perception_json_schema("detection_run_receipt")
        )
        validate_against_json_schema(
            transform.to_dict(), load_perception_json_schema("preprocessing_transform")
        )

        vr = validate_detection_bundle(
            detections=bundle["detections"],
            frame_status=bundle["detection_frame_status"],
            attributes=bundle["detection_attributes"],
            frames=bundle["frames"],
            videos=bundle["videos"],
            analysis_windows=bundle["analysis_windows"],
            specs=specs,
            receipt=receipt.to_dict(),
        )
        result.extras["bundle_status"] = vr.status
        if vr.status == "FAIL":
            for e in vr.errors:
                result.err(f"bundle: {e}")

        # Parquet roundtrip
        from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet

        pq_dir = work / "parquet"
        pq_dir.mkdir()
        for name in ("detections", "detection_frame_status", "detection_attributes"):
            path = pq_dir / f"{name}.parquet"
            write_contract_parquet(bundle[name], path, specs[name])
            table2 = read_contract_parquet(path, specs[name])
            if table2.num_rows != bundle[name].num_rows:
                result.err(f"parquet roundtrip row mismatch: {name}")

        result.extras["work_dir"] = str(work)
    except Exception as exc:  # noqa: BLE001
        result.err(f"unexpected: {exc}", config=True)
    finally:
        if work is not None and work.exists():
            shutil.rmtree(work, ignore_errors=True)
            result.extras["cleanup"] = "removed_work_dir"

    result.finalize(strict=bool(args.strict))
    report_name = (
        f"detection_contract_validation_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path = runtime_root / report_name
    report_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.status, "report": str(report_path)}, indent=2))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
