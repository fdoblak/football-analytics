#!/usr/bin/env python3
"""Validate Stage 4A broadcast shot/camera contracts.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/broadcast_contract_checks")


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


def _run_id() -> str:
    from football_analytics.core.run_id import generate_run_id

    return generate_run_id()


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    spec = get_contract(name, 1)
    schema = compile_arrow_schema(spec)
    return pa.Table.from_pylist(rows, schema=schema)


def _videos(run_id: str, video_id: str = "clip_demo_01") -> Any:
    return _cast(
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


def _frames(run_id: str, video_id: str = "clip_demo_01") -> Any:
    rows = []
    for i in range(8):
        rows.append(
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
        )
    return _cast("frames", rows)


def _valid_bundle(run_id: str) -> dict[str, Any]:
    video_id = "clip_demo_01"
    boundaries = _cast(
        "shot_boundaries",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "boundary_id": "bnd_001",
                "boundary_time_us": 120000,
                "left_frame_index": 2,
                "right_frame_index": 3,
                "transition_type": "hard_cut",
                "transition_duration_us": 0,
                "confidence": 0.95,
                "detection_source": "manual",
                "evidence_ref": None,
                "review_status": "accepted",
                "provenance_json": '{"label":"fixture"}',
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "boundary_id": "bnd_002",
                "boundary_time_us": 240000,
                "left_frame_index": 5,
                "right_frame_index": 6,
                "transition_type": "dissolve",
                "transition_duration_us": 40000,
                "confidence": 0.7,
                "detection_source": "rule",
                "evidence_ref": None,
                "review_status": "unreviewed",
                "provenance_json": None,
                "contract_version": 1,
            },
        ],
    )
    shots = _cast(
        "shot_segments",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "shot_id": "shot_001",
                "start_time_us": 0,
                "end_time_us": 120000,
                "start_frame_index": 0,
                "end_frame_index_exclusive": 3,
                "start_boundary_id": None,
                "end_boundary_id": "bnd_001",
                "duration_us": 120000,
                "frame_count": 3,
                "timeline_mapping_quality": "exact_identity",
                "segment_status": "active",
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "shot_id": "shot_002",
                "start_time_us": 120000,
                "end_time_us": 240000,
                "start_frame_index": 3,
                "end_frame_index_exclusive": 6,
                "start_boundary_id": "bnd_001",
                "end_boundary_id": "bnd_002",
                "duration_us": 120000,
                "frame_count": 3,
                "timeline_mapping_quality": "timestamp_preserved",
                "segment_status": "active",
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "shot_id": "shot_003",
                "start_time_us": 240000,
                "end_time_us": 320000,
                "start_frame_index": 6,
                "end_frame_index_exclusive": 8,
                "start_boundary_id": "bnd_002",
                "end_boundary_id": None,
                "duration_us": 80000,
                "frame_count": 2,
                "timeline_mapping_quality": "derived_with_resampling",
                "segment_status": "active",
                "provenance_json": None,
                "contract_version": 1,
            },
        ],
    )
    cameras = _cast(
        "camera_view_segments",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "camera_segment_id": "cam_001",
                "shot_id": "shot_001",
                "start_time_us": 0,
                "end_time_us": 120000,
                "start_frame_index": 0,
                "end_frame_index_exclusive": 3,
                "view_family": "main_broadcast",
                "framing_scale": "wide",
                "camera_position": "sideline",
                "camera_motion": "pan",
                "replay_status": "live",
                "graphics_status": "partial_overlay",
                "playability": "playable",
                "calibration_suitability": "suitable",
                "tracking_suitability": "suitable",
                "target_identity_suitability": "conditionally_suitable",
                "classification_source": "manual",
                "confidence": 0.9,
                "coverage": 1.0,
                "review_status": "accepted",
                "evidence_refs": [],
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "camera_segment_id": "cam_002",
                "shot_id": "shot_002",
                "start_time_us": 120000,
                "end_time_us": 200000,
                "start_frame_index": 3,
                "end_frame_index_exclusive": 5,
                "view_family": "player_isolation",
                "framing_scale": "close_up",
                "camera_position": "field_level",
                "camera_motion": "static",
                "replay_status": "live",
                "graphics_status": "none",
                "playability": "partially_playable",
                "calibration_suitability": "unsuitable",
                "tracking_suitability": "conditionally_suitable",
                "target_identity_suitability": "suitable",
                "classification_source": "rule",
                "confidence": 0.8,
                "coverage": 0.5,
                "review_status": "unreviewed",
                "evidence_refs": [],
                "provenance_json": None,
                "contract_version": 1,
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "camera_segment_id": "cam_003",
                "shot_id": "shot_002",
                "start_time_us": 200000,
                "end_time_us": 240000,
                "start_frame_index": 5,
                "end_frame_index_exclusive": 6,
                "view_family": "graphics",
                "framing_scale": "unknown",
                "camera_position": "unknown",
                "camera_motion": "unknown",
                "replay_status": "unknown",
                "graphics_status": "full_screen",
                "playability": "non_playable",
                "calibration_suitability": "unsuitable",
                "tracking_suitability": "unsuitable",
                "target_identity_suitability": "unsuitable",
                "classification_source": "manual",
                "confidence": None,
                "coverage": 1.0,
                "review_status": "accepted",
                "evidence_refs": [],
                "provenance_json": None,
                "contract_version": 1,
            },
        ],
    )
    return {
        "videos": _videos(run_id, video_id),
        "frames": _frames(run_id, video_id),
        "shot_boundaries": boundaries,
        "shot_segments": shots,
        "camera_view_segments": cameras,
    }


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.broadcast.contracts import (
        CONTRACT_NAMES,
        assert_broadcast_contracts_registered,
        broadcast_schema_fingerprints,
        compile_broadcast_schemas,
        load_all_broadcast_contracts,
    )
    from football_analytics.broadcast.types import ShotBoundary, ShotSegment
    from football_analytics.broadcast.validation import validate_broadcast_bundle
    from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
    from football_analytics.data.registry import load_schema_registry
    from football_analytics.data.validation import validate_table

    result = Result()
    reg_path = Path(args.registry)
    if not reg_path.is_absolute():
        reg_path = REPO_ROOT / reg_path
    if not reg_path.is_file():
        result.err(f"registry missing: {reg_path}", config=True)
        return result.finalize(strict=args.strict)

    try:
        reg = load_schema_registry(reg_path, project_root=REPO_ROOT)
        assert_broadcast_contracts_registered(registry=reg)
        specs = load_all_broadcast_contracts(registry=reg)
        schemas = compile_broadcast_schemas(registry=reg)
        fps = broadcast_schema_fingerprints(registry=reg)
        fps2 = broadcast_schema_fingerprints(registry=reg)
        if fps != fps2:
            result.err("broadcast fingerprints unstable", integrity=True)
        for name in CONTRACT_NAMES:
            if name not in schemas or len(fps[name]) != 64:
                result.err(f"compile/fingerprint failed for {name}", integrity=True)
        result.extras["fingerprints"] = fps
        result.extras["contract_names"] = list(CONTRACT_NAMES)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config/schema failure: {type(exc).__name__}: {exc}", config=True)
        return result.finalize(strict=args.strict)

    run_id = _run_id()
    fixture_root = None
    RUNTIME_ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fixture_root = RUNTIME_ROOT / "fixtures" / f"run_{stamp}"
    fixture_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    result.extras["fixture_root"] = str(fixture_root)

    try:
        bundle = _valid_bundle(run_id)
        br = validate_broadcast_bundle(
            bundle["shot_boundaries"],
            bundle["shot_segments"],
            bundle["camera_view_segments"],
            videos=bundle["videos"],
            frames=bundle["frames"],
        )
        if br.status == "FAIL":
            for e in br.errors[:10]:
                result.err(f"valid bundle failed: {e}")
        result.extras["valid_bundle_status"] = br.status

        # Typed roundtrip
        b0 = ShotBoundary.from_dict(bundle["shot_boundaries"].to_pylist()[0])
        s0 = ShotSegment.from_dict(bundle["shot_segments"].to_pylist()[0])
        if b0.to_dict()["boundary_id"] != "bnd_001":
            result.err("ShotBoundary roundtrip failed")
        if s0.duration_us != 120000:
            result.err("ShotSegment roundtrip failed")

        # Invalid: overlapping active shots
        bad_shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_a",
                    "start_time_us": 0,
                    "end_time_us": 100000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 100000,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_b",
                    "start_time_us": 50000,
                    "end_time_us": 150000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 100000,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
            ],
        )
        bad = validate_broadcast_bundle(None, bad_shots, None, videos=bundle["videos"])
        if bad.status != "FAIL":
            result.err("overlapping active shots were accepted")
        result.extras["overlap_rejected"] = bad.status == "FAIL"

        # Invalid: crowd playable
        bad_cam = _cast(
            "camera_view_segments",
            [
                {
                    "run_id": run_id,
                    "video_id": "clip_demo_01",
                    "camera_segment_id": "cam_bad",
                    "shot_id": None,
                    "start_time_us": 0,
                    "end_time_us": 40000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "view_family": "crowd",
                    "framing_scale": "wide",
                    "camera_position": "unknown",
                    "camera_motion": "static",
                    "replay_status": "live",
                    "graphics_status": "none",
                    "playability": "playable",
                    "calibration_suitability": "unknown",
                    "tracking_suitability": "unknown",
                    "target_identity_suitability": "unknown",
                    "classification_source": "manual",
                    "confidence": 0.5,
                    "coverage": 1.0,
                    "review_status": "unreviewed",
                    "evidence_refs": [],
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        bad2 = validate_broadcast_bundle(None, None, bad_cam, videos=bundle["videos"])
        if bad2.status != "FAIL":
            result.err("crowd playable segment was accepted")
        result.extras["crowd_playable_rejected"] = bad2.status == "FAIL"

        # Table semantic validation
        for name in CONTRACT_NAMES:
            vr = validate_table(bundle[name], specs[name])
            if vr.status == "FAIL":
                result.err(f"table validation failed for {name}: {vr.errors[:3]}")

        # Parquet roundtrip
        for name in CONTRACT_NAMES:
            path = fixture_root / f"{name}.parquet"
            write_contract_parquet(bundle[name], path, specs[name], contain_root=fixture_root)
            loaded = read_contract_parquet(path, specs[name], contain_root=fixture_root)
            if loaded.num_rows != bundle[name].num_rows:
                result.err(f"row mismatch {name}", integrity=True)
            if loaded.to_pylist() != bundle[name].to_pylist():
                result.err(f"content mismatch {name}", integrity=True)
        result.extras["parquet_roundtrip"] = True
    except Exception as exc:  # noqa: BLE001
        result.err(f"runtime checks failed: {type(exc).__name__}: {exc}", integrity=True)
    finally:
        if fixture_root is not None and fixture_root.exists():
            shutil.rmtree(fixture_root, ignore_errors=False)
            if fixture_root.exists():
                result.err("fixture cleanup incomplete", integrity=True)
            else:
                result.extras["fixture_cleaned"] = True

    return result.finalize(strict=bool(args.strict))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 4A broadcast contract validator")
    p.add_argument(
        "--registry",
        default="configs/data/schema_registry.yaml",
        help="Path to schema registry YAML",
    )
    p.add_argument("--json-out")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    try:
        result = run_checks(args)
    except Exception as exc:  # noqa: BLE001
        print(f"status=FAIL exit_code={EXIT_CONFIG} error={type(exc).__name__}", file=sys.stderr)
        return EXIT_CONFIG
    payload = result.to_dict()
    if args.json_out:
        out = Path(args.json_out)
    else:
        RUNTIME_ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = RUNTIME_ROOT / f"broadcast_contract_validation_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result.extras["report_path"] = str(out)
    if not args.quiet:
        for w in result.warnings:
            print(f"WARNING: {w}")
        for e in result.errors:
            print(f"ERROR: {e}")
        print(f"status={result.status} exit_code={result.exit_code} report={out}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
