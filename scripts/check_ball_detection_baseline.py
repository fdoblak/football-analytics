#!/usr/bin/env python3
"""Validate Stage 5C ball detection baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/ball_detection_checks")
MODEL_PATH = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
EXPECTED_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


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
            "gate": (
                "PASS_WITH_FINDINGS — BALL DETECTION BASELINE ACTIVE; "
                "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
                if self.status in {"PASS", "PASS_WITH_FINDINGS"}
                else self.status
            ),
        }
        body.update(self.extras)
        return body


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.hashing import sha256_file
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import write_contract_parquet
    from football_analytics.perception.adapters.base import RawDetectionBox
    from football_analytics.perception.ball_detection import filter_raw_ball_boxes
    from football_analytics.perception.ball_detector_config import (
        ball_detector_config_fingerprint,
        load_ball_detector_config,
    )
    from football_analytics.perception.ball_evaluation import (
        NOT_EVALUATED_BALL,
        evaluate_ball_detections,
        evaluate_ball_from_rows,
    )
    from football_analytics.perception.ball_fixtures import (
        assert_runtime_root,
        make_analysis_window_row,
        make_frame_rows,
        write_tiny_mp4_with_ball,
    )
    from football_analytics.perception.candidate_merge import BallCandidate, merge_ball_candidates
    from football_analytics.perception.detection_evaluation import parse_detection_boxes
    from football_analytics.perception.taxonomy import load_detection_taxonomy, map_model_class
    from football_analytics.perception.tiling import generate_tiles, map_tile_bbox_to_source

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_ball_detector_config(cfg_path)
        result.extras["config_fingerprint"] = ball_detector_config_fingerprint(config)
        result.extras["inference_mode"] = config["inference_mode"]
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    import yaml

    registry = yaml.safe_load((REPO_ROOT / "model_registry.yaml").read_text(encoding="utf-8"))
    entry = next(
        (m for m in registry.get("models", []) if m.get("id") == "ultralytics_yolo11n_coco_person"),
        None,
    )
    if entry is None:
        result.err("model registry missing ultralytics_yolo11n_coco_person", integrity=True)
    else:
        result.extras["model_registry_id"] = entry["id"]
        result.extras["model_sha256_registry"] = entry.get("sha256")
        result.extras["production_approved"] = entry.get("production_approved", False)
        caps = entry.get("capabilities") or {}
        sb = caps.get("sports_ball") or entry.get("classes", {}).get("sports_ball") or {}
        if int(sb.get("class_id", -1)) != 32:
            result.err("sports_ball class_id must be 32 in registry capabilities")
        if str(sb.get("class_name", "")).strip().lower() not in {"sports ball", "sports_ball"}:
            result.err('sports_ball class_name must be "sports ball"')
        if entry.get("license") == "AGPL-3.0":
            result.finding(
                "AGPL-3.0 distribution risk — evaluation_only / production_approved=false (reuse)"
            )
        if entry.get("approval") != "evaluation_only":
            result.warn("expected approval=evaluation_only")
        if entry.get("file_path") != str(MODEL_PATH):
            result.err("registry path must reuse existing yolo11n.pt", integrity=True)

    if not MODEL_PATH.is_file():
        result.err(f"model missing: {MODEL_PATH}", integrity=True)
    else:
        actual = sha256_file(MODEL_PATH)
        result.extras["model_sha256_actual"] = actual
        result.extras["model_size_bytes"] = MODEL_PATH.stat().st_size
        if actual != EXPECTED_SHA:
            result.err("model sha256 mismatch vs Stage 5C expected", integrity=True)
        if entry and actual.lower() != str(entry.get("sha256", "")).lower():
            result.err("model sha256 mismatch vs registry", integrity=True)
        if MODEL_PATH.stat().st_size != 5613764:
            result.warn(f"unexpected model size: {MODEL_PATH.stat().st_size}")

    import sys as _sys

    before = {k for k in _sys.modules if "ultralytics" in k.lower()}
    import football_analytics.perception as perc  # noqa: F401

    after = {k for k in _sys.modules if "ultralytics" in k.lower()}
    if after - before:
        result.err("import football_analytics.perception loaded ultralytics", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    tax = load_detection_taxonomy(REPO_ROOT / "configs/perception/detection_taxonomy.yaml")
    mapped = map_model_class(32, "sports_ball", taxonomy=tax)
    if mapped.entity_type.value != "ball" or mapped.role_label.value != "unknown":
        result.err("sports_ball must map to ball/unknown")

    # Tiling + merge
    tiles = generate_tiles(
        128, 96, tile_width=64, tile_height=64, overlap_x=16, overlap_y=16, max_tiles=8
    )
    if not tiles:
        result.err("tiling produced zero tiles")
    else:
        mapped_box = map_tile_bbox_to_source((1, 2, 5, 6), tiles[0], coordinate_space="tile_local")
        result.extras["tile_count"] = len(tiles)
        result.extras["tile_map_sample"] = list(mapped_box)
    merged = merge_ball_candidates(
        [BallCandidate(10, 10, 20, 20, 0.9, 32, "sports ball", "full_frame")],
        [BallCandidate(11, 11, 21, 21, 0.8, 32, "sports ball", "tile:r0c0")],
        merge_iou=0.5,
    )
    if len(merged) != 1:
        result.err("class-aware merge failed to suppress duplicate")

    filters = dict(config["filters"])
    filtered = filter_raw_ball_boxes(
        [
            RawDetectionBox(0, 0, 8, 8, 0.9, 32, "sports ball"),
            RawDetectionBox(0, 0, 40, 80, 0.99, 0, "person"),
        ],
        confidence_threshold=0.15,
        filters=filters,
        frame_width=128,
        frame_height=96,
        model_input_size=640,
        taxonomy=tax,
    )
    if len(filtered) != 1 or filtered[0].entity_type.value != "ball":
        result.err("ball filter person rejection / sports-ball accept failed")

    preds = parse_detection_boxes(
        [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 10,
                "bbox_y2": 10,
                "confidence": 0.9,
            },
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox_x1": 50,
                "bbox_y1": 50,
                "bbox_x2": 60,
                "bbox_y2": 60,
                "confidence": 0.8,
            },
        ]
    )
    gts = parse_detection_boxes(
        [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox_x1": 1,
                "bbox_y1": 1,
                "bbox_x2": 11,
                "bbox_y2": 11,
                "is_reviewed_ground_truth": True,
            }
        ]
    )
    metrics = evaluate_ball_detections(preds, gts, iou_threshold=0.5)
    if metrics.true_positives != 1 or metrics.false_positives != 1 or metrics.false_negatives != 0:
        result.err(
            f"frozen IoU match unexpected tp/fp/fn="
            f"{metrics.true_positives}/{metrics.false_positives}/{metrics.false_negatives}"
        )
    noneval = evaluate_ball_from_rows(preds, None)
    if noneval.status != NOT_EVALUATED_BALL:
        result.err("missing GT must yield NOT_EVALUATED_NO_REVIEWED_BALL_GROUND_TRUTH")
    result.finding("NOT_EVALUATED_NO_REVIEWED_BALL_GROUND_TRUTH — no reviewed football ball GT")

    session = Path(tempfile.mkdtemp(prefix="ball_val_", dir=str(RUNTIME_ROOT)))
    try:
        if MODEL_PATH.is_file():
            import pyarrow as pa

            from football_analytics.data.compiler import compile_arrow_schema
            from football_analytics.perception.adapters.ultralytics_ball import (
                UltralyticsBallAdapter,
            )
            from football_analytics.perception.ball_service import run_ball_detection

            video = session / "tiny.mp4"
            write_tiny_mp4_with_ball(video, n_frames=8)
            rid = generate_run_id()
            frames = make_frame_rows(rid, "vid_ball_01", 8)
            windows = [
                make_analysis_window_row(rid, "vid_ball_01", n_frames=8, ball_analysis="eligible")
            ]
            frames_path = session / "frames.parquet"
            windows_path = session / "analysis_windows.parquet"
            write_contract_parquet(
                pa.Table.from_pylist(
                    frames, schema=compile_arrow_schema(get_contract("frames", 1))
                ),
                frames_path,
                get_contract("frames", 1),
                contain_root=RUNTIME_ROOT,
            )
            write_contract_parquet(
                pa.Table.from_pylist(
                    windows, schema=compile_arrow_schema(get_contract("analysis_windows", 1))
                ),
                windows_path,
                get_contract("analysis_windows", 1),
                contain_root=RUNTIME_ROOT,
            )
            out1 = session / "out1"
            out1.mkdir()
            from types import MappingProxyType

            def _unfreeze(v: Any) -> Any:
                if isinstance(v, (MappingProxyType, dict)):
                    return {k: _unfreeze(x) for k, x in dict(v).items()}
                if isinstance(v, tuple):
                    return [_unfreeze(x) for x in v]
                return v

            smoke_cfg = _unfreeze(config)
            smoke_cfg["maximum_frames_per_run"] = 8
            smoke_cfg["input_size"] = 320
            smoke_cfg["inference_mode"] = "hybrid"
            smoke_cfg["tiling"]["max_tiles"] = 4
            res1 = run_ball_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out1),
                config=smoke_cfg,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ball_01",
                project_root=REPO_ROOT,
            )
            if not res1.accepted:
                result.err(f"smoke detect failed: {res1.error_code}")
            else:
                result.extras["smoke_detection_count"] = res1.detection_count
                result.extras["smoke_human_count"] = res1.human_detection_count
                result.extras["smoke_ball_count"] = res1.ball_detection_count
                if res1.human_detection_count != 0:
                    result.err("Stage 5C must keep human_detection_count=0", integrity=True)
                receipt = json.loads(Path(str(res1.receipt_json)).read_text(encoding="utf-8"))
                if receipt.get("human_detection_count") != 0:
                    result.err("receipt human_detection_count must be 0")
                if Path(str(res1.receipt_json)).name != "ball_detection_run_receipt.json":
                    result.err("expected ball_detection_run_receipt.json")
                eval_p = Path(str(res1.evaluation_json or ""))
                if eval_p.is_file():
                    ev = json.loads(eval_p.read_text(encoding="utf-8"))
                    if ev.get("status") != NOT_EVALUATED_BALL:
                        result.warn(f"smoke eval status={ev.get('status')}")
                res2 = run_ball_detection(
                    source=str(video),
                    timeline=str(frames_path),
                    analysis_windows=str(windows_path),
                    output_dir=str(out1),
                    config=smoke_cfg,
                    contain_root=RUNTIME_ROOT,
                    run_id=rid,
                    video_id="vid_ball_01",
                    project_root=REPO_ROOT,
                )
                if res2.accepted or res2.error_code != "OVERWRITE_FORBIDDEN":
                    result.err("overwrite must be forbidden")
                adapter = UltralyticsBallAdapter()
                adapter.load(str(MODEL_PATH), EXPECTED_SHA)
                names = adapter.model_names()
                if names.get(32) != "sports ball":
                    result.err(
                        f"runtime sports ball name missing/mismatch: {names.get(32)!r}",
                        integrity=True,
                    )
                import numpy as np

                img = np.zeros((96, 128, 3), dtype="uint8")
                _ = adapter.predict_balls(
                    img,
                    conf=0.15,
                    iou=0.5,
                    imgsz=320,
                    device="cpu",
                    half=False,
                    class_ids=[32],
                    class_names=["sports ball"],
                )
                adapter.unload()
                result.extras["adapter_cpu_smoke"] = True
                result.extras["sports_ball_runtime_name"] = "sports ball"
        else:
            result.warn("skipped inference smoke — model absent")
    except Exception as exc:  # noqa: BLE001
        result.err(f"smoke exception: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(session, ignore_errors=True)

    import subprocess

    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "*.pt", "*.mp4"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.stdout.strip():
        result.err(f"model/video tracked in git: {tracked.stdout.strip()}", integrity=True)

    result.extras["coordinate_space_note"] = str(config.get("coordinate_space_note", ""))[:200]
    return result.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/perception/ball_detector_baseline.yaml",
        help="Ball detector config path",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional explicit report path (default: runtime UTC name)",
    )
    args = parser.parse_args()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = run_checks(args)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (
        Path(args.json_out)
        if args.json_out
        else RUNTIME_ROOT / f"ball_detection_validation_{ts}.json"
    )
    if out.exists():
        print(f"refusing overwrite: {out}", file=sys.stderr)
        return EXIT_INTEGRITY
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"status: {result.status}")
    print(f"exit_code: {result.exit_code}")
    print(f"report: {out}")
    for f in result.findings:
        print(f"finding: {f}")
    for e in result.errors:
        print(f"error: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
