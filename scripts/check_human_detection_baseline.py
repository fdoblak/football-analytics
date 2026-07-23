#!/usr/bin/env python3
"""Validate Stage 5B human detection baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_detection_checks")
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
                "PASS_WITH_FINDINGS — HUMAN DETECTION BASELINE ACTIVE; "
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
    from football_analytics.perception.adapters.base import RawPersonBox
    from football_analytics.perception.detection_evaluation import (
        NOT_EVALUATED,
        evaluate_from_rows,
        evaluate_human_detections,
        parse_detection_boxes,
    )
    from football_analytics.perception.human_detection import filter_raw_person_boxes
    from football_analytics.perception.human_detector_config import (
        human_detector_config_fingerprint,
        load_human_detector_config,
    )
    from football_analytics.perception.human_fixtures import (
        assert_runtime_root,
        make_analysis_window_row,
        make_frame_rows,
        write_tiny_mp4,
    )
    from football_analytics.perception.taxonomy import load_detection_taxonomy, map_model_class
    from football_analytics.perception.transforms import (
        build_preprocessing_transform,
        inverse_bbox,
        roundtrip_bbox,
    )

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_human_detector_config(cfg_path)
        result.extras["config_fingerprint"] = human_detector_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    # Model artifact / hash / registry
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
        if entry.get("license") == "AGPL-3.0":
            result.finding(
                "AGPL-3.0 distribution risk — evaluation_only / production_approved=false"
            )
        if entry.get("approval") != "evaluation_only":
            result.warn("expected approval=evaluation_only")

    if not MODEL_PATH.is_file():
        result.err(f"model missing: {MODEL_PATH}", integrity=True)
    else:
        actual = sha256_file(MODEL_PATH)
        result.extras["model_sha256_actual"] = actual
        if actual != EXPECTED_SHA:
            result.err("model sha256 mismatch vs Stage 5B expected", integrity=True)
        if entry and actual.lower() != str(entry.get("sha256", "")).lower():
            result.err("model sha256 mismatch vs registry", integrity=True)
        prov = Path(str(MODEL_PATH) + ".provenance.json")
        if not prov.is_file():
            result.warn("provenance JSON sibling missing")

    # Lazy import: perception package must not load YOLO weights
    import sys as _sys

    before = {k for k in _sys.modules if "ultralytics" in k.lower()}
    import football_analytics.perception as perc  # noqa: F401

    after = {k for k in _sys.modules if "ultralytics" in k.lower()}
    if after - before:
        result.err("import football_analytics.perception loaded ultralytics", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    # Taxonomy mapping person → human/unknown
    tax = load_detection_taxonomy(REPO_ROOT / "configs/perception/detection_taxonomy.yaml")
    mapped = map_model_class(0, "person", taxonomy=tax)
    if mapped.entity_type.value != "human" or mapped.role_label.value != "unknown":
        result.err("person must map to human/unknown")
    if mapped.role_label.value == "player":
        result.err("person must never map to player", integrity=True)

    # Preprocessing / inverse roundtrip
    tf = build_preprocessing_transform(
        source_width=128, source_height=72, model_input_width=640, model_input_height=640
    )
    try:
        roundtrip_bbox((10.0, 10.0, 40.0, 50.0), tf)
        inv = inverse_bbox((100.0, 100.0, 200.0, 300.0), tf)
        result.extras["inverse_sample"] = list(inv)
    except Exception as exc:  # noqa: BLE001
        result.err(f"transform failed: {exc}")

    # Evaluator frozen fixtures
    preds = parse_detection_boxes(
        [
            {
                "frame_index": 0,
                "entity_type": "human",
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 10,
                "bbox_y2": 10,
                "confidence": 0.9,
            },
            {
                "frame_index": 0,
                "entity_type": "human",
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
                "entity_type": "human",
                "bbox_x1": 1,
                "bbox_y1": 1,
                "bbox_x2": 11,
                "bbox_y2": 11,
                "is_reviewed_ground_truth": True,
            }
        ]
    )
    metrics = evaluate_human_detections(preds, gts, iou_threshold=0.5)
    if metrics.true_positives != 1 or metrics.false_positives != 1 or metrics.false_negatives != 0:
        result.err(
            f"frozen IoU match unexpected tp/fp/fn="
            f"{metrics.true_positives}/{metrics.false_positives}/{metrics.false_negatives}"
        )
    noneval = evaluate_from_rows(preds, None)
    if noneval.status != NOT_EVALUATED:
        result.err("missing GT must yield NOT_EVALUATED_NO_REVIEWED_GROUND_TRUTH")
    result.finding(
        "NOT_EVALUATED_NO_REVIEWED_GROUND_TRUTH — no reviewed football GT for real accuracy"
    )

    # Pure filter rejects ball-like classes via taxonomy (sports_ball → ball entity)
    filtered = filter_raw_person_boxes(
        [
            RawPersonBox(0, 0, 20, 40, 0.9, 0, "person"),
            RawPersonBox(0, 0, 5, 5, 0.9, 32, "sports_ball"),
        ],
        confidence_threshold=0.25,
        minimum_bbox_area=64.0,
        maximum_aspect_ratio=8.0,
        frame_width=128,
        frame_height=72,
        model_input_size=640,
        taxonomy=tax,
    )
    if len(filtered) != 1 or filtered[0].role_label.value != "unknown":
        result.err("filter_raw_person_boxes failed person/ball separation")

    # Bounded inference smoke
    session = Path(tempfile.mkdtemp(prefix="human_val_", dir=str(RUNTIME_ROOT)))
    try:
        if MODEL_PATH.is_file():
            import pyarrow as pa

            from football_analytics.data.compiler import compile_arrow_schema
            from football_analytics.perception.adapters.ultralytics_person import (
                UltralyticsPersonAdapter,
            )
            from football_analytics.perception.detection_service import run_human_detection

            video = session / "tiny.mp4"
            write_tiny_mp4(video, n_frames=8)
            rid = generate_run_id()
            frames = make_frame_rows(rid, "vid_human_01", 8)
            windows = [make_analysis_window_row(rid, "vid_human_01", n_frames=8)]
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
            smoke_cfg = dict(config)
            # MappingProxy → plain via rebuild from load is frozen; override via mutable copy
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
            res1 = run_human_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out1),
                config=smoke_cfg,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_human_01",
                project_root=REPO_ROOT,
            )
            if not res1.accepted:
                result.err(f"smoke detect failed: {res1.error_code}")
            else:
                result.extras["smoke_detection_count"] = res1.detection_count
                result.extras["smoke_ball_count"] = res1.ball_detection_count
                if res1.ball_detection_count != 0:
                    result.err("Stage 5B must keep ball_detection_count=0", integrity=True)
                receipt = json.loads(Path(str(res1.receipt_json)).read_text(encoding="utf-8"))
                if receipt.get("ball_detection_count") != 0:
                    result.err("receipt ball_detection_count must be 0")
                # Atomic / no-overwrite
                res2 = run_human_detection(
                    source=str(video),
                    timeline=str(frames_path),
                    analysis_windows=str(windows_path),
                    output_dir=str(out1),
                    config=smoke_cfg,
                    contain_root=RUNTIME_ROOT,
                    run_id=rid,
                    video_id="vid_human_01",
                    project_root=REPO_ROOT,
                )
                if res2.accepted or res2.error_code != "OVERWRITE_FORBIDDEN":
                    result.err("overwrite must be forbidden")
                # Adapter direct predict smoke
                adapter = UltralyticsPersonAdapter()
                adapter.load(str(MODEL_PATH), EXPECTED_SHA)
                import numpy as np

                img = np.zeros((48, 64, 3), dtype="uint8")
                _ = adapter.predict_persons(
                    img,
                    conf=0.25,
                    iou=0.5,
                    imgsz=320,
                    device="cpu",
                    half=False,
                    class_ids=[0],
                    class_names=["person"],
                )
                adapter.unload()
                result.extras["adapter_cpu_smoke"] = True
        else:
            result.warn("skipped inference smoke — model absent")
    except Exception as exc:  # noqa: BLE001
        result.err(f"smoke exception: {type(exc).__name__}: {exc}")
    finally:
        # Cleanup intermediate temps; keep validation report elsewhere
        shutil.rmtree(session, ignore_errors=True)

    # Ensure model/video not tracked in git
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
        default="configs/perception/human_detector_baseline.yaml",
        help="Human detector config path",
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
        else RUNTIME_ROOT / f"human_detection_validation_{ts}.json"
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
