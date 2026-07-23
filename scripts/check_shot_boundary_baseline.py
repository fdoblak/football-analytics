#!/usr/bin/env python3
"""Validate Stage 4B shot boundary baseline (synthetic fixtures + metrics).

Exit codes:
  0 success
  1 validation finding
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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/shot_boundary_checks")


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

    def finalize(self) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
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
            "overall_status": self.status,
        }
        body.update(self.extras)
        return body


def _load_gt(path: Path) -> tuple[list[dict[str, Any]], int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("boundaries") or []), payload.get("duration_us")


def _detect_fixture(
    *,
    manifest: dict[str, Any],
    session: Path,
    config: Any,
    run_id: str,
    video_id: str,
) -> tuple[Any, list[dict[str, Any]]]:
    from football_analytics.broadcast.contracts import load_broadcast_contract
    from football_analytics.broadcast.shot_service import (
        prepare_cfr_timeline_for_video,
        run_shot_boundary_detection,
    )
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.video.types import MappingQuality

    video = Path(manifest["video_path"])
    work = session / f"work_{video_id}"
    work.mkdir(parents=True, exist_ok=True)
    frames_path = work / "frames.parquet"
    prepare_cfr_timeline_for_video(
        video,
        frames_out=frames_path,
        run_id=run_id or generate_run_id(),
        video_id=video_id,
        fps=int(manifest["fps"]),
        contain_root=RUNTIME_ROOT,
    )
    out = work / "detect"
    out.mkdir()
    res = run_shot_boundary_detection(
        source=str(video),
        timeline=str(frames_path),
        output_dir=str(out),
        config=config,
        contain_root=RUNTIME_ROOT,
        run_id=run_id,
        video_id=video_id,
        mapping_quality=MappingQuality.DERIVED_WITH_CONSTANT_OFFSET,
    )
    if not res.accepted:
        raise RuntimeError(f"detect failed for {video_id}: {res.error_code}")
    table = read_contract_parquet(
        Path(str(res.boundaries_parquet)), load_broadcast_contract("shot_boundaries")
    )
    return res, table.to_pylist()


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.broadcast.shot_config import (
        load_shot_boundary_config,
        shot_config_fingerprint,
    )
    from football_analytics.broadcast.shot_evaluation import evaluate_from_rows
    from football_analytics.broadcast.shot_fixtures import materialize_standard_fixtures
    from football_analytics.core.run_id import generate_run_id

    result = Result()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_shot_boundary_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    result.extras["config_fingerprint"] = shot_config_fingerprint(config)
    result.extras["thresholds"] = {
        "hard_cut_f1_min": 0.95,
        "gradual_f1_min": 0.80,
        "overall_f1_min": 0.90,
        "negative_fp_max": 0,
    }
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(
        tempfile.mkdtemp(
            prefix=f"s4b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_",
            dir=str(RUNTIME_ROOT),
        )
    )
    result.extras["session"] = str(session)
    try:
        fixtures = materialize_standard_fixtures(session_dir=session)
        result.extras["fixtures"] = {k: v.get("video_sha256") for k, v in fixtures.items()}
        tol = int(config["evaluation"]["matching_tolerance_us"])
        run_id = generate_run_id()

        # --- Frozen evaluation metrics ---
        eval_names = ("hard_cut", "dissolve", "fade")
        all_pred: list[dict[str, Any]] = []
        all_gt: list[dict[str, Any]] = []
        hard_pred: list[dict[str, Any]] = []
        hard_gt: list[dict[str, Any]] = []
        gradual_pred: list[dict[str, Any]] = []
        gradual_gt: list[dict[str, Any]] = []
        per_fixture: dict[str, Any] = {}

        for name in eval_names:
            man = fixtures[name]
            gt_rows, duration_us = _load_gt(Path(man["ground_truth_path"]))
            _res, pred_rows = _detect_fixture(
                manifest=man,
                session=session,
                config=config,
                run_id=run_id,
                video_id=f"vid_{name}",
            )
            metrics = evaluate_from_rows(
                pred_rows, gt_rows, tolerance_us=tol, duration_us=duration_us
            )
            per_fixture[name] = metrics.to_dict()
            all_pred.extend(pred_rows)
            all_gt.extend(gt_rows)
            if name == "hard_cut":
                hard_pred.extend(pred_rows)
                hard_gt.extend(gt_rows)
            else:
                gradual_pred.extend(pred_rows)
                gradual_gt.extend(gt_rows)

        hard_m = evaluate_from_rows(hard_pred, hard_gt, tolerance_us=tol)
        gradual_m = evaluate_from_rows(gradual_pred, gradual_gt, tolerance_us=tol)
        overall_m = evaluate_from_rows(all_pred, all_gt, tolerance_us=tol)
        result.extras["metrics"] = {
            "hard_cut": hard_m.to_dict(),
            "gradual": gradual_m.to_dict(),
            "overall": overall_m.to_dict(),
            "per_fixture": per_fixture,
        }

        if hard_m.f1 is None or hard_m.f1 < 0.95:
            result.err(f"hard-cut F1 below 0.95: {hard_m.f1}")
        if gradual_m.f1 is None or gradual_m.f1 < 0.80:
            result.err(f"gradual F1 below 0.80: {gradual_m.f1}")
        if overall_m.f1 is None or overall_m.f1 < 0.90:
            result.err(f"overall F1 below 0.90: {overall_m.f1}")

        # --- Negative controls: FP must be 0 ---
        neg_fp = 0
        for name in ("flash", "static", "pan"):
            man = fixtures[name]
            gt_rows, duration_us = _load_gt(Path(man["ground_truth_path"]))
            _res, pred_rows = _detect_fixture(
                manifest=man,
                session=session,
                config=config,
                run_id=run_id,
                video_id=f"vid_{name}",
            )
            metrics = evaluate_from_rows(
                pred_rows, gt_rows, tolerance_us=tol, duration_us=duration_us
            )
            per_fixture[name] = metrics.to_dict()
            neg_fp += metrics.false_positives
            if metrics.false_positives != 0:
                result.err(f"negative-control {name} FP={metrics.false_positives}")
        result.extras["negative_false_positives"] = neg_fp

        # --- Determinism: repeat hard_cut ---
        man = fixtures["hard_cut"]
        _r1, rows1 = _detect_fixture(
            manifest=man,
            session=session / "det_a",
            config=config,
            run_id=run_id,
            video_id="vid_hard_cut_a",
        )
        _r2, rows2 = _detect_fixture(
            manifest=man,
            session=session / "det_b",
            config=config,
            run_id=run_id,
            video_id="vid_hard_cut_b",
        )
        # Compare boundary times + types (ids differ by video_id)
        key1 = [(r["boundary_time_us"], r["transition_type"]) for r in rows1]
        key2 = [(r["boundary_time_us"], r["transition_type"]) for r in rows2]
        deterministic = key1 == key2
        result.extras["deterministic_repeat"] = deterministic
        if not deterministic:
            result.err(f"deterministic repeat mismatch: {key1} vs {key2}")

        result.extras["cleanup_verified"] = False
    except Exception as exc:  # noqa: BLE001
        result.err(f"validator exception: {type(exc).__name__}: {exc}", integrity=True)
    finally:
        shutil.rmtree(session, ignore_errors=True)
        result.extras["cleanup_verified"] = not session.exists()

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/broadcast/shot_boundary_baseline.yaml",
        help="Shot boundary baseline config",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=RUNTIME_ROOT,
        help="Directory for validation JSON report",
    )
    args = parser.parse_args(argv)
    result = run_checks(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"shot_boundary_validation_{stamp}.json"
    report_path.write_text(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
    print(f"status: {result.status}")
    print(f"exit_code: {result.exit_code}")
    print(f"report: {report_path}")
    if result.errors:
        for e in result.errors:
            print(f"error: {e}")
    metrics = result.extras.get("metrics") or {}
    for label in ("hard_cut", "gradual", "overall"):
        m = metrics.get(label) or {}
        print(f"{label}_f1: {m.get('f1')}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
