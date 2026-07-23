#!/usr/bin/env python3
"""Validate Stage 4C camera-view baseline (synthetic fixtures + metrics).

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/camera_view_checks")


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


def _load_gt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_fixture(
    *,
    manifest: dict[str, Any],
    session: Path,
    config: Any,
    run_id: str,
    video_id: str,
) -> tuple[Any, list[dict[str, Any]]]:
    from football_analytics.broadcast.camera_service import (
        prepare_cfr_timeline_for_video,
        run_camera_view_classification,
        write_single_shot_parquet,
    )
    from football_analytics.broadcast.contracts import load_broadcast_contract
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.parquet import read_contract_parquet

    video = Path(manifest["video_path"])
    work = session / f"work_{video_id}"
    work.mkdir(parents=True, exist_ok=True)
    frames_path = work / "frames.parquet"
    shots_path = work / "shot_segments.parquet"
    rid = run_id or generate_run_id()
    timeline = prepare_cfr_timeline_for_video(
        video,
        frames_out=frames_path,
        run_id=rid,
        video_id=video_id,
        fps=int(manifest["fps"]),
        contain_root=RUNTIME_ROOT,
    )
    duration_us = int(manifest["duration_us"])
    # Cover [0, last_frame_time + step) ≈ duration
    end_us = max(duration_us, timeline[-1][1] + 1)
    write_single_shot_parquet(
        shots_path,
        run_id=rid,
        video_id=video_id,
        shot_id=f"shot_{manifest['name']}",
        start_time_us=0,
        end_time_us=end_us,
        start_frame_index=0,
        end_frame_index_exclusive=len(timeline),
        contain_root=RUNTIME_ROOT,
    )
    out = work / "classify"
    out.mkdir()
    res = run_camera_view_classification(
        source=str(video),
        timeline=str(frames_path),
        shots=str(shots_path),
        output_dir=str(out),
        config=config,
        contain_root=RUNTIME_ROOT,
        run_id=rid,
        video_id=video_id,
    )
    if not res.accepted:
        raise RuntimeError(f"classify failed for {video_id}: {res.error_code}")
    table = read_contract_parquet(
        Path(str(res.cameras_parquet)), load_broadcast_contract("camera_view_segments")
    )
    rows = table.to_pylist()
    for r in rows:
        r["fixture_id"] = manifest["name"]
        r["name"] = manifest["name"]
        r["is_ood"] = bool(manifest.get("is_ood"))
    return res, rows


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.broadcast.camera_config import (
        camera_config_fingerprint,
        load_camera_view_config,
    )
    from football_analytics.broadcast.camera_evaluation import (
        combined_view_framing_macro_f1,
        evaluate_camera_predictions,
    )
    from football_analytics.broadcast.camera_fixtures import materialize_standard_fixtures
    from football_analytics.core.run_id import generate_run_id

    result = Result()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_camera_view_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    result.extras["config_fingerprint"] = camera_config_fingerprint(config)
    thresholds = {
        "view_framing_macro_f1_min": float(config["evaluation"]["view_framing_macro_f1_min"]),
        "graphics_macro_f1_min": float(config["evaluation"]["graphics_macro_f1_min"]),
        "motion_macro_f1_min": float(config["evaluation"]["motion_macro_f1_min"]),
        "playability_macro_f1_min": float(config["evaluation"]["playability_macro_f1_min"]),
        "unsafe_playable_fp_rate_max": float(config["evaluation"]["unsafe_playable_fp_rate_max"]),
        "ood_abstention_rate_min": float(config["evaluation"]["ood_abstention_rate_min"]),
    }
    result.extras["thresholds"] = thresholds
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(
        tempfile.mkdtemp(
            prefix=f"s4c_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_",
            dir=str(RUNTIME_ROOT),
        )
    )
    result.extras["session"] = str(session)
    try:
        fixtures = materialize_standard_fixtures(session_dir=session)
        result.extras["fixtures"] = {k: v.get("video_sha256") for k, v in fixtures.items()}
        run_id = generate_run_id()

        frozen_names = [
            "wide_pitch",
            "medium_pitch",
            "player_isolation",
            "fullscreen_graphics",
            "partial_overlay",
            "dominant_overlay",
            "pan_motion",
            "zoom_motion",
            "unstable_motion",
        ]
        pred_rows: list[dict[str, Any]] = []
        gt_rows: list[dict[str, Any]] = []
        per_fixture: dict[str, Any] = {}

        for name in frozen_names:
            man = fixtures[name]
            gt = _load_gt(Path(man["ground_truth_path"]))
            _res, rows = _classify_fixture(
                manifest=man,
                session=session,
                config=config,
                run_id=run_id,
                video_id=f"vid_{name}",
            )
            if not rows:
                result.err(f"no predictions for {name}")
                continue
            pred = dict(rows[0])
            pred["fixture_id"] = name
            pred["shot_id"] = gt.get("shot_id", f"shot_{name}")
            pred_rows.append(pred)
            gt_row = dict(gt)
            gt_row["video_id"] = f"vid_{name}"
            gt_row["shot_id"] = gt.get("shot_id", f"shot_{name}")
            gt_row["fixture_id"] = name
            gt_rows.append(gt_row)
            per_fixture[name] = {
                "pred": {
                    k: pred.get(k)
                    for k in (
                        "view_family",
                        "framing_scale",
                        "camera_motion",
                        "graphics_status",
                        "playability",
                    )
                },
                "gt": {
                    k: gt.get(k)
                    for k in (
                        "view_family",
                        "framing_scale",
                        "camera_motion",
                        "graphics_status",
                        "playability",
                    )
                },
            }

        # OOD
        ood_pred: list[dict[str, Any]] = []
        ood_gt: list[dict[str, Any]] = []
        for name in ("ood_crowd",):
            man = fixtures[name]
            gt = _load_gt(Path(man["ground_truth_path"]))
            _res, rows = _classify_fixture(
                manifest=man,
                session=session,
                config=config,
                run_id=run_id,
                video_id=f"vid_{name}",
            )
            pred = dict(rows[0])
            pred["fixture_id"] = name
            pred["is_ood"] = True
            ood_pred.append(pred)
            gt_row = dict(gt)
            gt_row["video_id"] = f"vid_{name}"
            gt_row["shot_id"] = gt.get("shot_id", f"shot_{name}")
            gt_row["fixture_id"] = name
            gt_row["is_ood"] = True
            ood_gt.append(gt_row)
            per_fixture[name] = {
                "pred": {"view_family": pred.get("view_family")},
                "gt": {"view_family": gt.get("view_family"), "is_ood": True},
            }

        supported = {
            "view_family": list(config["supported_axes"]["view_family"]),
            "framing_scale": list(config["supported_axes"]["framing_scale"]),
            "camera_motion": list(config["supported_axes"]["camera_motion"]),
            "graphics_status": list(config["supported_axes"]["graphics_status"]),
            "playability": list(config["supported_axes"]["playability"]),
        }
        report = evaluate_camera_predictions(
            pred_rows,
            gt_rows,
            supported_labels=supported,
            ood_fixture_ids=[],
        )
        # Merge OOD into separate ood rate calculation
        ood_report = evaluate_camera_predictions(
            ood_pred,
            ood_gt,
            supported_labels=supported,
            ood_fixture_ids=["ood_crowd"],
        )

        view_framing_f1 = combined_view_framing_macro_f1(report)
        graphics_f1 = report.axes["graphics_status"].macro_f1
        motion_f1 = report.axes["camera_motion"].macro_f1
        play_f1 = report.axes["playability"].macro_f1
        unsafe = report.unsafe_playable_false_positive_rate
        # Prefer explicit OOD abstention from ood fixtures
        ood_rate = ood_report.ood_abstention_rate
        if ood_rate is None and ood_pred:
            ood_rate = sum(1 for r in ood_pred if r.get("view_family") == "unknown") / len(ood_pred)

        metrics = {
            "view_framing_macro_f1": view_framing_f1,
            "graphics_macro_f1": graphics_f1,
            "motion_macro_f1": motion_f1,
            "playability_macro_f1": play_f1,
            "unsafe_playable_false_positive_rate": unsafe,
            "ood_abstention_rate": ood_rate,
            "axes": {k: v.to_dict() for k, v in report.axes.items()},
            "per_fixture": per_fixture,
        }
        result.extras["metrics"] = metrics

        if view_framing_f1 is None or view_framing_f1 < thresholds["view_framing_macro_f1_min"]:
            result.err(
                f"view/framing macro F1 below {thresholds['view_framing_macro_f1_min']}: "
                f"{view_framing_f1}"
            )
        if graphics_f1 is None or graphics_f1 < thresholds["graphics_macro_f1_min"]:
            result.err(
                f"graphics macro F1 below {thresholds['graphics_macro_f1_min']}: {graphics_f1}"
            )
        if motion_f1 is None or motion_f1 < thresholds["motion_macro_f1_min"]:
            result.err(f"motion macro F1 below {thresholds['motion_macro_f1_min']}: {motion_f1}")
        if play_f1 is None or play_f1 < thresholds["playability_macro_f1_min"]:
            result.err(
                f"playability macro F1 below {thresholds['playability_macro_f1_min']}: {play_f1}"
            )
        if unsafe is None:
            result.warn("unsafe playable FP rate not_evaluable (no non_playable GT)")
        elif unsafe > thresholds["unsafe_playable_fp_rate_max"]:
            result.err(f"unsafe playable FP rate > 0: {unsafe}")
        if ood_rate is None or ood_rate < thresholds["ood_abstention_rate_min"]:
            result.err(
                f"OOD abstention rate below {thresholds['ood_abstention_rate_min']}: {ood_rate}"
            )

        # Determinism: repeat wide_pitch
        man = fixtures["wide_pitch"]
        _r1, rows1 = _classify_fixture(
            manifest=man,
            session=session / "det_a",
            config=config,
            run_id=run_id,
            video_id="vid_wide_a",
        )
        _r2, rows2 = _classify_fixture(
            manifest=man,
            session=session / "det_b",
            config=config,
            run_id=run_id,
            video_id="vid_wide_b",
        )
        keys = (
            "view_family",
            "framing_scale",
            "camera_motion",
            "graphics_status",
            "playability",
            "coverage",
        )
        key1 = [{k: rows1[0].get(k) for k in keys}]
        key2 = [{k: rows2[0].get(k) for k in keys}]
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
        default="configs/broadcast/camera_view_baseline.yaml",
        help="Camera-view baseline config",
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
    report_path = report_dir / f"camera_view_validation_{stamp}.json"
    report_path.write_text(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
    print(f"status: {result.status}")
    print(f"exit_code: {result.exit_code}")
    print(f"report: {report_path}")
    if result.errors:
        for e in result.errors:
            print(f"error: {e}")
    metrics = result.extras.get("metrics") or {}
    for label in (
        "view_framing_macro_f1",
        "graphics_macro_f1",
        "motion_macro_f1",
        "playability_macro_f1",
        "unsafe_playable_false_positive_rate",
        "ood_abstention_rate",
    ):
        print(f"{label}: {metrics.get(label)}")
    print(f"deterministic_repeat: {result.extras.get('deterministic_repeat')}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
