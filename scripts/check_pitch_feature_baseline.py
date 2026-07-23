#!/usr/bin/env python3
"""Validate Stage 8B pitch keypoint / line detection baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/pitch_feature_checks")
KP_PATH = Path("/home/fdoblak/models/soccernet/sn-banner/SV_kp.pth")
LINES_PATH = Path("/home/fdoblak/models/soccernet/sn-banner/SV_lines.pth")
EXPECTED_KP_SHA = "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113"
EXPECTED_LINES_SHA = "2751242917f8c0f858a396e0cfe4521be39fe07bf049590eb21714526acecac1"
EXPECTED_KP_SIZE = 264964645
EXPECTED_LINES_SIZE = 264857893
GATE = (
    "PASS_WITH_FINDINGS — PITCH FEATURE DETECTION BASELINE ACTIVE; "
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
    from football_analytics.calibration.pitch_feature_config import (
        load_pitch_feature_config,
        pitch_feature_config_fingerprint,
    )
    from football_analytics.calibration.pitch_feature_evaluation import (
        NOT_EVALUATED_PITCH_FEATURES,
        evaluate_pitch_features,
    )
    from football_analytics.calibration.pitch_feature_fixtures import (
        assert_runtime_root,
        fixture_image_bundle,
        make_solid_rgb,
    )
    from football_analytics.calibration.pitch_feature_mapping import (
        NBJW_LINES_LIST,
        mapping_table_summary,
    )
    from football_analytics.calibration.pitch_feature_postprocess import (
        decode_keypoints_from_heatmap,
        decode_lines_from_heatmap,
        fit_line_from_mask,
        make_synthetic_peak_heatmap,
    )
    from football_analytics.calibration.pitch_feature_preprocess import (
        build_stretch_transform,
        model_point_to_source,
        source_point_to_model,
    )
    from football_analytics.calibration.pitch_feature_service import run_pitch_feature_detect
    from football_analytics.core.hashing import sha256_file

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_pitch_feature_config(cfg_path)
        result.extras["config_fingerprint"] = pitch_feature_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    if config["auto_homography"] is not False:
        result.err("auto_homography must be false", integrity=True)
    if config["network_sources_allowed"] is not False:
        result.err("network_sources_allowed must be false", integrity=True)

    # Registry / weights
    import yaml

    registry = yaml.safe_load((REPO_ROOT / "model_registry.yaml").read_text(encoding="utf-8"))
    models = {m["id"]: m for m in registry.get("models", []) if isinstance(m, dict)}
    for mid, path, exp_sha, exp_size in (
        ("sn_banner_sv_kp", KP_PATH, EXPECTED_KP_SHA, EXPECTED_KP_SIZE),
        ("sn_banner_sv_lines", LINES_PATH, EXPECTED_LINES_SHA, EXPECTED_LINES_SIZE),
    ):
        entry = models.get(mid)
        if entry is None:
            result.err(f"registry missing {mid}", integrity=True)
            continue
        if entry.get("production_approved") is True:
            result.err(f"{mid} must not be production_approved", integrity=True)
        if entry.get("approval") != "evaluation_only":
            result.warn(f"{mid} expected approval=evaluation_only")
        result.finding(
            f"{mid}: GPL-2.0 architecture linking risk → evaluation_only / "
            "production_approved=false"
        )
        if not path.is_file():
            result.err(f"weight missing: {path}", integrity=True)
            continue
        actual = sha256_file(path)
        size = path.stat().st_size
        result.extras[f"{mid}_sha256"] = actual
        result.extras[f"{mid}_size"] = size
        if actual != exp_sha or size != exp_size:
            result.err(f"{mid} sha/size mismatch", integrity=True)
        if actual.lower() != str(entry.get("sha256", "")).lower():
            result.err(f"{mid} sha mismatch vs registry", integrity=True)

    # Lazy import: calibration package must not load HRNet / torch weights
    import sys as _sys

    before = {k for k in _sys.modules if "cls_hrnet" in k.lower() or k == "fa_nbjw_cls_hrnet_kp"}
    import football_analytics.calibration as cal  # noqa: F401

    after = {k for k in _sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
    if after - before:
        result.err("import football_analytics.calibration loaded HRNet modules", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    # Frozen calibrations FP
    try:
        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        if fps["calibrations"] != EXPECTED_CALIBRATIONS_FP:
            result.err("calibrations fingerprint regression", integrity=True)
        result.extras["calibrations_fp"] = fps["calibrations"]
    except Exception as exc:  # noqa: BLE001
        result.err(f"fingerprint check failed: {exc}", integrity=True)

    # Mapping trailing space
    if NBJW_LINES_LIST[7] != "Goal left post left ":
        result.err("lines_list trailing-space name regression", integrity=True)
    result.extras["mapping_summary"] = mapping_table_summary()

    # Preprocess roundtrip
    tf = build_stretch_transform(source_width=320, source_height=180)
    mx, my = source_point_to_model(160.0, 90.0, tf)
    sx, sy = model_point_to_source(mx, my, tf)
    if abs(sx - 160.0) > 1e-6 or abs(sy - 90.0) > 1e-6:
        result.err("stretch inverse roundtrip failed")
    result.extras["preprocess_ok"] = True

    # Synthetic heatmap postprocess (no GPU / no weights)
    try:
        heat = make_synthetic_peak_heatmap(
            channels=57,
            height=270,
            width=480,
            peaks=[(0, 100, 120, 0.9), (1, 110, 130, 0.85), (2, 140, 200, 0.05)],
        )
        tf960 = build_stretch_transform(source_width=960, source_height=540)
        kps = decode_keypoints_from_heatmap(
            heat, transform=tf960, score_threshold=0.1, expected_channels=57
        )
        accepted = [k for k in kps if not k.rejected]
        if len(accepted) < 1:
            result.err("synthetic keypoint peak decode produced no acceptances")
        heat_l = make_synthetic_peak_heatmap(
            channels=23,
            height=270,
            width=480,
            peaks=[(0, 40, 50, 0.9), (0, 200, 400, 0.88)],
        )
        lines = decode_lines_from_heatmap(
            heat_l,
            transform=tf960,
            score_threshold=0.1,
            expected_channels=23,
            minimum_length_px=8.0,
        )
        if not any(not ln.rejected for ln in lines):
            result.err("synthetic line decode produced no acceptances")
        import numpy as np

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 10:90] = 1
        fitted = fit_line_from_mask(mask, minimum_length_px=8.0)
        if fitted is None:
            result.err("connected-component line fit failed")
        result.extras["synthetic_postprocess_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result.err(f"synthetic postprocess failed: {exc}")

    # Evaluation status
    ev = evaluate_pitch_features()
    if ev.ground_truth_evaluation_status != NOT_EVALUATED_PITCH_FEATURES:
        result.err("expected NOT_EVALUATED pitch feature GT code")
    result.extras["evaluation_status"] = NOT_EVALUATED_PITCH_FEATURES

    # Channel mismatch hard-fail
    try:
        bad = make_synthetic_peak_heatmap(channels=10, height=270, width=480, peaks=[])
        decode_keypoints_from_heatmap(
            bad, transform=tf960, score_threshold=0.1, expected_channels=57
        )
        result.err("channel mismatch should hard-fail")
    except Exception:
        result.extras["channel_mismatch_hard_fail"] = True

    session = Path(tempfile.mkdtemp(prefix="pitch_feat_", dir=str(RUNTIME_ROOT)))
    try:
        # Conditional model smoke
        if KP_PATH.is_file() and LINES_PATH.is_file() and not args.skip_model_smoke:
            bundle = fixture_image_bundle()
            # Force single small frame for smoke speed
            bundle = {
                **bundle,
                "images": [
                    {
                        "frame_index": 0,
                        "video_time_us": 0,
                        "rgb": make_solid_rgb(width=960, height=540),
                        "eligible": True,
                    }
                ],
            }
            # Use cpu_only for agent-safe smoke
            # MappingProxy → rebuild via yaml reload already frozen; unfreeze keys we need
            from football_analytics.calibration.pitch_feature_config import (
                unfreeze_pitch_feature_config,
            )

            cfg_mut = unfreeze_pitch_feature_config(config)
            cfg_mut["device_policy"] = "cpu_only"
            cfg_mut["maximum_frames_per_run"] = 1
            r1 = run_pitch_feature_detect(
                output_dir=str(session / "smoke1"),
                config=cfg_mut,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=bundle,
                project_root=REPO_ROOT,
            )
            if not r1.accepted:
                result.err(f"model smoke failed: {r1.error_code}", integrity=True)
            else:
                result.extras["model_smoke_ok"] = True
                result.extras["device"] = r1.summary.get("device")
                result.extras["feature_count"] = r1.summary.get("feature_count")
                # Schema shapes recorded in receipt
                # Determinism
                r2 = run_pitch_feature_detect(
                    output_dir=str(session / "smoke2"),
                    config=cfg_mut,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=bundle,
                    project_root=REPO_ROOT,
                )
                if r1.accepted and r2.accepted:
                    a = [
                        (x["feature_id"], x.get("image_x"), x.get("line_x1"))
                        for x in r1.summary["feature_rows"]
                    ]
                    b = [
                        (x["feature_id"], x.get("image_x"), x.get("line_x1"))
                        for x in r2.summary["feature_rows"]
                    ]
                    if a != b:
                        result.err("model smoke not deterministic")
                # No overwrite
                again = run_pitch_feature_detect(
                    output_dir=str(session / "smoke1"),
                    config=cfg_mut,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=bundle,
                    project_root=REPO_ROOT,
                )
                if again.error_code != "OVERWRITE_FORBIDDEN":
                    result.err("overwrite must be forbidden")
                # Failure cleanup
                fail = run_pitch_feature_detect(
                    output_dir=str(session / "fail"),
                    config=cfg_mut,
                    contain_root=RUNTIME_ROOT,
                    in_memory_bundle=bundle,
                    project_root=REPO_ROOT,
                    inject_failure=True,
                )
                if fail.accepted:
                    result.err("injected failure should not accept")
                if (session / "fail" / "calibration_features.parquet").exists():
                    result.err("failure cleanup incomplete", integrity=True)
        else:
            result.warn("model smoke skipped (weights missing or --skip-model-smoke)")

        # Hash mismatch
        from football_analytics.calibration.pitch_feature_adapter import verify_weight_file

        try:
            verify_weight_file(KP_PATH, expected_sha256="0" * 64, expected_size=EXPECTED_KP_SIZE)
            result.err("hash mismatch should raise")
        except Exception:
            result.extras["hash_mismatch_detected"] = True

        result.finding("No reviewed pitch-feature ground truth — " + NOT_EVALUATED_PITCH_FEATURES)
        result.finding("Real football pitch-feature accuracy not yet validated")
        result.finding("Homography solve deferred to Stage 8C")
        result.extras["adapter_choice"] = config["adapter_choice"]
        result.extras["production_approved"] = False
    except Exception as exc:  # noqa: BLE001
        result.err(f"validator scenario failed: {type(exc).__name__}: {exc}")
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)
        else:
            result.extras["session_dir"] = str(session)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/calibration/pitch_feature_baseline.yaml",
        help="Pitch feature baseline config path",
    )
    parser.add_argument("--keep", action="store_true", help="Keep validator session dir")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--skip-model-smoke",
        action="store_true",
        help="Skip loading SV weights (synthetic checks only)",
    )
    args = parser.parse_args(argv)
    result = run_checks(args)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload['status']}")
        print(f"gate: {payload['gate']}")
        print(f"exit_code: {payload['exit_code']}")
        for w in payload["warnings"]:
            print(f"warning: {w}")
        for f in payload["findings"]:
            print(f"finding: {f}")
        for e in payload["errors"]:
            print(f"error: {e}")
        for k, v in payload.items():
            if k in {
                "schema_version",
                "timestamp",
                "status",
                "exit_code",
                "errors",
                "warnings",
                "findings",
                "overall_status",
                "gate",
            }:
                continue
            print(f"{k}: {v}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
