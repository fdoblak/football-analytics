#!/usr/bin/env python3
"""Validate Stage 3C safe FFmpeg video normalization pipeline.

Exit codes:
  0 success (PASS / PASS_WITH_WARNINGS)
  1 validation finding
  2 configuration failure
  3 integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/video_normalization_checks")
DiskUsage = namedtuple("DiskUsage", "total used free")


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


def _ample_disk(_path: str) -> DiskUsage:
    return DiskUsage(total=10**15, used=0, free=10**15)


def _generate_mpeg4(path: Path) -> Path:
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=1:r=25",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    return path


def run_checks(args: argparse.Namespace) -> Result:
    import importlib.metadata as md

    from football_analytics.core.hashing import sha256_file
    from football_analytics.video.contracts import load_ingest_policy
    from football_analytics.video.ffmpeg import (
        assert_libx264_available,
        get_ffmpeg_version,
        resolve_ffmpeg_binary,
    )
    from football_analytics.video.fixtures import generate_cfr_video
    from football_analytics.video.normalization import plan_normalization
    from football_analytics.video.normalization_service import run_video_normalization
    from football_analytics.video.types import NormalizationStatus

    result = Result()
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    try:
        policy = load_ingest_policy(policy_path)
    except Exception as exc:  # noqa: BLE001
        result.err(f"policy load failed: {exc}", config=True)
        return result.finalize(strict=args.strict)

    ff = policy["ffmpeg_policy"]
    result.extras["policy_version"] = policy["policy_version"]
    result.extras["ffmpeg_path"] = ff["ffmpeg_binary"]
    result.extras["cleanup_verified"] = False
    result.extras["synthetic_cases"] = []
    result.extras["security_cases"] = []
    result.extras["accepted_cases"] = []
    result.extras["skipped_cases"] = []

    try:
        binary = resolve_ffmpeg_binary(
            ff["ffmpeg_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
        )
        version = get_ffmpeg_version(binary)
        assert_libx264_available(binary)
        result.extras["ffmpeg_version"] = version.to_dict()
    except Exception as exc:  # noqa: BLE001
        result.err(f"ffmpeg unavailable: {exc}", config=True)
        return result.finalize(strict=args.strict)

    session = RUNTIME_ROOT / f"validator_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    session.mkdir(parents=True, exist_ok=True)
    try:
        # Canonical skip
        media = session / "cfr.mp4"
        generate_cfr_video(media, with_audio=False)
        digest = sha256_file(media)
        out = session / "out_skip.mp4"
        res = run_video_normalization(
            source=str(media),
            output=str(out),
            policy=policy,
            expected_source_sha256=digest,
            execute=True,
            contain_root=RUNTIME_ROOT,
            receipt_dir=str(session / "r_skip"),
            disk_usage_fn=_ample_disk,
        )
        if res.status != NormalizationStatus.SKIPPED:
            result.err(f"canonical not skipped: {res.status} {res.error_code}")
        else:
            result.extras["skipped_cases"].append("canonical_h264")
            result.extras["synthetic_cases"].append("cfr_skip")

        # mpeg4 → h264
        media2 = session / "mpeg4.mp4"
        _generate_mpeg4(media2)
        digest2 = sha256_file(media2)
        out2 = session / "out_norm.mp4"
        res2 = run_video_normalization(
            source=str(media2),
            output=str(out2),
            policy=policy,
            expected_source_sha256=digest2,
            execute=True,
            contain_root=RUNTIME_ROOT,
            receipt_dir=str(session / "r_norm"),
            disk_usage_fn=_ample_disk,
        )
        if res2.status != NormalizationStatus.SUCCEEDED or not out2.is_file():
            result.err(f"mpeg4 normalize failed: {res2.error_code}")
        else:
            result.extras["accepted_cases"].append("mpeg4_to_h264")
            result.extras["synthetic_cases"].append("mpeg4_normalize")

        # dry-run no output
        media3 = session / "dry.mp4"
        _generate_mpeg4(media3)
        digest3 = sha256_file(media3)
        out3 = session / "out_dry.mp4"
        res3 = run_video_normalization(
            source=str(media3),
            output=str(out3),
            policy=policy,
            expected_source_sha256=digest3,
            execute=False,
            contain_root=RUNTIME_ROOT,
            receipt_dir=str(session / "r_dry"),
            disk_usage_fn=_ample_disk,
        )
        if out3.exists():
            result.err("dry-run produced output", integrity=True)
        elif res3.status != NormalizationStatus.PLANNED:
            result.err(f"dry-run expected planned: {res3.status}")
        else:
            result.extras["synthetic_cases"].append("dry_run_planned")

        # URL rejection
        res4 = run_video_normalization(
            source="https://example.com/a.mp4",
            output=str(session / "url_out.mp4"),
            policy=policy,
            execute=False,
            contain_root=RUNTIME_ROOT,
            disk_usage_fn=_ample_disk,
        )
        if res4.accepted and res4.status == NormalizationStatus.SUCCEEDED:
            result.err("URL incorrectly accepted", integrity=True)
        else:
            result.extras["security_cases"].append("url_rejected")

        # planner import sanity
        if plan_normalization is None:
            result.err("planner missing")
    except Exception as exc:  # noqa: BLE001
        result.err(f"integration failed: {exc}")
    finally:
        shutil.rmtree(session, ignore_errors=True)
        result.extras["cleanup_verified"] = not session.exists()

    pkgs = {}
    for p in [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "opencv-python",
        "opencv-python-headless",
        "ultralytics",
        "SoccerNet",
        "pyarrow",
    ]:
        pkgs[p] = md.version(p)
    result.extras["protected_package_versions"] = pkgs
    return result.finalize(strict=args.strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="configs/video/ingest_policy.yaml")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(args)
    payload = result.to_dict()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = RUNTIME_ROOT / out.name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = RUNTIME_ROOT / f"video_normalization_validation_{stamp}.json"
    if out.exists():
        print(f"json-out already exists: {out}", file=sys.stderr)
        return EXIT_CONFIG
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    payload["report_path"] = str(out)
    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        print(f"report={out}")
        for err in result.errors:
            print(f"ERROR: {err}")
        for warn in result.warnings:
            print(f"WARNING: {warn}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
