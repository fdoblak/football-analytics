#!/usr/bin/env python3
"""Validate Stage 3B safe FFprobe media probe pipeline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/video_probe_checks")


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


def run_checks(args: argparse.Namespace) -> Result:
    import importlib.metadata as md

    from football_analytics.video.contracts import load_ingest_policy
    from football_analytics.video.ffprobe import get_ffprobe_version, resolve_ffprobe_binary
    from football_analytics.video.fixtures import generate_cfr_video
    from football_analytics.video.probe_parser import (
        map_ffprobe_json_to_video_probe,
        parse_rational,
    )
    from football_analytics.video.probe_service import run_media_probe
    from football_analytics.video.types import Rational

    result = Result()
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    try:
        policy = load_ingest_policy(policy_path)
    except Exception as exc:  # noqa: BLE001
        result.err(f"policy load failed: {exc}", config=True)
        return result.finalize(strict=args.strict)

    ff = policy["ffprobe_policy"]
    result.extras["policy_version"] = policy["policy_version"]
    result.extras["ffprobe_path"] = ff["ffprobe_binary"]
    result.extras["cleanup_verified"] = False
    result.extras["synthetic_cases"] = []
    result.extras["security_cases"] = []
    result.extras["accepted_cases"] = []
    result.extras["rejected_cases"] = []
    result.extras["timeouts"] = 0

    try:
        binary = resolve_ffprobe_binary(
            ff["ffprobe_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
        )
        version = get_ffprobe_version(binary)
        result.extras["ffprobe_version"] = version.to_dict()
    except Exception as exc:  # noqa: BLE001
        result.err(f"ffprobe unavailable: {exc}", config=True)
        return result.finalize(strict=args.strict)

    # Parser reference fixtures (no subprocess)
    try:
        assert parse_rational("0/0", label="t") is None
        assert parse_rational("30000/1001", label="t") == Rational(30000, 1001)
        sample = {
            "format": {"format_name": "mp4", "duration": "1.000000", "bit_rate": "800000"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 160,
                    "height": 120,
                    "pix_fmt": "yuv420p",
                    "r_frame_rate": "25/1",
                    "avg_frame_rate": "25/1",
                    "time_base": "1/12800",
                    "disposition": {"default": 1, "attached_pic": 0, "forced": 0},
                }
            ],
        }
        probe = map_ffprobe_json_to_video_probe(
            sample,
            source_id="src_ref_one",
            source_sha256="a" * 64,
            file_size_bytes=1024,
            probe_tool_version=version.version_token,
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        result.extras["contract_fingerprints"] = {"probe_ref": probe.fingerprint()}
    except Exception as exc:  # noqa: BLE001
        result.err(f"parser reference failed: {exc}")

    session = RUNTIME_ROOT / f"validator_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    session.mkdir(parents=True, exist_ok=True)
    try:
        media = session / "cfr.mp4"
        generate_cfr_video(media, with_audio=False)
        out = session / "out_ok"
        out.mkdir()
        res = run_media_probe(
            source=str(media),
            output_dir=str(out),
            policy=policy,
            contain_root=RUNTIME_ROOT,
        )
        if not res.accepted:
            result.err(f"synthetic CFR not accepted: {res.error_code}")
            result.extras["rejected_cases"].append("cfr")
        else:
            result.extras["accepted_cases"].append("cfr")
            result.extras["synthetic_cases"].append("cfr_video_only")

        # video+audio
        media2 = session / "cfr_a.mp4"
        generate_cfr_video(media2, with_audio=True)
        out2 = session / "out_audio"
        out2.mkdir()
        res2 = run_media_probe(
            source=str(media2),
            output_dir=str(out2),
            policy=policy,
            contain_root=RUNTIME_ROOT,
        )
        if res2.accepted:
            result.extras["accepted_cases"].append("cfr_audio")
            result.extras["synthetic_cases"].append("cfr_video_audio")
        else:
            result.err(f"synthetic audio case rejected: {res2.error_code}")

        # invalid media
        bad = session / "bad.mp4"
        bad.write_bytes(b"not-a-video")
        out3 = session / "out_bad"
        out3.mkdir()
        res3 = run_media_probe(
            source=str(bad),
            output_dir=str(out3),
            policy=policy,
            contain_root=RUNTIME_ROOT,
        )
        if res3.accepted:
            result.err("invalid media incorrectly accepted", integrity=True)
        else:
            result.extras["rejected_cases"].append("invalid_media")
            result.extras["security_cases"].append("invalid_media_rejected")

        # URL rejection
        out4 = session / "out_url"
        out4.mkdir()
        res4 = run_media_probe(
            source="https://example.com/a.mp4",
            output_dir=str(out4),
            policy=policy,
            contain_root=RUNTIME_ROOT,
        )
        if res4.accepted:
            result.err("URL incorrectly accepted", integrity=True)
        else:
            result.extras["security_cases"].append("url_rejected")
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
        out = RUNTIME_ROOT / f"video_probe_validation_{stamp}.json"
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
