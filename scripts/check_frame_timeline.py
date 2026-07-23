#!/usr/bin/env python3
"""Validate Stage 3D frame timeline pipeline (synthetic E2E).

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/frame_timeline_checks")
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


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.hashing import sha256_file
    from football_analytics.video.contracts import load_ingest_policy
    from football_analytics.video.fixtures import generate_cfr_video
    from football_analytics.video.frame_timeline_service import run_frame_timeline
    from football_analytics.video.normalization_service import run_video_normalization
    from football_analytics.video.probe_service import run_media_probe
    from football_analytics.video.types import FrameTimelineMode, FrameTimelineStatus

    result = Result()
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    try:
        policy = load_ingest_policy(policy_path)
    except Exception as exc:  # noqa: BLE001
        result.err(f"policy load failed: {exc}", config=True)
        return result.finalize(strict=args.strict)

    result.extras["policy_version"] = policy["policy_version"]
    result.extras["cleanup_verified"] = False
    result.extras["cases"] = []

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = RUNTIME_ROOT / f"e2e_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    session.mkdir(parents=True, exist_ok=True)
    try:
        src = session / "source.mp4"
        generate_cfr_video(src)
        digest = sha256_file(src)
        result.extras["cases"].append({"step": "generate", "ok": True, "sha256": digest})

        probe_dir = session / "probe"
        probe_dir.mkdir()
        probe_res = run_media_probe(
            source=str(src),
            output_dir=str(probe_dir),
            policy=policy,
            contain_root=RUNTIME_ROOT,
        )
        if not probe_res.accepted:
            result.err(f"probe failed: {probe_res.error_code}", integrity=True)
        else:
            result.extras["cases"].append({"step": "probe", "ok": True})

        norm_out = session / "normalized.mp4"
        norm_receipt_dir = session / "norm_receipts"
        norm_receipt_dir.mkdir()
        norm = run_video_normalization(
            source=str(src),
            output=str(norm_out),
            policy=policy,
            expected_source_sha256=digest,
            execute=True,
            contain_root=RUNTIME_ROOT,
            receipt_dir=str(norm_receipt_dir),
            disk_usage_fn=_ample_disk,
        )
        timeline_source = Path(norm.output_path) if norm.output_path else src
        if not timeline_source.is_file():
            timeline_source = src
        timeline_sha = sha256_file(timeline_source)
        result.extras["cases"].append(
            {
                "step": "normalize",
                "ok": True,
                "status": norm.status.value,
                "output": str(timeline_source),
            }
        )
        norm_receipt = norm_receipt_dir / "normalization_receipt.json"

        tl_dir = session / "timeline"
        tl_dir.mkdir()
        tl = run_frame_timeline(
            source=str(timeline_source),
            output_dir=str(tl_dir),
            policy=policy,
            mode=FrameTimelineMode.TIMELINE_ONLY,
            contain_root=RUNTIME_ROOT,
            expected_source_sha256=timeline_sha,
            video_id="vid_e2e",
            normalization_receipt=str(norm_receipt) if norm_receipt.is_file() else None,
        )
        if not tl.accepted or tl.status != FrameTimelineStatus.SUCCEEDED:
            result.err(f"timeline_only failed: {tl.error_code}", integrity=True)
        else:
            result.extras["cases"].append(
                {
                    "step": "timeline_only",
                    "ok": True,
                    "frame_count": tl.receipt.frame_count,
                    "mapping_quality": tl.receipt.mapping_quality.value,
                }
            )

        sm_dir = session / "sampled"
        sm_dir.mkdir()
        sm = run_frame_timeline(
            source=str(timeline_source),
            output_dir=str(sm_dir),
            policy=policy,
            mode=FrameTimelineMode.SAMPLED,
            contain_root=RUNTIME_ROOT,
            expected_source_sha256=timeline_sha,
            execute_materialize=True,
            sample_every=2,
            video_id="vid_e2e_s",
        )
        if not sm.accepted:
            result.err(f"sampled materialize failed: {sm.error_code}", integrity=True)
        else:
            result.extras["cases"].append(
                {
                    "step": "sampled",
                    "ok": True,
                    "materialized_frame_count": sm.receipt.materialized_frame_count,
                }
            )
    except Exception as exc:  # noqa: BLE001
        result.err(f"e2e exception: {exc}", integrity=True)
    finally:
        try:
            shutil.rmtree(session, ignore_errors=True)
            result.extras["cleanup_verified"] = not session.exists()
            if session.exists():
                result.warn("session cleanup incomplete")
        except Exception as exc:  # noqa: BLE001
            result.warn(f"cleanup error: {exc}")

    return result.finalize(strict=args.strict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="configs/video/ingest_policy.yaml",
        help="Ingest policy path",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_checks(args)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNTIME_ROOT / f"frame_timeline_validation_{stamp}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        print(f"report={out}")
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        for warn in result.warnings:
            print(f"WARN: {warn}")
    else:
        print(f"status={result.status} exit_code={result.exit_code}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
