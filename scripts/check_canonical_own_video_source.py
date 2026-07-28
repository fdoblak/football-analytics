#!/usr/bin/env python3
"""Validate R0-F2 canonical own-video source config + runtime manifest.

Exit codes:
  0 PASS / PASS_WITH_WARNINGS
  1 finding
  2 config
  3 integrity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"
DEFAULT_CFG = REPO_ROOT / "configs" / "sources" / "own_video_97b298e4.yaml"


def sha256_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    extras: dict[str, Any] = {}

    if not args.config.is_file():
        errors.append(f"missing config: {args.config}")
        payload = {
            "status": "FAIL",
            "exit_code": 2,
            "errors": errors,
            "warnings": warnings,
            "extras": extras,
        }
        if args.json_out:
            args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print("status=FAIL exit_code=2")
            for err in errors:
                print(f"ERROR: {err}")
        return 2

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        errors.append("config is not a mapping")
        return 2

    man_path = Path(str(cfg.get("runtime_manifest") or ""))
    if not man_path.is_file():
        errors.append(f"runtime manifest missing: {man_path}")
    else:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        for key in (
            "source_id",
            "canonical_absolute_path",
            "sha256",
            "size_bytes",
            "original_path_status",
        ):
            if cfg.get(key) != manifest.get(key):
                errors.append(f"config/manifest mismatch on {key}")

    canon = Path(str(cfg.get("canonical_absolute_path") or ""))
    extras["canonical_path"] = str(canon)
    if not canon.is_file():
        errors.append(f"canonical video missing: {canon}")
    else:
        if canon.is_symlink():
            errors.append("canonical video must not be a symlink")
        mode = canon.stat().st_mode
        if mode & stat.S_IWUSR:
            errors.append("canonical video is owner-writable; expected read-only")
        size = canon.stat().st_size
        if size != int(cfg.get("size_bytes") or -1):
            errors.append(f"size mismatch file={size} config={cfg.get('size_bytes')}")
        digest = sha256_file(canon)
        extras["sha256"] = digest
        if digest != EXPECTED_SHA or digest != cfg.get("sha256"):
            errors.append("SHA-256 mismatch against expected/config")
        if os.access(canon, os.W_OK):
            warnings.append("os.access reports writable (filesystem ACL quirk?)")

    if cfg.get("original_path_status") != "missing":
        warnings.append("original_path_status expected missing for Downloads path")
    original = Path(str(cfg.get("original_reported_path") or ""))
    if original.exists():
        warnings.append(f"original reported path unexpectedly exists: {original}")

    # Git must not track the binary (path outside repo or ignored).
    try:
        rel = canon.resolve().relative_to(REPO_ROOT.resolve())
        errors.append(f"canonical video is inside git repo: {rel}")
    except ValueError:
        extras["git_excluded"] = True

    exit_code = 0
    status = "PASS"
    if errors:
        status = "FAIL"
        exit_code = 3 if any("SHA" in e or "missing" in e for e in errors) else 1
    elif warnings:
        status = "PASS_WITH_WARNINGS"

    payload = {
        "status": status,
        "exit_code": exit_code,
        "errors": errors,
        "warnings": warnings,
        "extras": extras,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"status={status} exit_code={exit_code}")
        for err in errors:
            print(f"ERROR: {err}")
        for warn in warnings:
            print(f"WARNING: {warn}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
