#!/usr/bin/env python3
"""Validate Stage 2C data contracts, Parquet I/O, and migrations."""

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/data_contract_checks")


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False) -> None:
        self.errors.append(msg)
        self.exit_code = EXIT_INTEGRITY if integrity else EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code == EXIT_INTEGRITY or self.errors:
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


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.hashing import sha256_file
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.bundle import build_synthetic_bundle, validate_contract_bundle
    from football_analytics.data.compiler import compile_arrow_schema
    from football_analytics.data.fingerprint import contract_fingerprint
    from football_analytics.data.migrations import migrate_parquet
    from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
    from football_analytics.data.registry import load_schema_registry
    from football_analytics.data.validation import validate_table

    result = Result()
    reg_path = Path(args.registry)
    if not reg_path.is_file():
        result.err(f"registry missing: {reg_path}")
        return result.finalize(strict=args.strict)

    try:
        reg = load_schema_registry(reg_path, project_root=REPO_ROOT)
    except Exception as exc:  # noqa: BLE001
        result.err(f"registry load failed: {type(exc).__name__}", integrity=True)
        return result.finalize(strict=args.strict)

    fps = {}
    for name in reg.list_contracts():
        for ver in reg.get_entry(name).supported_versions:
            try:
                spec = reg.load_contract(name, ver)
                schema = compile_arrow_schema(spec)
                fp = contract_fingerprint(spec)
                fp2 = contract_fingerprint(spec)
                if fp != fp2 or len(fp) != 64:
                    result.err(f"fingerprint unstable for {name} v{ver}", integrity=True)
                if schema is None:
                    result.err(f"compile failed {name} v{ver}", integrity=True)
                fps[f"{name}:{ver}"] = fp
            except Exception as exc:  # noqa: BLE001
                result.err(f"{name} v{ver}: {type(exc).__name__}", integrity=True)
    result.extras["fingerprints"] = fps
    result.extras["contract_count"] = len(reg.list_contracts())

    if args.contract:
        try:
            spec = reg.load_contract(args.contract, args.version)
            result.extras["selected"] = {
                "contract": spec.contract_name,
                "version": spec.version,
                "fingerprint": contract_fingerprint(spec),
            }
        except Exception as exc:  # noqa: BLE001
            result.err(f"selected contract failed: {type(exc).__name__}", integrity=True)

    fixture_root = None
    if args.synthetic_roundtrip or args.migration_smoke:
        RUNTIME_ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fixture_root = RUNTIME_ROOT / "fixtures" / f"run_{stamp}"
        fixture_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        result.extras["fixture_root"] = str(fixture_root)

    try:
        if args.synthetic_roundtrip:
            assert fixture_root is not None
            rid = generate_run_id()
            bundle = build_synthetic_bundle(rid)
            specs = {n: reg.load_contract(n, 1) for n in bundle}
            br = validate_contract_bundle(bundle, specs)
            if br.status == "FAIL":
                for e in br.errors[:10]:
                    result.err(f"bundle: {e}", integrity=True)
            for name, table in bundle.items():
                path = fixture_root / f"{name}.parquet"
                write_contract_parquet(table, path, specs[name], contain_root=fixture_root)
                loaded = read_contract_parquet(path, specs[name], contain_root=fixture_root)
                if loaded.num_rows != table.num_rows:
                    result.err(f"row mismatch {name}", integrity=True)
                if loaded.to_pylist() != table.to_pylist():
                    result.err(f"content mismatch {name}", integrity=True)
                # tamper fingerprint metadata negative test on a copy
            # corrupt file negative
            det_path = fixture_root / "detections.parquet"
            blob = bytearray(det_path.read_bytes())
            if len(blob) > 100:
                blob[50] ^= 0xFF
                bad = fixture_root / "detections_corrupt.parquet"
                bad.write_bytes(bytes(blob))
                try:
                    read_contract_parquet(bad, specs["detections"], contain_root=fixture_root)
                    result.err("corrupt parquet was accepted", integrity=True)
                except Exception:
                    pass
            result.extras["synthetic_roundtrip"] = True

        if args.migration_smoke:
            assert fixture_root is not None
            import pyarrow as pa

            from football_analytics.data.compiler import compile_arrow_schema

            rid = generate_run_id()
            v0 = reg.load_contract("detections", 0)
            schema = compile_arrow_schema(v0)
            rows = [
                {
                    "run_id": rid,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 0,
                    "class_name": "player",
                    "confidence": 0.91,
                    "bbox_x": 10.0,
                    "bbox_y": 20.0,
                    "bbox_width": 30.0,
                    "bbox_height": 40.0,
                    "model_id": "legacy_det",
                }
            ]
            table = pa.Table.from_pylist(rows, schema=schema)
            src = fixture_root / "detections_v0.parquet"
            dst = fixture_root / "detections_v1_migrated.parquet"
            receipt = fixture_root / "detections_migration_receipt.json"
            write_contract_parquet(table, src, v0, contain_root=fixture_root)
            before = sha256_file(src)
            migrate_parquet(
                src,
                dst,
                registry=reg,
                contract="detections",
                from_version=0,
                to_version=1,
                receipt_path=receipt,
                contain_root=fixture_root,
            )
            after = sha256_file(src)
            if before != after:
                result.err("migration changed source file", integrity=True)
            v1 = reg.load_contract("detections", 1)
            migrated = read_contract_parquet(dst, v1, contain_root=fixture_root)
            vr = validate_table(migrated, v1)
            if vr.status == "FAIL":
                result.err(f"migrated invalid: {vr.errors[:3]}", integrity=True)
            if not receipt.is_file():
                result.err("migration receipt missing", integrity=True)
            result.extras["migration_smoke"] = True
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
    p = argparse.ArgumentParser(description="Stage 2C data contract validator")
    p.add_argument("--registry", required=True)
    p.add_argument("--contract")
    p.add_argument("--version", type=int)
    p.add_argument("--synthetic-roundtrip", action="store_true")
    p.add_argument("--migration-smoke", action="store_true")
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
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        for w in result.warnings:
            print(f"WARNING: {w}")
        for e in result.errors:
            print(f"ERROR: {e}")
        print(f"status={result.status} exit_code={result.exit_code}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
