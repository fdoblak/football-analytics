#!/usr/bin/env python3
"""Validate Stage 2B runtime foundation contracts."""

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

FOUNDATION_ROOT = Path("/home/fdoblak/workspace/foundation_checks")


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
    from football_analytics.core.config import (
        config_fingerprint,
        deep_merge,
        load_resolved_config,
        resolved_config_as_dict,
    )
    from football_analytics.core.environment import build_environment_record
    from football_analytics.core.hashing import sha256_bytes
    from football_analytics.core.records import write_json_record
    from football_analytics.core.redaction import REDACTED, redact_value, sanitize_remote_url
    from football_analytics.core.run_context import initialize_run_context
    from football_analytics.core.run_id import generate_run_id, parse_run_id, validate_run_id
    from football_analytics.core.structured_logging import configure_logger, log_event

    result = Result()
    config_path = Path(args.config)
    if not config_path.is_file():
        result.err(f"config missing: {config_path}")
        return result.finalize(strict=args.strict)

    try:
        cfg = load_resolved_config(defaults_path=config_path)
        plain = resolved_config_as_dict(cfg)
        if plain.get("schema_version") != 1:
            result.err("schema_version != 1", integrity=True)
        merged = deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
        if merged != {"a": {"b": 1, "c": 2}}:
            result.err("deep_merge failed", integrity=True)
        fp = config_fingerprint(cfg)
        if len(fp["digest"]) != 64:
            result.err("fingerprint digest invalid", integrity=True)
        result.extras["config_fingerprint"] = fp["digest"]
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {type(exc).__name__}", integrity=True)
        return result.finalize(strict=args.strict)

    # Run ID
    try:
        rid = generate_run_id()
        validate_run_id(rid)
        parsed = parse_run_id(rid)
        if parsed.value != rid:
            result.err("run_id roundtrip failed", integrity=True)
        result.extras["sample_run_id"] = rid
    except Exception as exc:  # noqa: BLE001
        result.err(f"run_id failed: {type(exc).__name__}", integrity=True)

    # Known SHA vectors
    if sha256_bytes(b"") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        result.err("SHA256 empty vector mismatch", integrity=True)
    if sha256_bytes(b"abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad":
        result.err("SHA256 abc vector mismatch", integrity=True)

    # Redaction / URL sanitize
    red = redact_value({"api_key": "secret", "ok": 1})
    if red["api_key"] != REDACTED or red["ok"] != 1:
        result.err("redaction failed", integrity=True)
    sanitized = sanitize_remote_url("https://user:token@github.com/org/repo.git")
    if "token" in sanitized or "user:" in sanitized:
        result.err("remote sanitization failed", integrity=True)

    # Atomic write + logging in temp
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        try:
            write_json_record(tdir / "rec.json", {"hello": "world"}, contain_root=tdir)
            logger = configure_logger(
                level="INFO",
                console=False,
                jsonl_path=tdir / "events.jsonl",
                contain_root=tdir,
                run_id="validator",
            )
            log_event(logger, "INFO", "validator ping", event="validator_ping", run_id="validator")
            line = (tdir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
            payload = json.loads(line)
            for key in (
                "schema_version",
                "timestamp_utc",
                "level",
                "logger",
                "event",
                "message",
                "run_id",
                "stage",
                "context",
                "exception",
            ):
                if key not in payload:
                    result.err(f"jsonl missing {key}", integrity=True)
        except Exception as exc:  # noqa: BLE001
            result.err(f"temp write/log failed: {type(exc).__name__}", integrity=True)

    # Environment record
    try:
        from football_analytics import __version__

        env = build_environment_record(
            project_version=__version__,
            config_fingerprint=fp,
            repo_root=REPO_ROOT,
        )
        if env["gpu_validation"]["torch_imported"] or env["gpu_validation"]["cuda_initialized"]:
            result.err("GPU context must not be started", integrity=True)
        if "password" in json.dumps(env).lower() and "token=" in json.dumps(env).lower():
            result.err("environment record may contain secrets", integrity=True)
        result.extras["git_commit"] = env["git"]["commit"]
    except Exception as exc:  # noqa: BLE001
        result.err(f"environment record failed: {type(exc).__name__}", integrity=True)

    # Synthetic run
    if args.synthetic_run:
        FOUNDATION_ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        synth_root = FOUNDATION_ROOT / f"synthetic_{stamp}"
        synth_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        result.extras["synthetic_root"] = str(synth_root)
        try:
            init = initialize_run_context(
                runs_root=synth_root,
                defaults_path=config_path,
                overrides={"logging": {"level": "WARNING"}},
                repo_root=REPO_ROOT,
                console_logging=False,
            )
            for p in (
                init.resolved_config_path,
                init.environment_path,
                init.run_context_path,
                init.log_path,
            ):
                if not p.is_file():
                    result.err(f"missing artifact: {p.name}", integrity=True)
            # fingerprint consistency
            from football_analytics.core.config import config_fingerprint as cfp
            from football_analytics.core.config import load_resolved_config as lrc

            again = lrc(
                defaults_path=config_path,
                overrides={"logging": {"level": "WARNING"}},
            )
            if cfp(again)["digest"] != init.config_fingerprint["digest"]:
                result.err("fingerprint inconsistency", integrity=True)
            # overwrite reject
            try:
                initialize_run_context(
                    runs_root=synth_root,
                    run_id=init.run_id,
                    defaults_path=config_path,
                    repo_root=REPO_ROOT,
                )
                result.err("overwrite of existing run_id was allowed", integrity=True)
            except Exception:
                pass
            result.extras["synthetic_run_id"] = init.run_id
        except Exception as exc:  # noqa: BLE001
            result.err(f"synthetic run failed: {type(exc).__name__}", integrity=True)
        finally:
            if synth_root.exists():
                shutil.rmtree(synth_root, ignore_errors=False)
            if synth_root.exists():
                result.err("synthetic cleanup incomplete", integrity=True)
            else:
                result.extras["synthetic_cleaned"] = True

    return result.finalize(strict=bool(args.strict))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2B runtime foundation validator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--synthetic-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

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
