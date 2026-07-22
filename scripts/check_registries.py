#!/usr/bin/env python3
"""Validate model/dataset registries and external_repos.lock.yaml (read-only)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
FULL_SHA_RE = re.compile(r"^[a-fA-F0-9]{40}$")
SECRET_URL_RE = re.compile(
    r"(token=|access_token=|api_key=|password=|secret=|Bearer%20|ghp_|github_pat_)",
    re.I,
)
CREDENTIAL_KEYS = {
    "credentials",
    "password",
    "token",
    "secret",
    "api_key",
    "access_key",
    "private_key",
}

MODEL_STATUS = {"planned", "missing", "available", "verified", "blocked", "deprecated"}
DATASET_STATUS = {
    "planned",
    "not_downloaded",
    "available",
    "verified",
    "blocked_access",
    "deprecated",
}
ACCESS_LEVELS = {
    "public",
    "registration_required",
    "nda_required",
    "restricted",
    "unknown",
}
LICENSE_STATUS = {"approved", "review_required", "restricted", "unknown", "prohibited"}

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3


@dataclass
class Result:
    status: str = "PASS"
    exit_code: int = EXIT_PASS
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def err(self, msg: str, integrity: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self) -> Result:
        if self.exit_code == EXIT_INTEGRITY:
            self.status = "FAIL"
        elif self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FAIL
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "exit_code": self.exit_code,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "extras": self.extras,
        }


def load_yaml(path: Path) -> tuple[Any | None, str | None]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_git(path: Path, *args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"JSON parent missing: {parent}")
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    fd, tmp = tempfile.mkstemp(prefix=".registry_validation_", dir=str(parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def validate_model_registry(
    data: dict[str, Any],
    result: Result,
    *,
    verify_files: bool,
    lock: dict[str, Any] | None,
    schema_path: Path | None,
) -> None:
    if data.get("schema_version") != 1:
        result.err("model_registry schema_version must be 1", integrity=True)
        return
    models = data.get("models")
    if not isinstance(models, list):
        result.err("model_registry.models must be a list", integrity=True)
        return
    if schema_path and schema_path.is_file() and jsonschema is not None:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(data, schema)
        except Exception as exc:  # noqa: BLE001
            result.err(f"model_registry schema validation failed: {exc}")

    ids = set()
    for item in models:
        if not isinstance(item, dict):
            result.err("model entry must be mapping")
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            result.err("model id missing")
            continue
        if mid in ids:
            result.err(f"duplicate model id: {mid}", integrity=True)
        ids.add(mid)
        status = item.get("status")
        if status not in MODEL_STATUS:
            result.err(f"{mid}: invalid status {status!r}")
        if item.get("license_status") not in LICENSE_STATUS:
            result.err(f"{mid}: invalid license_status")
        url = item.get("source_url")
        if isinstance(url, str) and SECRET_URL_RE.search(url):
            result.err(f"{mid}: source_url appears to contain a secret", integrity=True)
        for key in item:
            if key.lower() in CREDENTIAL_KEYS:
                result.err(f"{mid}: credential-like field forbidden: {key}", integrity=True)

        path = item.get("file_path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if status in {"available", "verified"}:
            if not isinstance(path, str) or not os.path.isabs(path):
                result.err(f"{mid}: available/verified requires absolute file_path")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                result.err(f"{mid}: available/verified requires positive size_bytes")
            if not isinstance(digest, str) or not SHA256_RE.match(digest):
                result.err(f"{mid}: available/verified requires full 64-char sha256")
            if isinstance(digest, str) and len(digest) != 64:
                result.err(f"{mid}: short sha256 not allowed", integrity=True)
            if verify_files and isinstance(path, str):
                p = Path(path)
                if not p.is_file():
                    result.err(f"{mid}: file missing: {path}", integrity=True)
                else:
                    actual_size = p.stat().st_size
                    if actual_size != size:
                        result.err(
                            f"{mid}: size mismatch actual={actual_size} registry={size}",
                            integrity=True,
                        )
                    actual_hash = sha256_file(p)
                    if actual_hash.lower() != str(digest).lower():
                        result.err(f"{mid}: sha256 mismatch", integrity=True)
        else:
            if path not in (None, "") and status in {"planned", "missing"}:
                # path may be planned absolute future location; existence not required
                pass

        if lock and item.get("source_repo"):
            src = item["source_repo"]
            tp = (lock.get("third_party_repositories") or {}).get(src)
            sn = (lock.get("repositories") or {}).get(src)
            entry = tp or sn
            if entry is None:
                result.warn(f"{mid}: source_repo {src!r} not found in external lock")
            else:
                commit = item.get("source_commit")
                if commit and commit != entry.get("commit"):
                    result.err(
                        f"{mid}: source_commit does not match lock for {src}",
                        integrity=True,
                    )

        if item.get("license_status") == "review_required":
            result.warn(f"{mid}: license_status=review_required")


def validate_dataset_registry(
    data: dict[str, Any],
    result: Result,
    *,
    schema_path: Path | None,
) -> None:
    if data.get("schema_version") != 1:
        result.err("dataset_registry schema_version must be 1", integrity=True)
        return
    datasets = data.get("datasets")
    if not isinstance(datasets, list):
        result.err("dataset_registry.datasets must be a list", integrity=True)
        return
    if schema_path and schema_path.is_file() and jsonschema is not None:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(data, schema)
        except Exception as exc:  # noqa: BLE001
            # credentials forbid in schema may be awkward; keep manual checks primary
            if "credentials" not in str(exc).lower():
                result.err(f"dataset_registry schema validation failed: {exc}")

    ids = set()
    for item in datasets:
        if not isinstance(item, dict):
            result.err("dataset entry must be mapping")
            continue
        did = item.get("id")
        if not isinstance(did, str) or not did:
            result.err("dataset id missing")
            continue
        if did in ids:
            result.err(f"duplicate dataset id: {did}", integrity=True)
        ids.add(did)
        status = item.get("status")
        if status not in DATASET_STATUS:
            result.err(f"{did}: invalid status {status!r}")
        access = item.get("access_level")
        if access not in ACCESS_LEVELS:
            result.err(f"{did}: invalid access_level {access!r}")
        if item.get("license_status") not in LICENSE_STATUS:
            result.err(f"{did}: invalid license_status")
        for key in item:
            if key.lower() in CREDENTIAL_KEYS:
                result.err(f"{did}: credential-like field forbidden: {key}", integrity=True)
        url = item.get("source_url")
        if isinstance(url, str) and SECRET_URL_RE.search(url):
            result.err(f"{did}: source_url appears to contain a secret", integrity=True)

        path = item.get("local_path")
        checksum = item.get("checksum")
        if status in {"planned", "not_downloaded"}:
            if path not in (None, ""):
                result.err(f"{did}: planned/not_downloaded must have local_path=null")
            if checksum not in (None, ""):
                result.err(f"{did}: planned/not_downloaded must have checksum=null")
            # must not claim available
            if status == "available":
                result.err(f"{did}: inconsistent status")
        if status in {"available", "verified"}:
            if not isinstance(path, str) or not os.path.isabs(path):
                result.err(f"{did}: available/verified requires absolute local_path")
            elif not Path(path).exists():
                result.err(f"{did}: available/verified path missing: {path}", integrity=True)
            if status == "verified" and not checksum:
                result.err(f"{did}: verified requires checksum")

        if item.get("license_status") == "review_required":
            result.warn(f"{did}: license_status=review_required")
        if access == "unknown":
            result.warn(f"{did}: access_level=unknown")


def validate_external_lock(
    lock: dict[str, Any],
    result: Result,
    *,
    verify_repos: bool,
) -> None:
    repos = lock.get("repositories") or {}
    third = lock.get("third_party_repositories") or {}
    if not isinstance(repos, dict) or not isinstance(third, dict):
        result.err("lock repositories groups must be mappings", integrity=True)
        return
    if len(repos) != 19:
        result.err(f"expected 19 SoccerNet repos, found {len(repos)}", integrity=True)
    if len(third) != 3:
        result.err(f"expected 3 third-party repos, found {len(third)}", integrity=True)
    all_ids = list(repos.keys()) + list(third.keys())
    if len(all_ids) != len(set(all_ids)):
        result.err("duplicate repo ids across lock groups", integrity=True)

    paths_seen: dict[str, str] = {}
    remotes_seen: dict[str, str] = {}
    matched = 0
    for _group_name, group in (("repositories", repos), ("third_party_repositories", third)):
        for rid, meta in group.items():
            if not isinstance(meta, dict):
                result.err(f"{rid}: invalid metadata")
                continue
            commit = meta.get("commit")
            if not isinstance(commit, str) or not FULL_SHA_RE.match(commit):
                result.err(f"{rid}: commit must be full 40-char SHA", integrity=True)
                continue
            path = meta.get("path")
            if not isinstance(path, str) or not path:
                result.err(f"{rid}: path missing")
                continue
            if path in paths_seen:
                result.err(f"duplicate path {path} ({paths_seen[path]} vs {rid})", integrity=True)
            paths_seen[path] = rid
            remote = meta.get("remote")
            if isinstance(remote, str):
                if SECRET_URL_RE.search(remote):
                    result.err(f"{rid}: remote contains secret-like token", integrity=True)
                if remote in remotes_seen and remotes_seen[remote] != rid:
                    result.warn(f"shared remote URL between {remotes_seen[remote]} and {rid}")
                remotes_seen[remote] = rid
            p = Path(path)
            if verify_repos:
                if not p.is_dir() or not (p / ".git").exists():
                    result.err(f"{rid}: git repo missing at {path}", integrity=True)
                    continue
                code, head = run_git(p, "rev-parse", "HEAD")
                if code != 0 or head != commit:
                    result.err(
                        f"{rid}: HEAD/lock mismatch head={head!r} lock={commit!r}",
                        integrity=True,
                    )
                else:
                    matched += 1
                code, dirty = run_git(p, "status", "--porcelain=v1")
                if code == 0 and dirty:
                    result.err(f"{rid}: working tree dirty", integrity=True)
                if meta.get("dirty") is True:
                    result.warn(f"{rid}: lock marks dirty=true")
    result.extras["external_repos"] = {
        "soccernet_count": len(repos),
        "third_party_count": len(third),
        "total": len(repos) + len(third),
        "verified_head_matches": matched if verify_repos else None,
    }


def run_checks(args: argparse.Namespace) -> Result:
    result = Result()
    model_path = Path(args.model_registry)
    dataset_path = Path(args.dataset_registry)
    lock_path = Path(args.external_lock)
    for required in (model_path, dataset_path, lock_path):
        if not required.is_file():
            result.err(f"missing file: {required}")
            result.exit_code = EXIT_CONFIG
            return result.finalize()

    model_data, err = load_yaml(model_path)
    if err or not isinstance(model_data, dict):
        result.err(f"model_registry load failed: {err or 'not a mapping'}")
        result.exit_code = EXIT_CONFIG
        return result.finalize()
    dataset_data, err = load_yaml(dataset_path)
    if err or not isinstance(dataset_data, dict):
        result.err(f"dataset_registry load failed: {err or 'not a mapping'}")
        result.exit_code = EXIT_CONFIG
        return result.finalize()
    lock_data, err = load_yaml(lock_path)
    if err or not isinstance(lock_data, dict):
        result.err(f"external lock load failed: {err or 'not a mapping'}")
        result.exit_code = EXIT_CONFIG
        return result.finalize()

    schema_root = Path(__file__).resolve().parents[1] / "schemas" / "registries"
    validate_model_registry(
        model_data,
        result,
        verify_files=bool(args.verify_files),
        lock=lock_data,
        schema_path=schema_root / "model_registry.schema.json",
    )
    validate_dataset_registry(
        dataset_data,
        result,
        schema_path=schema_root / "dataset_registry.schema.json",
    )
    validate_external_lock(lock_data, result, verify_repos=bool(args.verify_repos))

    result.extras["counts"] = {
        "models": len(model_data.get("models") or []),
        "datasets": len(dataset_data.get("datasets") or []),
    }
    if args.strict and result.warnings and not result.errors:
        for warning in list(result.warnings):
            result.err(f"strict: {warning}")
    return result.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate registries and external lock")
    p.add_argument("--model-registry", required=True)
    p.add_argument("--dataset-registry", required=True)
    p.add_argument("--external-lock", required=True)
    p.add_argument("--json-out", default=None)
    p.add_argument("--verify-files", action="store_true")
    p.add_argument("--verify-repos", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_CONFIG
        return code or EXIT_CONFIG
    try:
        result = run_checks(args)
    except Exception as exc:  # noqa: BLE001
        result = Result()
        result.err(f"unhandled exception: {exc}")
        result.extras["traceback"] = traceback.format_exc(limit=20)
        result.exit_code = EXIT_FAIL
        result.finalize()
    payload = result.to_dict()
    if args.json_out:
        try:
            write_json_atomic(Path(args.json_out), payload)
        except Exception as exc:  # noqa: BLE001
            result.err(f"json-out failed: {exc}")
            result.exit_code = EXIT_CONFIG
            result.finalize()
            payload = result.to_dict()
    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        for error in result.errors:
            print(f"ERROR: {error}")
    return int(result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
