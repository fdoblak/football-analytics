#!/usr/bin/env python3
"""Storage contract validator for football-analytics.

Default mode is read-only. Write/read/hash/cleanup probe runs only with --probe.
No package installs, mounts, chmod/chown, or recursive deletes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - ai-dev has PyYAML
    yaml = None


SUPPORTED_BACKENDS = frozenset({"wsl_local", "windows_drvfs"})
REQUIRED_PATH_KEYS = (
    "raw_matches",
    "test_clips",
    "datasets",
    "results",
    "rendered_outputs",
    "reports",
    "model_archive",
    "experiments_archive",
    "backups",
)
FORBIDDEN_ACTIVE_ROOTS = frozenset(
    {
        "/",
        "/home",
        "/home/fdoblak",
        "/home/fdoblak/projects",
        "/home/fdoblak/projects/football-analytics",
    }
)
ENV_VAR_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_SECURITY = 3


@dataclass
class ValidationResult:
    status: str = "PASS"
    exit_code: int = EXIT_PASS
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config_path: str = ""
    active_backend: str = ""
    active_root: dict[str, Any] = field(default_factory=dict)
    filesystem: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def finalize(self, security_failure: bool = False) -> ValidationResult:
        if security_failure or self.exit_code == EXIT_SECURITY:
            self.status = "FAIL"
            self.exit_code = EXIT_SECURITY
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
        payload = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_path": self.config_path,
            "status": self.status,
            "exit_code": self.exit_code,
            "active_backend": self.active_backend,
            "active_root": self.active_root,
            "filesystem": self.filesystem,
            "thresholds": self.thresholds,
            "paths": self.paths,
            "probe": self.probe,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload


def load_config(config_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if yaml is None:
        return None, "PyYAML is not available in this interpreter"
    if not config_path.is_file():
        return None, f"Config file not found: {config_path}"
    try:
        raw = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except Exception as exc:  # noqa: BLE001 - surface parse errors
        return None, f"YAML parse error: {exc}"
    if not isinstance(data, dict):
        return None, "Config root must be a mapping"
    return data, None


def _is_absolute_safe(path_str: str) -> bool:
    if not isinstance(path_str, str) or not path_str:
        return False
    if path_str.startswith("~") or "~/" in path_str or path_str == "~":
        return False
    if ENV_VAR_RE.search(path_str):
        return False
    return os.path.isabs(path_str)


def _normalize_configured(path_str: str) -> Path:
    # Pure lexical normalization; do not resolve symlinks yet.
    return Path(os.path.normpath(path_str))


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _path_mode(path: Path) -> str | None:
    try:
        return oct(path.stat().st_mode & 0o777)
    except OSError:
        return None


def _owner_ids(path: Path) -> tuple[int | None, int | None]:
    try:
        st = path.stat()
        return st.st_uid, st.st_gid
    except OSError:
        return None, None


def _filesystem_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": None,
        "type": None,
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
        "free_percent": None,
    }
    try:
        usage = shutil.disk_usage(path)
        info["total_bytes"] = int(usage.total)
        info["used_bytes"] = int(usage.used)
        info["free_bytes"] = int(usage.free)
        if usage.total:
            info["free_percent"] = round(100.0 * usage.free / usage.total, 4)
    except OSError as exc:
        info["disk_usage_error"] = str(exc)

    try:
        proc = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE", "-T", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split()
            if len(parts) >= 1:
                info["device"] = parts[0]
            if len(parts) >= 2:
                info["type"] = parts[1]
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["findmnt_error"] = str(exc)
    return info


def validate_config_structure(
    data: dict[str, Any], result: ValidationResult
) -> dict[str, Any] | None:
    storage = data.get("storage")
    if not isinstance(storage, dict):
        result.add_error("Missing or invalid 'storage' mapping")
        result.exit_code = EXIT_CONFIG
        return None

    validation = data.get("storage_validation")
    if not isinstance(validation, dict):
        result.add_error("Missing or invalid 'storage_validation' mapping")
        result.exit_code = EXIT_CONFIG
        return None

    backend = storage.get("active_backend")
    if backend not in SUPPORTED_BACKENDS:
        result.add_error(
            f"Unsupported active_backend={backend!r}; " f"supported={sorted(SUPPORTED_BACKENDS)}"
        )
        result.exit_code = EXIT_CONFIG
        return None
    result.active_backend = str(backend)

    for key in ("minimum_free_bytes", "warning_free_bytes"):
        value = validation.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            result.add_error(f"storage_validation.{key} must be a positive integer")
            result.exit_code = EXIT_CONFIG
            return None

    minimum = validation["minimum_free_bytes"]
    warning = validation["warning_free_bytes"]
    if minimum >= warning:
        result.add_error(
            "storage_validation.minimum_free_bytes must be strictly less than " "warning_free_bytes"
        )
        result.exit_code = EXIT_CONFIG
        return None

    result.thresholds = {
        "minimum_free_bytes": minimum,
        "warning_free_bytes": warning,
        "require_absolute_paths": bool(validation.get("require_absolute_paths", True)),
        "require_paths_under_active_root": bool(
            validation.get("require_paths_under_active_root", True)
        ),
        "reject_symlink_escape": bool(validation.get("reject_symlink_escape", True)),
    }

    archive_status = storage.get("planned_archive_status")
    if not isinstance(archive_status, str) or not archive_status:
        result.add_error("storage.planned_archive_status must be a non-empty string")
        result.exit_code = EXIT_CONFIG
        return None

    for key in REQUIRED_PATH_KEYS:
        if key not in storage:
            result.add_error(f"Missing required storage path key: {key}")
            result.exit_code = EXIT_CONFIG
            return None

    if "active_root" not in storage:
        result.add_error("Missing storage.active_root")
        result.exit_code = EXIT_CONFIG
        return None

    result.extras["planned_archive"] = {
        "root": storage.get("planned_archive_root"),
        "status": archive_status,
        "validated_as_active": False,
        "note": "Planned archive is not subject to active-root existence checks",
    }
    return {"storage": storage, "storage_validation": validation}


def validate_active_root(storage: dict[str, Any], result: ValidationResult) -> Path | None:
    configured = storage.get("active_root")
    root_info: dict[str, Any] = {
        "configured": configured,
        "resolved": None,
        "exists": False,
        "is_directory": False,
        "is_symlink": False,
        "owner_uid": None,
        "owner_gid": None,
        "mode": None,
        "readable": False,
        "writable": False,
        "traversable": False,
    }
    result.active_root = root_info

    if not isinstance(configured, str) or not configured:
        result.add_error("storage.active_root must be a non-empty string")
        return None
    if not _is_absolute_safe(configured):
        result.add_error(
            "storage.active_root must be an absolute path without ~ or environment variables"
        )
        return None

    configured_norm = _normalize_configured(configured)
    if str(configured_norm) in FORBIDDEN_ACTIVE_ROOTS:
        result.add_error(f"storage.active_root is a forbidden broad path: {configured_norm}")
        return None
    if ".." in Path(configured).parts and (
        "/../" in configured or configured.endswith("/..") or configured.startswith("../")
    ):
        # normpath may collapse; still reject explicit traversal intent in config text
        result.add_error("storage.active_root must not contain '..' traversal")
        return None

    root_path = Path(configured)
    root_info["is_symlink"] = root_path.is_symlink()
    if root_info["is_symlink"]:
        try:
            target = root_path.resolve(strict=False)
            root_info["symlink_target"] = str(target)
            result.add_warning(f"active_root is a symlink -> {target}; evaluating resolved target")
        except OSError as exc:
            result.add_error(f"Unable to resolve active_root symlink: {exc}")
            return None

    try:
        resolved = root_path.resolve(strict=False)
    except OSError as exc:
        result.add_error(f"Unable to resolve active_root: {exc}")
        return None

    root_info["resolved"] = str(resolved)
    if str(resolved) in FORBIDDEN_ACTIVE_ROOTS:
        result.add_error(f"Resolved active_root is forbidden: {resolved}")
        return None

    if not resolved.exists():
        result.add_error(f"active_root does not exist: {resolved}")
        return None
    root_info["exists"] = True
    if not resolved.is_dir():
        result.add_error(f"active_root is not a directory: {resolved}")
        return None
    root_info["is_directory"] = True

    uid, gid = _owner_ids(resolved)
    root_info["owner_uid"] = uid
    root_info["owner_gid"] = gid
    root_info["mode"] = _path_mode(resolved)
    root_info["readable"] = os.access(resolved, os.R_OK)
    root_info["writable"] = os.access(resolved, os.W_OK)
    root_info["traversable"] = os.access(resolved, os.X_OK)

    try:
        mode = resolved.stat().st_mode
        if mode & stat.S_IWOTH:
            result.add_warning(f"active_root is world-writable (mode={root_info['mode']})")
    except OSError:
        pass

    if not root_info["readable"] or not root_info["traversable"]:
        result.add_error("active_root is not readable/traversable")
    if not root_info["writable"]:
        result.add_error("active_root is not writable")

    # Optional consistency: ssd_root should match active_root when present
    ssd_root = storage.get("ssd_root")
    if (
        isinstance(ssd_root, str)
        and ssd_root
        and _normalize_configured(ssd_root) != configured_norm
    ):
        result.add_warning(f"storage.ssd_root ({ssd_root}) differs from active_root ({configured})")

    return resolved


def validate_required_paths(
    storage: dict[str, Any],
    active_root: Path,
    result: ValidationResult,
    reject_symlink_escape: bool,
) -> None:
    seen_canonical: dict[str, str] = {}
    for key in REQUIRED_PATH_KEYS:
        entry: dict[str, Any] = {
            "configured": storage.get(key),
            "status": "FAIL",
            "exists": False,
            "is_directory": False,
            "is_symlink": False,
            "resolved": None,
            "under_active_root": False,
            "owner_uid": None,
            "owner_gid": None,
            "mode": None,
        }
        result.paths[key] = entry
        configured = storage.get(key)
        if not isinstance(configured, str) or not configured:
            result.add_error(f"{key}: path must be a non-empty string")
            continue
        if not _is_absolute_safe(configured):
            result.add_error(
                f"{key}: must be absolute and must not contain ~ or environment variables"
            )
            continue

        configured_path = Path(configured)
        if ".." in configured_path.parts:
            result.add_error(f"{key}: path contains '..' traversal segment")
            continue

        lexically = _normalize_configured(configured)
        if not _is_under(lexically, _normalize_configured(str(active_root))):
            # Compare against configured active_root string as well as resolved
            configured_root = _normalize_configured(str(result.active_root["configured"]))
            if not _is_under(lexically, configured_root):
                result.add_error(f"{key}: path escapes active_root lexically ({lexically})")
                continue

        entry["is_symlink"] = configured_path.is_symlink()
        try:
            resolved = configured_path.resolve(strict=False)
        except OSError as exc:
            result.add_error(f"{key}: resolve failed: {exc}")
            continue
        entry["resolved"] = str(resolved)

        if not _is_under(resolved, active_root):
            msg = f"{key}: resolved path escapes active_root ({resolved})"
            result.add_error(msg)
            if entry["is_symlink"] and reject_symlink_escape:
                result.exit_code = EXIT_SECURITY
            continue
        entry["under_active_root"] = True

        canon = str(resolved)
        if canon in seen_canonical:
            result.add_error(
                f"{key}: duplicate canonical path with {seen_canonical[canon]} ({canon})"
            )
            continue
        seen_canonical[canon] = key

        if configured_path.is_symlink() and not configured_path.exists():
            result.add_error(f"{key}: broken symlink")
            continue
        if not resolved.exists():
            result.add_error(f"{key}: path does not exist")
            continue
        entry["exists"] = True
        if not resolved.is_dir():
            result.add_error(f"{key}: path is not a directory")
            continue
        entry["is_directory"] = True
        uid, gid = _owner_ids(resolved)
        entry["owner_uid"] = uid
        entry["owner_gid"] = gid
        entry["mode"] = _path_mode(resolved)
        entry["status"] = "PASS"


def apply_capacity_gate(result: ValidationResult) -> None:
    free_bytes = result.filesystem.get("free_bytes")
    minimum = result.thresholds.get("minimum_free_bytes")
    warning = result.thresholds.get("warning_free_bytes")
    if free_bytes is None or minimum is None or warning is None:
        result.add_error("Unable to evaluate capacity thresholds")
        return
    if free_bytes < minimum:
        result.add_error(f"free_bytes {free_bytes} is below minimum_free_bytes {minimum}")
        result.filesystem["capacity_status"] = "FAIL"
    elif free_bytes < warning:
        result.add_warning(f"free_bytes {free_bytes} is below warning_free_bytes {warning}")
        result.filesystem["capacity_status"] = "WARNING"
    else:
        result.filesystem["capacity_status"] = "PASS"


def run_probe(active_root: Path, result: ValidationResult) -> bool:
    """Returns True if security/cleanup failure should force exit code 3."""
    probe_info: dict[str, Any] = {
        "requested": True,
        "executed": False,
        "passed": False,
        "cleanup_verified": False,
        "path": None,
        "sha256": None,
        "size_bytes": None,
    }
    result.probe = probe_info
    security_fail = False
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    name = f".storage_probe_{ts}_{secrets.token_hex(8)}"
    probe_path = active_root / name
    probe_info["path"] = str(probe_path)

    if not _is_under(probe_path.resolve(strict=False), active_root.resolve(strict=False)):
        result.add_error("Refusing probe path outside active_root")
        security_fail = True
        return security_fail

    payload = secrets.token_bytes(2048) + f"\ncheck_storage_probe:{ts}\n".encode()
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(str(probe_path), flags, 0o644)
        created = True
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        probe_info["executed"] = True
        size1 = probe_path.stat().st_size
        digest1 = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        data2 = probe_path.read_bytes()
        size2 = len(data2)
        digest2 = hashlib.sha256(data2).hexdigest()
        probe_info["size_bytes"] = size1
        probe_info["sha256"] = digest1
        if size1 != size2 or size1 != len(payload):
            result.add_error("Probe size mismatch")
        if digest1 != digest2:
            result.add_error("Probe SHA-256 mismatch")
        if not result.errors:
            probe_info["passed"] = True
    except FileExistsError:
        result.add_error(f"Probe path already exists (refusing overwrite): {probe_path}")
        security_fail = True
    except OSError as exc:
        result.add_error(f"Probe write/read failed: {exc}")
    finally:
        if created and probe_path.exists():
            try:
                probe_path.unlink()
            except OSError as exc:
                result.add_error(f"Probe cleanup failed: {exc}")
                security_fail = True
        if created:
            if probe_path.exists():
                result.add_error(f"Probe file still present after cleanup: {probe_path}")
                security_fail = True
                probe_info["cleanup_verified"] = False
            else:
                probe_info["cleanup_verified"] = True
                # Ensure no leftover probe names from this run (exact path only)
        elif probe_info["path"] and Path(probe_info["path"]).exists() and not created:
            # Did not create; do not delete pre-existing
            probe_info["cleanup_verified"] = None

    if probe_info.get("passed") and not probe_info.get("cleanup_verified"):
        probe_info["passed"] = False
        security_fail = True
    return security_fail


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(f"JSON output parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"JSON output parent is not a directory: {parent}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing JSON report: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=".storage_validation_", dir=str(parent))
    tmp_path = Path(tmp_name)
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


def run_validation(
    config_path: Path,
    *,
    do_probe: bool = False,
    strict: bool = False,
) -> ValidationResult:
    result = ValidationResult(config_path=str(config_path))
    result.probe = {
        "requested": do_probe,
        "executed": False,
        "passed": None,
        "cleanup_verified": None,
    }

    data, err = load_config(config_path)
    if err:
        result.add_error(err)
        result.exit_code = EXIT_CONFIG
        return result.finalize()

    assert data is not None
    parsed = validate_config_structure(data, result)
    if parsed is None:
        return result.finalize()

    storage = parsed["storage"]
    validation = parsed["storage_validation"]
    active_root = validate_active_root(storage, result)
    if active_root is None:
        return result.finalize()

    result.filesystem = _filesystem_info(active_root)
    apply_capacity_gate(result)
    validate_required_paths(
        storage,
        active_root,
        result,
        reject_symlink_escape=bool(validation.get("reject_symlink_escape", True)),
    )

    security_failure = False
    if do_probe:
        if result.errors and strict:
            result.add_warning("Skipping probe because prior validation errors exist (--strict)")
        else:
            security_failure = run_probe(active_root, result)

    if strict and result.warnings and not result.errors:
        # strict turns warnings into failures for gate use
        for warning in list(result.warnings):
            result.add_error(f"strict: {warning}")

    return result.finalize(security_failure=security_failure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate football-analytics storage contract (read-only by default)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to paths.yaml",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional atomic JSON report output path (must not already exist)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Opt-in write/read/hash/cleanup probe under active_root",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary (JSON/errors still apply)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_CONFIG
        return code if code else EXIT_CONFIG

    config_path = Path(args.config)
    try:
        result = run_validation(config_path, do_probe=bool(args.probe), strict=bool(args.strict))
    except Exception as exc:  # noqa: BLE001
        result = ValidationResult(config_path=str(config_path))
        result.add_error(f"Unhandled validator exception: {exc}")
        result.extras["traceback"] = traceback.format_exc(limit=20)
        result.exit_code = EXIT_FAIL
        result.finalize()

    payload = result.to_dict()
    if args.json_out:
        try:
            write_json_atomic(Path(args.json_out), payload)
        except FileExistsError as exc:
            result.add_error(str(exc))
            result.exit_code = EXIT_CONFIG
            result.finalize()
            payload = result.to_dict()
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            result.add_error(f"JSON output failed: {exc}")
            result.exit_code = EXIT_CONFIG
            result.finalize()
            payload = result.to_dict()

    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        print(f"active_backend={result.active_backend}")
        print(f"active_root={result.active_root.get('resolved')}")
        print(
            "capacity={} free_bytes={}".format(
                result.filesystem.get("capacity_status"),
                result.filesystem.get("free_bytes"),
            )
        )
        if result.probe.get("requested"):
            print(
                "probe_passed={} cleanup_verified={}".format(
                    result.probe.get("passed"),
                    result.probe.get("cleanup_verified"),
                )
            )
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        for error in result.errors:
            print(f"ERROR: {error}")

    return int(result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
