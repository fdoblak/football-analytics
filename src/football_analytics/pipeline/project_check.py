"""Unified project health check runner (Stage 2D).

Shared by ``scripts/check_project.py`` and the CLI ``project check`` command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from football_analytics.core.redaction import redact_text, sanitize_remote_url

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

SUBPROCESS_TIMEOUT_SEC = 120
MAX_CAPTURE_CHARS = 32_000

CheckStatus = Literal["PASS", "WARN", "FAIL", "SKIP"]
Severity = Literal["info", "warn", "error", "integrity"]

PROTECTED_PACKAGE_VERSIONS: dict[str, str] = {
    "torch": "2.11.0+cu128",
    "numpy": "2.2.6",
    "opencv-python": "5.0.0.93",
    "ultralytics": "8.4.91",
    "SoccerNet": "0.1.62",
}

PIPELINE_SCHEMAS = (
    "schemas/pipeline/artifact_ref.schema.json",
    "schemas/pipeline/stage_request.schema.json",
    "schemas/pipeline/stage_result.schema.json",
    "schemas/pipeline/stage_execution_receipt.schema.json",
    "schemas/cache/cache_manifest.schema.json",
)

HOST_ONLY_SKIP_CHECKS = (
    ("host_wsl_gpu", "WSL GPU validation is host-only"),
    ("host_absolute_paths", "/home/fdoblak absolute machine paths are host-only"),
    ("host_soccernet_clones", "Local SoccerNet clones are host-only"),
    ("host_model_weights", "Model weight presence checks are host-only"),
    ("host_storage_backend", "Real storage backend probes are host-only"),
    ("host_git_credential", "Git credential materialization is host-only"),
    ("host_nda_datasets", "NDA dataset presence checks are host-only"),
)


@dataclass
class CheckRecord:
    id: str
    category: str
    status: CheckStatus
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectCheckReport:
    schema_version: int
    profile: str
    mode: str
    started_at_utc: str
    finished_at_utc: str
    duration_ms: int
    overall_status: str
    exit_code: int
    checks: list[CheckRecord]
    summary: dict[str, int]
    environment_classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "mode": self.mode,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_ms": self.duration_ms,
            "overall_status": self.overall_status,
            "exit_code": self.exit_code,
            "checks": [c.to_dict() for c in self.checks],
            "summary": dict(self.summary),
            "environment_classification": self.environment_classification,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_check(check_id: str, category: str, fn: Callable[[], CheckRecord]) -> CheckRecord:
    t0 = time.perf_counter()
    try:
        record = fn()
    except Exception as exc:  # noqa: BLE001 — isolate check failures
        record = CheckRecord(
            id=check_id,
            category=category,
            status="FAIL",
            severity="integrity",
            message=f"check raised {type(exc).__name__}: {redact_text(str(exc))}",
            evidence={},
        )
    record.duration_ms = max(0, int((time.perf_counter() - t0) * 1000))
    if record.id != check_id:
        record.id = check_id
    if not record.category:
        record.category = category
    return record


def _bounded_text(text: str | None) -> str:
    raw = text or ""
    if len(raw) > MAX_CAPTURE_CHARS:
        raw = raw[:MAX_CAPTURE_CHARS] + "\n...[truncated]..."
    return redact_text(raw)


def _run_script(
    root: Path,
    script_rel: str,
    args: list[str],
    *,
    check_id: str,
    category: str,
) -> CheckRecord:
    script = root / script_rel
    if not script.is_file():
        return CheckRecord(
            id=check_id,
            category=category,
            status="FAIL",
            severity="error",
            message=f"script missing: {script_rel}",
        )
    cmd = [sys.executable, str(script), *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
            check=False,
            shell=False,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
        )
    except subprocess.TimeoutExpired:
        return CheckRecord(
            id=check_id,
            category=category,
            status="FAIL",
            severity="integrity",
            message=f"subprocess timed out after {SUBPROCESS_TIMEOUT_SEC}s",
            evidence={"cmd": [script_rel, *args]},
        )
    except OSError as exc:
        return CheckRecord(
            id=check_id,
            category=category,
            status="FAIL",
            severity="error",
            message=f"subprocess failed: {type(exc).__name__}",
        )

    evidence = {
        "returncode": proc.returncode,
        "stdout": _bounded_text(proc.stdout),
        "stderr": _bounded_text(proc.stderr),
    }
    if proc.returncode == 0:
        return CheckRecord(
            id=check_id,
            category=category,
            status="PASS",
            severity="info",
            message="ok",
            evidence=evidence,
        )
    if proc.returncode == 2:
        return CheckRecord(
            id=check_id,
            category=category,
            status="FAIL",
            severity="error",
            message="config/usage failure",
            evidence=evidence,
        )
    return CheckRecord(
        id=check_id,
        category=category,
        status="FAIL",
        severity="integrity" if proc.returncode == 3 else "error",
        message=f"validator exit {proc.returncode}",
        evidence=evidence,
    )


def _check_git(root: Path) -> CheckRecord:
    from football_analytics.core.environment import collect_git_metadata

    meta = collect_git_metadata(root)
    dirty = meta.get("dirty")
    status: CheckStatus = "PASS"
    severity: Severity = "info"
    message = f"branch={meta.get('branch')} dirty={dirty}"
    if dirty is True:
        status = "WARN"
        severity = "warn"
        message = "working tree is dirty"
    elif dirty is None:
        status = "WARN"
        severity = "warn"
        message = "git metadata unavailable"
    remote = meta.get("remote_sanitized")
    evidence = {
        "branch": meta.get("branch"),
        "commit": meta.get("commit"),
        "dirty": dirty,
        "remote_sanitized": remote,
    }
    if isinstance(remote, str) and re.search(r"https?://[^/]*:[^/]*@", remote):
        return CheckRecord(
            id="git_status",
            category="git",
            status="FAIL",
            severity="integrity",
            message="remote URL appears to contain credentials",
            evidence=evidence,
        )
    return CheckRecord(
        id="git_status",
        category="git",
        status=status,
        severity=severity,
        message=message,
        evidence=evidence,
    )


def _check_remote_sanitized(root: Path) -> CheckRecord:
    from football_analytics.core.environment import collect_git_metadata

    meta = collect_git_metadata(root)
    remote_raw = None
    try:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
        if proc.returncode == 0:
            remote_raw = (proc.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        remote_raw = None
    if not remote_raw:
        return CheckRecord(
            id="git_remote_sanitized",
            category="git",
            status="WARN",
            severity="warn",
            message="origin remote unavailable",
        )
    sanitized = sanitize_remote_url(remote_raw)
    if remote_raw != sanitized and ("@" in remote_raw or "token" in remote_raw.lower()):
        return CheckRecord(
            id="git_remote_sanitized",
            category="git",
            status="FAIL",
            severity="integrity",
            message="origin remote embeds credentials",
            evidence={"remote_sanitized": sanitized},
        )
    return CheckRecord(
        id="git_remote_sanitized",
        category="git",
        status="PASS",
        severity="info",
        message="remote URL has no embedded credentials",
        evidence={"remote_sanitized": meta.get("remote_sanitized") or sanitized},
    )


def _check_package_metadata() -> CheckRecord:
    from football_analytics import __version__

    try:
        dist = metadata.version("football-analytics")
    except metadata.PackageNotFoundError:
        dist = None
    return CheckRecord(
        id="package_metadata",
        category="package",
        status="PASS",
        severity="info",
        message=f"version={__version__}",
        evidence={"__version__": __version__, "dist_version": dist},
    )


def _check_protected_packages(*, profile: str) -> CheckRecord:
    if profile == "ci":
        return CheckRecord(
            id="protected_package_versions",
            category="packages",
            status="SKIP",
            severity="info",
            message="protected CUDA/vision pins are host/ai-dev only",
            evidence={"skipped": True},
        )
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected in PROTECTED_PACKAGE_VERSIONS.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = None
        if found != expected:
            mismatches[name] = {"expected": expected, "found": found}
    if mismatches:
        return CheckRecord(
            id="protected_package_versions",
            category="packages",
            status="FAIL",
            severity="integrity",
            message="protected package version mismatch",
            evidence={"mismatches": mismatches},
        )
    return CheckRecord(
        id="protected_package_versions",
        category="packages",
        status="PASS",
        severity="info",
        message="protected package versions match",
        evidence={"checked": list(PROTECTED_PACKAGE_VERSIONS)},
    )


def _check_config_defaults(root: Path) -> CheckRecord:
    path = root / "configs" / "project" / "defaults.yaml"
    if not path.is_file():
        return CheckRecord(
            id="config_defaults",
            category="config",
            status="FAIL",
            severity="error",
            message="defaults.yaml missing",
        )
    try:
        from football_analytics.core.config import load_resolved_config

        load_resolved_config(defaults_path=path)
    except Exception as exc:  # noqa: BLE001
        return CheckRecord(
            id="config_defaults",
            category="config",
            status="FAIL",
            severity="integrity",
            message=f"defaults load failed: {type(exc).__name__}",
        )
    return CheckRecord(
        id="config_defaults",
        category="config",
        status="PASS",
        severity="info",
        message="defaults.yaml loads",
    )


def _check_storage_readonly(root: Path, *, profile: str) -> CheckRecord:
    if profile == "ci":
        return CheckRecord(
            id="storage_paths",
            category="storage",
            status="SKIP",
            severity="info",
            message="real storage backend path checks are host-only",
        )
    return _run_script(
        root,
        "scripts/check_storage.py",
        ["--config", "configs/system/paths.yaml", "--quiet"],
        check_id="storage_paths",
        category="storage",
    )


def _check_cache_policy(root: Path) -> CheckRecord:
    path = root / "configs" / "system" / "cache_policy.yaml"
    try:
        from football_analytics.pipeline.cache import load_cache_policy

        policy = load_cache_policy(path)
        if policy.automatic_purge:
            return CheckRecord(
                id="cache_policy",
                category="cache",
                status="FAIL",
                severity="integrity",
                message="automatic_purge must be false",
            )
    except Exception as exc:  # noqa: BLE001
        return CheckRecord(
            id="cache_policy",
            category="cache",
            status="FAIL",
            severity="integrity",
            message=f"cache policy load failed: {type(exc).__name__}",
        )
    return CheckRecord(
        id="cache_policy",
        category="cache",
        status="PASS",
        severity="info",
        message="cache policy valid",
    )


def _check_pipeline_schemas(root: Path) -> CheckRecord:
    missing = [rel for rel in PIPELINE_SCHEMAS if not (root / rel).is_file()]
    if missing:
        return CheckRecord(
            id="pipeline_schemas",
            category="schemas",
            status="FAIL",
            severity="integrity",
            message="pipeline/cache schemas missing",
            evidence={"missing": missing},
        )
    return CheckRecord(
        id="pipeline_schemas",
        category="schemas",
        status="PASS",
        severity="info",
        message="pipeline/cache schemas present",
        evidence={"count": len(PIPELINE_SCHEMAS)},
    )


def _check_stage_import_smoke() -> CheckRecord:
    before = {m for m in sys.modules if m.split(".", 1)[0] in {"torch", "pyarrow"}}
    try:
        import football_analytics.pipeline  # noqa: F401
        from football_analytics.pipeline.cache_key import compute_cache_key
        from football_analytics.pipeline.stage import SyntheticEchoStage
        from football_analytics.pipeline.types import ArtifactRef

        _ = (compute_cache_key, SyntheticEchoStage, ArtifactRef)
    except Exception as exc:  # noqa: BLE001
        return CheckRecord(
            id="stage_import_smoke",
            category="pipeline",
            status="FAIL",
            severity="integrity",
            message=f"import failed: {type(exc).__name__}",
        )
    after = {m for m in sys.modules if m.split(".", 1)[0] in {"torch", "pyarrow"}}
    newly = sorted(after - before)
    if newly:
        return CheckRecord(
            id="stage_import_smoke",
            category="pipeline",
            status="FAIL",
            severity="integrity",
            message="pipeline import loaded torch/pyarrow",
            evidence={"modules": newly[:12]},
        )
    return CheckRecord(
        id="stage_import_smoke",
        category="pipeline",
        status="PASS",
        severity="info",
        message="pipeline imports without torch/pyarrow",
    )


def _check_stage_key_tempdir() -> CheckRecord:
    import tempfile

    from football_analytics.core.hashing import hash_canonical_json
    from football_analytics.pipeline.artifacts import build_artifact_ref
    from football_analytics.pipeline.cache_key import compute_cache_key
    from football_analytics.pipeline.stage import SyntheticEchoStage

    stage = SyntheticEchoStage()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "x.bin"
        path.write_bytes(b"project-check-key")
        ref = build_artifact_ref("x", path, root=root, media_type="application/octet-stream")
        cfg = hash_canonical_json({"k": 1})
        compat = hash_canonical_json({"c": 1})
        k1 = compute_cache_key(
            stage=stage.identity,
            config_fingerprint=cfg,
            compatibility_fingerprint=compat,
            inputs={"x": ref},
        )
        k2 = compute_cache_key(
            stage=stage.identity,
            config_fingerprint=cfg,
            compatibility_fingerprint=compat,
            inputs={"x": ref},
        )
        if k1 != k2 or len(k1) != 64:
            return CheckRecord(
                id="stage_key_determinism",
                category="pipeline",
                status="FAIL",
                severity="integrity",
                message="cache key determinism failed",
            )
    return CheckRecord(
        id="stage_key_determinism",
        category="pipeline",
        status="PASS",
        severity="info",
        message="cache key determinism ok",
        evidence={"sample_key_prefix": k1[:12]},
    )


def _check_ci_workflow(root: Path) -> CheckRecord:
    from football_analytics.pipeline.ci_workflow_check import run_ci_workflow_checks

    wf = root / ".github" / "workflows" / "ci.yml"
    if not wf.is_file():
        return CheckRecord(
            id="ci_workflow",
            category="ci",
            status="FAIL",
            severity="error",
            message="CI workflow file missing",
            evidence={"path": str(wf)},
        )
    result = run_ci_workflow_checks(workflow=wf, project_root=root, strict=False)
    if result.errors:
        return CheckRecord(
            id="ci_workflow",
            category="ci",
            status="FAIL",
            severity="integrity",
            message="workflow safety findings",
            evidence={"errors": result.errors[:20], "warnings": result.warnings[:10]},
        )
    if result.warnings:
        return CheckRecord(
            id="ci_workflow",
            category="ci",
            status="WARN",
            severity="warn",
            message="workflow safety warnings",
            evidence={"warnings": result.warnings[:20]},
        )
    return CheckRecord(
        id="ci_workflow",
        category="ci",
        status="PASS",
        severity="info",
        message="workflow safety ok",
    )


def _host_only_skips() -> list[CheckRecord]:
    return [
        CheckRecord(
            id=check_id,
            category="host",
            status="SKIP",
            severity="info",
            message=message,
            evidence={"host_only": True},
        )
        for check_id, message in HOST_ONLY_SKIP_CHECKS
    ]


def _summarize(checks: list[CheckRecord]) -> dict[str, int]:
    summary = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for check in checks:
        summary[check.status] = summary.get(check.status, 0) + 1
    return summary


def _overall(checks: list[CheckRecord], *, strict: bool) -> tuple[str, int]:
    if any(c.status == "FAIL" for c in checks):
        return "FAIL", EXIT_INTEGRITY
    if any(c.status == "WARN" for c in checks):
        if strict:
            return "FAIL", EXIT_FINDING
        return "PASS_WITH_WARNINGS", EXIT_PASS
    return "PASS", EXIT_PASS


def run_project_checks(
    *,
    project_root: Path | None = None,
    profile: str = "local",
    mode: str = "quick",
    strict: bool = False,
) -> ProjectCheckReport:
    """Run the unified project check suite."""
    if profile not in {"local", "ci"}:
        raise ValueError(f"invalid profile: {profile}")
    if mode not in {"quick", "deep"}:
        raise ValueError(f"invalid mode: {mode}")

    root = project_root or _project_root()
    started = _now()
    t0 = time.perf_counter()
    checks: list[CheckRecord] = []

    def add(check_id: str, category: str, fn: Callable[[], CheckRecord]) -> None:
        checks.append(_run_check(check_id, category, fn))

    if profile == "ci":
        checks.extend(_host_only_skips())

    add("package_metadata", "package", _check_package_metadata)
    add(
        "protected_package_versions",
        "packages",
        lambda: _check_protected_packages(profile=profile),
    )
    add("config_defaults", "config", lambda: _check_config_defaults(root))
    add("cache_policy", "cache", lambda: _check_cache_policy(root))
    add("pipeline_schemas", "schemas", lambda: _check_pipeline_schemas(root))
    add("stage_import_smoke", "pipeline", _check_stage_import_smoke)
    add("stage_key_determinism", "pipeline", _check_stage_key_tempdir)

    if profile == "local":
        add("git_status", "git", lambda: _check_git(root))
        add("git_remote_sanitized", "git", lambda: _check_remote_sanitized(root))
        add("storage_paths", "storage", lambda: _check_storage_readonly(root, profile=profile))
    else:
        # Explicit SKIP already listed for host storage; still record workflow + secrets.
        pass

    add(
        "secrets",
        "security",
        lambda: _run_script(
            root,
            "scripts/check_secrets.py",
            ["--root", str(root), "--quiet"],
            check_id="secrets",
            category="security",
        ),
    )
    add(
        "runtime_foundation",
        "runtime",
        lambda: _run_script(
            root,
            "scripts/check_runtime_foundation.py",
            ["--config", "configs/project/defaults.yaml", "--quiet"],
            check_id="runtime_foundation",
            category="runtime",
        ),
    )
    add(
        "data_contracts",
        "data",
        lambda: _run_script(
            root,
            "scripts/check_data_contracts.py",
            ["--registry", "configs/data/schema_registry.yaml", "--quiet"],
            check_id="data_contracts",
            category="data",
        ),
    )
    add("ci_workflow", "ci", lambda: _check_ci_workflow(root))

    # Document: full pytest is run by CI workflow separately to avoid recursion.
    add(
        "unit_tests",
        "tests",
        lambda: CheckRecord(
            id="unit_tests",
            category="tests",
            status="SKIP",
            severity="info",
            message="pytest not executed inside project check (CI runs pytest separately)",
            evidence={"reason": "avoid_recursive_pytest"},
        ),
    )

    if mode == "deep":
        if profile == "ci":
            checks.append(
                CheckRecord(
                    id="deep_host_smokes",
                    category="deep",
                    status="SKIP",
                    severity="info",
                    message="deep synthetic smokes are local/host oriented",
                )
            )
        else:
            add(
                "data_contracts_synthetic",
                "data",
                lambda: _run_script(
                    root,
                    "scripts/check_data_contracts.py",
                    [
                        "--registry",
                        "configs/data/schema_registry.yaml",
                        "--synthetic-roundtrip",
                        "--migration-smoke",
                        "--quiet",
                    ],
                    check_id="data_contracts_synthetic",
                    category="data",
                ),
            )
            add(
                "stage_cache_synthetic",
                "cache",
                lambda: _run_script(
                    root,
                    "scripts/check_stage_cache.py",
                    [
                        "--config",
                        "configs/system/cache_policy.yaml",
                        "--synthetic",
                        "--corruption-smoke",
                        "--concurrency-smoke",
                        "--quiet",
                    ],
                    check_id="stage_cache_synthetic",
                    category="cache",
                ),
            )
            add(
                "runtime_foundation_synthetic",
                "runtime",
                lambda: _run_script(
                    root,
                    "scripts/check_runtime_foundation.py",
                    [
                        "--config",
                        "configs/project/defaults.yaml",
                        "--synthetic-run",
                        "--quiet",
                    ],
                    check_id="runtime_foundation_synthetic",
                    category="runtime",
                ),
            )

            # Cleanup verification: ensure stage_cache fixture root has no leftover run_* dirs
            def _cleanup_verification() -> CheckRecord:
                stage_root = Path("/home/fdoblak/workspace/stage_cache_checks/fixtures")
                leftover: list[str] = []
                if stage_root.is_dir():
                    leftover = [p.name for p in stage_root.iterdir() if p.is_dir()]
                if leftover:
                    return CheckRecord(
                        id="cleanup_verification",
                        category="cleanup",
                        status="FAIL",
                        severity="integrity",
                        message="synthetic fixtures left behind",
                        evidence={"leftover": leftover[:20]},
                    )
                return CheckRecord(
                    id="cleanup_verification",
                    category="cleanup",
                    status="PASS",
                    severity="info",
                    message="no leftover synthetic fixtures",
                )

            add("cleanup_verification", "cleanup", _cleanup_verification)

    finished = _now()
    duration_ms = max(0, int((time.perf_counter() - t0) * 1000))
    summary = _summarize(checks)
    overall_status, exit_code = _overall(checks, strict=strict)
    return ProjectCheckReport(
        schema_version=1,
        profile=profile,
        mode=mode,
        started_at_utc=started,
        finished_at_utc=finished,
        duration_ms=duration_ms,
        overall_status=overall_status,
        exit_code=exit_code,
        checks=checks,
        summary=summary,
        environment_classification=("CI_PROFILE" if profile == "ci" else "LOCAL_HOST_PROFILE"),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2D unified project validator")
    p.add_argument("--profile", choices=("local", "ci"), default="local")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", default=False)
    mode.add_argument("--deep", action="store_true", default=False)
    p.add_argument("--json-out")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mode = "deep" if args.deep else "quick"
    try:
        report = run_project_checks(
            profile=str(args.profile),
            mode=mode,
            strict=bool(args.strict),
        )
    except ValueError as exc:
        print(f"status=FAIL exit_code={EXIT_CONFIG} error={exc}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001
        print(f"status=FAIL exit_code={EXIT_CONFIG} error={type(exc).__name__}", file=sys.stderr)
        return EXIT_CONFIG

    payload = report.to_dict()
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        for check in report.checks:
            print(f"{check.status:4} {check.id}: {check.message} ({check.duration_ms}ms)")
        print(
            f"overall={report.overall_status} exit_code={report.exit_code} "
            f"summary={report.summary}"
        )
    return int(report.exit_code)
