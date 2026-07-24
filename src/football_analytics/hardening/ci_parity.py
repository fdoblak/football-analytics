"""15D CI / local parity helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy
from football_analytics.pipeline.ci_workflow_check import run_ci_workflow_checks
from football_analytics.pipeline.project_check import PROTECTED_PACKAGE_VERSIONS


def remote_ci_status(policy: HardeningPolicy | None = None) -> dict[str, Any]:
    """Document remote CI as unverifiable in agent API context (RISK-042)."""
    pol = policy or load_hardening_policy()
    return {
        "risk_id": "RISK-042",
        "remote_ci_status": str(pol.raw["ci_parity"]["remote_ci_status"]),
        "invented_green_remote_ci": False,
        "local_parity_required": bool(pol.raw["ci_parity"]["require_local_deep_validation"]),
        "note": (
            "If GitHub API returns 403, treat remote CI as external finding "
            "and run local CI-equivalent."
        ),
    }


def check_protected_pins() -> dict[str, Any]:
    """Ensure protected pin map is present (versions enforced by project_check)."""
    if not PROTECTED_PACKAGE_VERSIONS:
        raise AssertionError("protected pin map empty")
    return {
        "protected_packages": sorted(PROTECTED_PACKAGE_VERSIONS.keys()),
        "pin_count": len(PROTECTED_PACKAGE_VERSIONS),
        "status": "PASS",
    }


def check_external_repo_lock(repo_root: Path) -> dict[str, Any]:
    lock = repo_root / "external_repos.lock.yaml"
    if not lock.is_file():
        raise FileNotFoundError(f"external repo lock missing: {lock}")
    text = lock.read_text(encoding="utf-8")
    if "repositories:" not in text and "repositories" not in text:
        raise AssertionError("external_repos.lock.yaml missing repositories")
    return {"path": str(lock), "bytes": lock.stat().st_size, "status": "PASS"}


def check_workflow_sha_pins(repo_root: Path) -> dict[str, Any]:
    """Reuse Stage 2D CI workflow validator (SHA pins / allowlist)."""
    result = run_ci_workflow_checks(
        workflow=Path(".github/workflows/ci.yml"),
        project_root=repo_root,
    )
    return {
        "exit_code": int(result.exit_code),
        "status": result.status,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }


def local_deep_validation_commands() -> list[str]:
    """Documented local CI-equivalent command list."""
    return [
        "ruff check src tests scripts",
        "black --check src tests scripts",
        "isort --check-only src tests scripts",
        "mypy src/football_analytics",
        "pytest -q",
        "python scripts/check_secrets.py --root .",
        "python scripts/check_ci_workflow.py",
        "python scripts/check_project.py --profile local --quick",
        "python scripts/local_deep_validation.py --skip-project",
        "python scripts/check_prerelease_hardening.py",
    ]
