"""15B licensing / adapter isolation gates (no invented legal approval)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy

COPYLEFT_MARKERS = ("AGPL", "GPL-2.0", "GPL-3.0", "GPL")


class LicensingGateError(ValueError):
    """Licensing isolation gate failure."""


def load_model_registry(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "models" not in raw:
        raise LicensingGateError("model_registry.yaml must contain models list")
    return raw


def scan_model_registry_approvals(
    registry_path: Path,
    *,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    """Assert no model is production_approved; copyleft stays evaluation_only."""
    pol = policy or load_hardening_policy()
    if bool(pol.raw["licensing"].get("invent_legal_approval")):
        raise LicensingGateError("invent_legal_approval must remain false")
    registry = load_model_registry(registry_path)
    models = list(registry.get("models") or [])
    findings: list[str] = []
    copyleft_ids: list[str] = []
    evaluation_only_ids: list[str] = []
    for entry in models:
        if not isinstance(entry, dict):
            findings.append("non-mapping model entry")
            continue
        mid = str(entry.get("id") or "<missing>")
        if entry.get("production_approved") is True:
            findings.append(f"{mid}: production_approved=true forbidden")
        approval = str(entry.get("approval") or "")
        if approval == "evaluation_only":
            evaluation_only_ids.append(mid)
        notes = " ".join(
            [
                str(entry.get("notes") or ""),
                str(entry.get("license") or ""),
                str(entry.get("license_status") or ""),
            ]
        ).upper()
        if any(
            m in notes or m in str(entry.get("source_repo") or "").upper() for m in COPYLEFT_MARKERS
        ):
            copyleft_ids.append(mid)
            if approval != "evaluation_only":
                findings.append(f"{mid}: copyleft-adjacent must be evaluation_only")
        # Ultralytics / NBJW known copyleft adapters
        src = str(entry.get("source_repo") or "").lower()
        if "ultralytics" in mid.lower() or "no_bells" in src or "yolo" in mid.lower():
            if entry.get("production_approved") is True:
                findings.append(f"{mid}: AGPL/GPL adapter cannot be production_approved")
            if approval != "evaluation_only":
                findings.append(f"{mid}: AGPL/GPL adapter must be evaluation_only")
    if findings and bool(pol.raw["licensing"]["forbid_production_approved_true"]):
        raise LicensingGateError("; ".join(findings))
    return {
        "model_count": len(models),
        "evaluation_only_ids": evaluation_only_ids,
        "copyleft_adjacent_ids": sorted(set(copyleft_ids)),
        "production_approved_any": False,
        "legal_approval_invented": False,
        "findings": findings,
        "status": "PASS" if not findings else "FAIL",
    }


def fallback_no_model_behavior() -> dict[str, Any]:
    """Documented safe behavior when a gated model cannot run."""
    return {
        "behavior": "cpu_stub_or_skip",
        "metrics": "not_evaluable",
        "reason_code": "MODEL_UNAVAILABLE_OR_LICENSE_GATED",
        "production_approved": False,
        "invent_outputs": False,
    }


def third_party_notices_present(repo_root: Path) -> dict[str, Any]:
    """Check license inventory / notices docs exist (not a legal clearance)."""
    inventory = repo_root / "docs" / "legal" / "license_inventory.md"
    notices = repo_root / "docs" / "legal" / "third_party_notices.md"
    return {
        "license_inventory": inventory.is_file(),
        "third_party_notices": notices.is_file(),
        "path_inventory": str(inventory),
        "path_notices": str(notices),
        "legal_clearance": False,
        "note": "Technical inventory only; not legal approval",
    }
