"""Integrity / lineage checks for Stage 9E physical metric fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class PipelineIntegrityError(ValueError):
    """Hard integrity failure — no silent repair."""


def _ids(block: Mapping[str, Any] | None) -> tuple[str, str, str]:
    if not block:
        return "", "", ""
    return (
        str(block.get("run_id") or ""),
        str(block.get("video_id") or ""),
        str(block.get("target_player_id") or block.get("target_id") or ""),
    )


def assert_same_target_scope(
    *blocks: Mapping[str, Any] | None,
    require: bool = True,
) -> tuple[str, str, str]:
    ids = [_ids(b) for b in blocks if b]
    if not ids:
        raise PipelineIntegrityError("no input blocks for identity scope")
    base = ids[0]
    for other in ids[1:]:
        if other != base and require:
            raise PipelineIntegrityError(f"SOURCE_MIX run/video/target mismatch: {base} vs {other}")
    if not all(base):
        raise PipelineIntegrityError("missing run_id/video_id/target_player_id")
    return base


def assert_confirmed_identity(
    identity: Mapping[str, Any],
    *,
    require_confirmed: bool = True,
    forbid_revoked: bool = True,
) -> str | None:
    """Return identity status error code or None if OK."""
    status = str(
        identity.get("identity_status")
        or identity.get("identity_quality")
        or identity.get("assignment_status")
        or ""
    ).lower()
    revoked = identity.get("assignment_revoked") is True or status in {
        "revoked",
        "identity_revoked",
    }
    if forbid_revoked and revoked:
        return "identity_unconfirmed"
    if require_confirmed and status and status != "confirmed":
        return "identity_unconfirmed"
    return None


def assert_fingerprint_match(
    *,
    expected: str | None,
    actual: str | None,
    label: str,
) -> None:
    if expected is None or actual is None:
        raise PipelineIntegrityError(f"FINGERPRINT_MISSING:{label}")
    if str(expected) != str(actual):
        raise PipelineIntegrityError(f"FINGERPRINT_MISMATCH:{label}")


def assert_receipt_fresh(
    receipt: Mapping[str, Any],
    *,
    expected_status: str = "succeeded",
) -> None:
    if str(receipt.get("status") or receipt.get("completion_status") or "") not in {
        expected_status,
        "succeeded",
    }:
        raise PipelineIntegrityError("STALE_OR_FAILED_RECEIPT")
    if receipt.get("evaluation_status") is None and receipt.get("config_fingerprint") is None:
        raise PipelineIntegrityError("STALE_OR_INCOMPLETE_RECEIPT")


def assert_finite_non_negative(value: float | int | None, *, label: str) -> None:
    if value is None:
        return
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise PipelineIntegrityError(f"INVALID_NUMERIC:{label}")


def check_duration_mass_consistency(
    *,
    eligible_us: int,
    component_us: Sequence[int],
    tolerance_us: int = 2,
    label: str,
) -> str | None:
    """Soft check: component durations should not exceed eligible (+tol)."""
    total = sum(int(x) for x in component_us)
    if total < 0:
        return f"source_inconsistent:{label}_negative"
    if total > int(eligible_us) + int(tolerance_us):
        return f"source_inconsistent:{label}_mass_exceeds_eligible"
    return None


def check_distance_recount(
    *,
    reported_m: float | None,
    recounted_m: float | None,
    tol: float = 1e-3,
) -> str | None:
    if reported_m is None or recounted_m is None:
        return None
    if abs(float(reported_m) - float(recounted_m)) > tol:
        return "source_inconsistent:distance_recount_mismatch"
    return None


__all__ = [
    "PipelineIntegrityError",
    "assert_same_target_scope",
    "assert_confirmed_identity",
    "assert_fingerprint_match",
    "assert_receipt_fresh",
    "assert_finite_non_negative",
    "check_duration_mass_consistency",
    "check_distance_recount",
]
