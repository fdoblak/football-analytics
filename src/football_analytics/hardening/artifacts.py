"""Secret/log redaction and large-artifact prevention gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.core.redaction import REDACTED, is_sensitive_key, redact_value
from football_analytics.evidence.collector import is_safe_evidence_file
from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


class ArtifactPolicyError(ValueError):
    """Large or unsafe artifact rejected."""


def assert_redacted(payload: Any) -> Any:
    """Return a deep-redacted copy; sensitive keys become REDACTED."""
    out = redact_value(payload)
    if isinstance(payload, dict):
        for key in payload:
            if is_sensitive_key(key) and isinstance(out, dict) and out.get(key) != REDACTED:
                raise ArtifactPolicyError(f"sensitive key not redacted: {key}")
    return out


def assert_safe_evidence_candidate(
    path: Path,
    *,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    """Reject videos/weights/oversized files from evidence."""
    pol = policy or load_hardening_policy()
    ok, reason = is_safe_evidence_file(path)
    suffix = path.suffix.lower()
    unsafe = {str(s).lower() for s in pol.raw["artifacts"]["unsafe_suffixes"]}
    if suffix in unsafe:
        raise ArtifactPolicyError(f"unsafe evidence suffix: {suffix}")
    if not ok:
        raise ArtifactPolicyError(f"evidence rejected: {reason}")
    size = path.stat().st_size
    max_bytes = int(pol.raw["artifacts"]["max_evidence_file_bytes"])
    if size > max_bytes:
        raise ArtifactPolicyError(f"evidence too large: {size} > {max_bytes}")
    return {"path": str(path), "ok": True, "size_bytes": size, "reason": reason}
