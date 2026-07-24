"""Stage 15 pre-release hardening policies and machine-local gates."""

from __future__ import annotations

from football_analytics.hardening.policy import (
    HardeningPolicy,
    hardening_policy_fingerprint,
    load_hardening_policy,
)

GATE_HINT = (
    "PASS_WITH_FINDINGS — STAGE 15 PRE-RELEASE COMPLETE; "
    "ALL IMPLEMENTATION STAGES CLOSED; "
    "ONLY REAL-MATCH ACCEPTANCE STAGE 16 REMAINS"
)

__all__ = [
    "GATE_HINT",
    "HardeningPolicy",
    "hardening_policy_fingerprint",
    "load_hardening_policy",
]
