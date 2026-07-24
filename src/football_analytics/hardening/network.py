"""No-network defaults and outbound policy checks."""

from __future__ import annotations

from typing import Any

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


class NetworkPolicyError(ValueError):
    """Network policy violation."""


def network_policy_snapshot(policy: HardeningPolicy | None = None) -> dict[str, Any]:
    pol = policy or load_hardening_policy()
    net = pol.network_defaults
    return {
        "network_video_download_allowed": bool(net["network_video_download_allowed"]),
        "dataset_download_allowed": bool(net["dataset_download_allowed"]),
        "large_model_download_allowed": bool(net["large_model_download_allowed"]),
        "allow_outbound_by_default": bool(net["allow_outbound_by_default"]),
        "no_network_default": True,
    }


def assert_no_network_default(policy: HardeningPolicy | None = None) -> dict[str, Any]:
    snap = network_policy_snapshot(policy)
    if any(
        [
            snap["network_video_download_allowed"],
            snap["dataset_download_allowed"],
            snap["large_model_download_allowed"],
            snap["allow_outbound_by_default"],
        ]
    ):
        raise NetworkPolicyError("network downloads must remain disabled by default")
    return snap


def assert_download_allowed(kind: str, *, policy: HardeningPolicy | None = None) -> None:
    """Raise unless an explicit policy flag allows the download kind."""
    snap = network_policy_snapshot(policy)
    key = {
        "video": "network_video_download_allowed",
        "dataset": "dataset_download_allowed",
        "model": "large_model_download_allowed",
    }.get(kind)
    if key is None:
        raise NetworkPolicyError(f"unknown download kind: {kind}")
    if not snap[key]:
        raise NetworkPolicyError(f"{kind} download blocked by no-network default")
