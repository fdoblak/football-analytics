"""GPU / CPU device profiles for Stage 15 (RTX 3050 4GB + Agent GPU unverifiable)."""

from __future__ import annotations

from typing import Any

from football_analytics.core.environment import GPU_CLASSIFICATION
from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


def agent_gpu_classification() -> str:
    """Canonical agent-context classification (never invent verified host GPU)."""
    return GPU_CLASSIFICATION


def rtx3050_bounded_batch_profile(policy: HardeningPolicy | None = None) -> dict[str, Any]:
    """Return the bounded batch profile for RTX 3050 4GB class cards."""
    pol = policy or load_hardening_policy()
    profile = dict(pol.rtx3050_profile)
    profile["cpu_fallback_required"] = bool(pol.raw["gpu"]["cpu_fallback_required"])
    profile["prefer_cuda_else_cpu"] = bool(pol.raw["gpu"]["prefer_cuda_else_cpu"])
    profile["agent_gpu_classification"] = agent_gpu_classification()
    return profile


def resolve_device_request(
    *,
    prefer_cuda: bool = True,
    cuda_available: bool | None = None,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    """Resolve device with mandatory CPU fallback; do not claim Agent GPU verified."""
    pol = policy or load_hardening_policy()
    classification = agent_gpu_classification()
    if classification != pol.gpu_classification_default:
        # Keep environment constant as source of truth.
        classification = agent_gpu_classification()
    if cuda_available is None:
        # Do not import torch here; caller may probe optionally.
        selected = "cpu"
        reason = "cuda_availability_unprobed_default_cpu"
    elif cuda_available and prefer_cuda and bool(pol.raw["gpu"]["prefer_cuda_else_cpu"]):
        selected = "cuda"
        reason = "cuda_available"
    else:
        selected = "cpu"
        reason = "cpu_fallback" if not cuda_available else "prefer_cpu"
    profile = rtx3050_bounded_batch_profile(pol)
    return {
        "device": selected,
        "reason": reason,
        "gpu_classification": classification,
        "batch_profile": profile,
        "cpu_fallback_required": True,
        "agent_gpu_unverifiable": classification == "AGENT_CONTEXT_GPU_UNVERIFIABLE",
    }


def optional_cuda_smoke() -> dict[str, Any]:
    """Best-effort CUDA smoke; never fails the host when torch/CUDA absent."""
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return {
            "cuda_smoke": "skipped",
            "reason": f"torch_unavailable:{type(exc).__name__}",
            "gpu_classification": agent_gpu_classification(),
        }
    available = bool(torch.cuda.is_available())
    out: dict[str, Any] = {
        "cuda_smoke": "ran",
        "cuda_available": available,
        "gpu_classification": agent_gpu_classification(),
        "torch_imported": True,
    }
    if available:
        try:
            name = torch.cuda.get_device_name(0)
            out["device_name"] = name
            # Tiny allocation smoke then free.
            x = torch.zeros(1, device="cuda")
            del x
            torch.cuda.empty_cache()
            out["allocation_ok"] = True
        except Exception as exc:  # noqa: BLE001
            out["allocation_ok"] = False
            out["allocation_error"] = type(exc).__name__
    return out
