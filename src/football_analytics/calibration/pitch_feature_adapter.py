"""Lazy NBJW HRNet adapter via importlib (Stage 8B).

Architecture is loaded from locked absolute paths under SoccerNet sn-banner /
No_Bells_Just_Whistles (GPL-2.0). Source is NOT vendored into this repository.
Linking risk → evaluation_only; production_approved=false.

Importing ``football_analytics.calibration`` must NOT load weights or HRNet modules.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from football_analytics.calibration.pitch_feature_postprocess import (
    DecodedKeypoint,
    DecodedLine,
    decode_keypoints_from_heatmap,
    decode_lines_from_heatmap,
)
from football_analytics.calibration.pitch_feature_preprocess import (
    StretchTransform,
    preprocess_rgb_uint8_to_tensor,
)
from football_analytics.core.hashing import sha256_file

# Locked absolute paths (also in baseline YAML).
DEFAULT_KP_MODULE = Path(
    "/home/fdoblak/projects/soccernet/sn-banner/camera_calibration/"
    "No_Bells_Just_Whistles/model/cls_hrnet.py"
)
DEFAULT_LINES_MODULE = Path(
    "/home/fdoblak/projects/soccernet/sn-banner/camera_calibration/"
    "No_Bells_Just_Whistles/model/cls_hrnet_l.py"
)
DEFAULT_KP_CFG = Path(
    "/home/fdoblak/projects/soccernet/sn-banner/camera_calibration/"
    "No_Bells_Just_Whistles/config/hrnetv2_w48.yaml"
)
DEFAULT_LINES_CFG = Path(
    "/home/fdoblak/projects/soccernet/sn-banner/camera_calibration/"
    "No_Bells_Just_Whistles/config/hrnetv2_w48_l.yaml"
)

EXPECTED_KP_SHA = "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113"
EXPECTED_LINES_SHA = "2751242917f8c0f858a396e0cfe4521be39fe07bf049590eb21714526acecac1"
EXPECTED_KP_SIZE = 264964645
EXPECTED_LINES_SIZE = 264857893


class PitchFeatureAdapterError(RuntimeError):
    """Adapter load / inference failure."""


@dataclass
class PitchFeatureInferenceResult:
    keypoints: list[DecodedKeypoint]
    lines: list[DecodedLine]
    transform: StretchTransform
    kp_heatmap_shape: tuple[int, ...]
    lines_heatmap_shape: tuple[int, ...]
    device: str
    kp_model_sha256: str
    lines_model_sha256: str


def reject_network_path(path: str | Path, *, label: str) -> Path:
    text = str(path).strip()
    if not text:
        raise PitchFeatureAdapterError(f"{label} empty")
    lowered = text.lower()
    if "://" in text or lowered.startswith(("http:", "https:", "ftp:", "s3:", "gs:")):
        raise PitchFeatureAdapterError("NETWORK_WEIGHTS_FORBIDDEN")
    target = Path(text)
    if not target.is_absolute():
        raise PitchFeatureAdapterError(f"{label} must be absolute")
    if ".." in target.parts:
        raise PitchFeatureAdapterError(f"{label} path escape")
    if target.is_symlink():
        raise PitchFeatureAdapterError(f"{label} must not be a symlink")
    if not target.is_file():
        raise PitchFeatureAdapterError(f"{label} missing: {target}")
    return target


def _set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_HOME", os.environ.get("TORCH_HOME", "/tmp/torch_offline_unused"))


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    path = reject_network_path(path, label=module_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PitchFeatureAdapterError(f"importlib spec failed for {path}")
    mod = importlib.util.module_from_spec(spec)
    # Unique names avoid colliding with unrelated packages.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    path = reject_network_path(path, label="hrnet_config")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PitchFeatureAdapterError("HRNet config must be a mapping")
    return data


def _torch_load_state(path: Path, *, map_location: str) -> Any:
    import torch

    # Prefer weights_only=True (torch>=2.0) for state_dict safety; fall back if rejected.
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception:
        # Some checkpoints may not be pure tensor/dicts under weights_only.
        # Provenance: local registry-locked SV_*.pth only — still evaluation_only.
        return torch.load(str(path), map_location=map_location, weights_only=False)


def verify_weight_file(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
) -> tuple[Path, str]:
    target = reject_network_path(path, label="weights")
    size = int(target.stat().st_size)
    if expected_size is not None and size != int(expected_size):
        raise PitchFeatureAdapterError(f"MODEL_SIZE_MISMATCH: got {size} expected {expected_size}")
    actual = sha256_file(target)
    if actual.lower() != str(expected_sha256).lower():
        raise PitchFeatureAdapterError("MODEL_HASH_MISMATCH")
    return target, actual.lower()


def resolve_device(policy: str) -> str:
    prefer_cuda = policy == "prefer_cuda_else_cpu"
    require_cuda = policy == "cuda_required"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if require_cuda and not cuda_ok:
        raise PitchFeatureAdapterError("CUDA_REQUIRED_UNAVAILABLE")
    if (prefer_cuda or require_cuda) and cuda_ok:
        return "cuda:0"
    return "cpu"


class NbjwHrnetPitchFeatureAdapter:
    """Holds both SV_kp and SV_lines HRNet models; constructed only on explicit load."""

    def __init__(
        self,
        *,
        kp_model: Any,
        lines_model: Any,
        device: str,
        kp_sha256: str,
        lines_sha256: str,
        kp_path: Path,
        lines_path: Path,
        config: Mapping[str, Any],
    ) -> None:
        self.kp_model = kp_model
        self.lines_model = lines_model
        self.device = device
        self.kp_sha256 = kp_sha256
        self.lines_sha256 = lines_sha256
        self.kp_path = kp_path
        self.lines_path = lines_path
        self.config = config

    @classmethod
    def load(
        cls,
        *,
        config: Mapping[str, Any],
        kp_weights_path: str | Path,
        lines_weights_path: str | Path,
        kp_expected_sha256: str,
        lines_expected_sha256: str,
        kp_expected_size: int | None = EXPECTED_KP_SIZE,
        lines_expected_size: int | None = EXPECTED_LINES_SIZE,
        device_policy: str | None = None,
    ) -> NbjwHrnetPitchFeatureAdapter:
        _set_offline_env()

        device = resolve_device(device_policy or str(config.get("device_policy", "cpu_only")))
        kp_path, kp_sha = verify_weight_file(
            kp_weights_path,
            expected_sha256=kp_expected_sha256,
            expected_size=kp_expected_size,
        )
        lines_path, lines_sha = verify_weight_file(
            lines_weights_path,
            expected_sha256=lines_expected_sha256,
            expected_size=lines_expected_size,
        )

        kp_mod = _load_module_from_path(
            "fa_nbjw_cls_hrnet_kp",
            Path(str(config["hrnet_kp_module_path"])),
        )
        lines_mod = _load_module_from_path(
            "fa_nbjw_cls_hrnet_lines",
            Path(str(config["hrnet_lines_module_path"])),
        )
        if not hasattr(kp_mod, "get_cls_net") or not hasattr(lines_mod, "get_cls_net"):
            raise PitchFeatureAdapterError("HRNet get_cls_net export missing")

        cfg_kp = _load_yaml_config(Path(str(config["hrnet_kp_config_path"])))
        cfg_lines = _load_yaml_config(Path(str(config["hrnet_lines_config_path"])))
        if int(cfg_kp.get("MODEL", {}).get("NUM_JOINTS", -1)) != 58:
            raise PitchFeatureAdapterError("kp NUM_JOINTS mismatch")
        if int(cfg_lines.get("MODEL", {}).get("NUM_JOINTS", -1)) != 24:
            raise PitchFeatureAdapterError("lines NUM_JOINTS mismatch")

        kp_state = _torch_load_state(kp_path, map_location="cpu")
        lines_state = _torch_load_state(lines_path, map_location="cpu")

        kp_model = kp_mod.get_cls_net(cfg_kp, pretrained="")
        kp_model.load_state_dict(kp_state)
        kp_model.to(device)
        kp_model.eval()

        lines_model = lines_mod.get_cls_net(cfg_lines, pretrained="")
        lines_model.load_state_dict(lines_state)
        lines_model.to(device)
        lines_model.eval()

        return cls(
            kp_model=kp_model,
            lines_model=lines_model,
            device=device,
            kp_sha256=kp_sha,
            lines_sha256=lines_sha,
            kp_path=kp_path,
            lines_path=lines_path,
            config=config,
        )

    def infer_rgb(self, image_rgb: Any) -> PitchFeatureInferenceResult:
        import torch

        tensor, transform = preprocess_rgb_uint8_to_tensor(image_rgb)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            heat_kp = self.kp_model(tensor)
            heat_lines = self.lines_model(tensor)
        if heat_kp.shape[1] != int(self.config["num_joints_kp"]):
            raise PitchFeatureAdapterError(
                f"kp output channels {heat_kp.shape[1]} != {self.config['num_joints_kp']}"
            )
        if heat_lines.shape[1] != int(self.config["num_joints_lines"]):
            raise PitchFeatureAdapterError(
                f"lines output channels {heat_lines.shape[1]} != {self.config['num_joints_lines']}"
            )
        # Drop background channel.
        heat_kp_u = heat_kp[:, :-1, :, :]
        heat_lines_u = heat_lines[:, :-1, :, :]
        kps = decode_keypoints_from_heatmap(
            heat_kp_u,
            transform=transform,
            score_threshold=float(self.config["kp_score_threshold"]),
            scale=int(self.config["peak_decode_scale"]),
            max_peaks=int(self.config["kp_max_peaks"]),
            min_distance=int(self.config["kp_min_peak_distance"]),
            expected_channels=int(self.config["kp_channels_used"]),
            duplicate_distance_px=float(self.config["duplicate_keypoint_distance_px"]),
        )
        lines = decode_lines_from_heatmap(
            heat_lines_u,
            transform=transform,
            score_threshold=float(self.config["line_score_threshold"]),
            scale=int(self.config["peak_decode_scale"]),
            max_peaks=int(self.config["lines_max_peaks"]),
            min_distance=int(self.config["lines_min_peak_distance"]),
            expected_channels=int(self.config["lines_channels_used"]),
            minimum_length_px=float(self.config["minimum_line_length_px"]),
            duplicate_endpoint_distance_px=float(
                self.config["duplicate_line_endpoint_distance_px"]
            ),
        )
        return PitchFeatureInferenceResult(
            keypoints=kps,
            lines=lines,
            transform=transform,
            kp_heatmap_shape=tuple(int(x) for x in heat_kp.shape),
            lines_heatmap_shape=tuple(int(x) for x in heat_lines.shape),
            device=self.device,
            kp_model_sha256=self.kp_sha256,
            lines_model_sha256=self.lines_sha256,
        )


__all__ = [
    "DEFAULT_KP_MODULE",
    "DEFAULT_LINES_MODULE",
    "DEFAULT_KP_CFG",
    "DEFAULT_LINES_CFG",
    "EXPECTED_KP_SHA",
    "EXPECTED_LINES_SHA",
    "EXPECTED_KP_SIZE",
    "EXPECTED_LINES_SIZE",
    "PitchFeatureAdapterError",
    "PitchFeatureInferenceResult",
    "NbjwHrnetPitchFeatureAdapter",
    "reject_network_path",
    "verify_weight_file",
    "resolve_device",
]
