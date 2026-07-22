"""Minimal CLI for Stage 2A: --version and info only."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_active_backend(paths_yaml: Path) -> str:
    if not paths_yaml.is_file():
        return "unknown (paths.yaml missing)"
    try:
        import yaml
    except ImportError:
        return "unknown (PyYAML not available)"
    try:
        data = yaml.safe_load(paths_yaml.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return "unknown (paths.yaml unreadable)"
    storage = data.get("storage") if isinstance(data, dict) else None
    if isinstance(storage, dict) and storage.get("active_backend"):
        return str(storage["active_backend"])
    return "unknown"


def cmd_info() -> int:
    from football_analytics import __version__

    root = _project_root()
    paths_yaml = root / "configs" / "system" / "paths.yaml"
    print(f"project_version: {__version__}")
    print(f"python_version: {platform.python_version()}")
    print(f"python_executable: {sys.executable}")
    print(f"project_root: {root}")
    print(f"paths_yaml_present: {paths_yaml.is_file()}")
    print(f"active_storage_backend: {_read_active_backend(paths_yaml)}")
    print("gpu_inference: not_started (Stage 2A CLI is side-effect free)")
    print("gpu_classification: AGENT_CONTEXT_GPU_UNVERIFIABLE")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-analytics",
        description="Broadcast football video analytics pipeline (Stage 2A CLI)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="Show safe environment summary (no secrets)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.version:
        from football_analytics import __version__

        print(__version__)
        return 0
    if args.command == "info":
        return cmd_info()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
