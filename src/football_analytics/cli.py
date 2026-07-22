"""CLI: version, info, and Stage 2B foundation-safe helpers."""

from __future__ import annotations

import argparse
import json
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


def cmd_run_id() -> int:
    from football_analytics.core.run_id import generate_run_id

    print(generate_run_id())
    return 0


def cmd_config_validate(config_path: Path) -> int:
    from football_analytics.core.config import (
        ConfigError,
        default_defaults_path,
        load_resolved_config,
    )

    try:
        defaults = default_defaults_path()
        if config_path.resolve() == defaults.resolve():
            load_resolved_config(defaults_path=config_path)
        else:
            load_resolved_config(defaults_path=defaults, user_config_path=config_path)
        print("config_valid: true")
        return 0
    except ConfigError as exc:
        print(f"config_valid: false\nerror: {exc}", file=sys.stderr)
        return 1


def cmd_config_fingerprint(config_path: Path, *, as_json: bool) -> int:
    from football_analytics.core.config import (
        ConfigError,
        config_fingerprint,
        default_defaults_path,
        load_resolved_config,
    )

    try:
        if config_path.resolve() == default_defaults_path().resolve():
            cfg = load_resolved_config(defaults_path=config_path)
        else:
            cfg = load_resolved_config(
                defaults_path=default_defaults_path(),
                user_config_path=config_path,
            )
        fp = config_fingerprint(cfg)
        if as_json:
            print(json.dumps(fp, sort_keys=True, ensure_ascii=False))
        else:
            print(f"algorithm: {fp['algorithm']}")
            print(f"canonicalization_version: {fp['canonicalization_version']}")
            print(f"digest: {fp['digest']}")
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_environment_show(*, as_json: bool) -> int:
    from football_analytics import __version__
    from football_analytics.core.config import (
        config_fingerprint,
        default_defaults_path,
        load_resolved_config,
    )
    from football_analytics.core.environment import build_environment_record

    cfg = load_resolved_config(defaults_path=default_defaults_path())
    fp = config_fingerprint(cfg)
    rec = build_environment_record(
        project_version=__version__,
        config_fingerprint=fp,
        repo_root=_project_root(),
    )
    if as_json:
        print(json.dumps(rec, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"project_version: {rec['project_version']}")
        print(f"python_version: {rec['python']['version']}")
        print(f"conda_env: {rec['conda']['environment_name']}")
        print(f"git_commit: {rec['git']['commit']}")
        print(f"git_dirty: {rec['git']['dirty']}")
        print(f"remote: {rec['git']['remote_sanitized']}")
        print(f"gpu_classification: {rec['gpu_validation']['classification']}")
        print(f"config_fingerprint: {rec['config_fingerprint']['digest']}")
        print(f"packages_allowlist_count: {len(rec['packages'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-analytics",
        description="Broadcast football video analytics pipeline (Stage 2B CLI)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="Show safe environment summary (no secrets)")
    sub.add_parser("run-id", help="Generate a canonical Stage 2B run ID")

    cfg = sub.add_parser("config", help="Config helpers")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    p_val = cfg_sub.add_parser("validate", help="Validate a YAML config against defaults merge")
    p_val.add_argument("--config", type=Path, required=True)
    p_fp = cfg_sub.add_parser("fingerprint", help="Print resolved config fingerprint")
    p_fp.add_argument("--config", type=Path, required=True)
    p_fp.add_argument("--json", action="store_true")

    p_env = sub.add_parser("environment", help="Environment record helpers")
    env_sub = p_env.add_subparsers(dest="environment_command")
    p_show = env_sub.add_parser("show", help="Print secret-safe environment record (stdout only)")
    p_show.add_argument("--json", action="store_true")
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
    if args.command == "run-id":
        return cmd_run_id()
    if args.command == "config":
        if args.config_command == "validate":
            return cmd_config_validate(args.config)
        if args.config_command == "fingerprint":
            return cmd_config_fingerprint(args.config, as_json=bool(args.json))
        parser.parse_args(["config", "--help"])
        return 2
    if args.command == "environment":
        if args.environment_command == "show":
            return cmd_environment_show(as_json=bool(args.json))
        parser.parse_args(["environment", "--help"])
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
