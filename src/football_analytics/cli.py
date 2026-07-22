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


def cmd_contracts_list() -> int:
    from football_analytics.data.compiler import list_contracts
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
    for name in list_contracts(registry=reg):
        entry = reg.get_entry(name)
        print(f"{name}\tcurrent={entry.current_version}\tstatus={entry.status}")
    return 0


def cmd_contracts_show(name: str, version: int | None) -> int:
    from football_analytics.data import DataContractError
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    try:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        spec = reg.load_contract(name, version)
    except DataContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"contract: {spec.contract_name}")
    print(f"version: {spec.version}")
    print(f"fields: {len(spec.fields)}")
    print(f"primary_key: {','.join(spec.primary_key)}")
    print(f"description: {spec.description}")
    return 0


def cmd_contracts_fingerprint(name: str, version: int | None, *, as_json: bool) -> int:
    from football_analytics.data import DataContractError
    from football_analytics.data.fingerprint import contract_fingerprint
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    try:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        spec = reg.load_contract(name, version)
        digest = contract_fingerprint(spec)
    except DataContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = {"contract": name, "version": spec.version, "algorithm": "sha256", "digest": digest}
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"digest: {digest}")
    return 0


def cmd_contracts_validate(
    name: str, parquet_path: Path, version: int | None, *, as_json: bool
) -> int:
    from football_analytics.data import DataContractError
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )
    from football_analytics.data.validation import validate_table

    try:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        spec = reg.load_contract(name, version)
        table = read_contract_parquet(parquet_path, spec)
        result = validate_table(table, spec)
    except DataContractError as exc:
        if as_json:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False))
    else:
        print(f"status: {result.status}")
        for e in result.errors:
            print(f"error: {e}")
    return 0 if result.status != "FAIL" else 1


def cmd_contracts_migrate(
    name: str,
    source: Path,
    destination: Path,
    from_version: int,
    to_version: int,
) -> int:
    from football_analytics.data import DataContractError
    from football_analytics.data.migrations import migrate_parquet
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    receipt = destination.with_suffix(destination.suffix + ".migration_receipt.json")
    try:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        migrate_parquet(
            source,
            destination,
            registry=reg,
            contract=name,
            from_version=from_version,
            to_version=to_version,
            receipt_path=receipt,
            contain_root=destination.parent.resolve(),
        )
    except DataContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"migrated: {destination}")
    print(f"receipt: {receipt}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-analytics",
        description="Broadcast football video analytics pipeline (Stage 2C CLI)",
        epilog="Use 'football-analytics --version' for the package version.",
    )
    # Package --version is handled in main() before parse so it does not
    # collide with contracts ... --version N (integer).
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

    p_ct = sub.add_parser("contracts", help="Canonical Arrow/Parquet contract helpers")
    ct_sub = p_ct.add_subparsers(dest="contracts_command")
    ct_sub.add_parser("list", help="List registered contracts")
    p_show_c = ct_sub.add_parser("show", help="Show contract summary")
    p_show_c.add_argument("contract")
    p_show_c.add_argument("--version", type=int, default=None)
    p_fp_c = ct_sub.add_parser("fingerprint", help="Print contract fingerprint")
    p_fp_c.add_argument("contract")
    p_fp_c.add_argument("--version", type=int, default=None)
    p_fp_c.add_argument("--json", action="store_true")
    p_val_c = ct_sub.add_parser("validate", help="Validate a Parquet file against a contract")
    p_val_c.add_argument("contract")
    p_val_c.add_argument("parquet_path", type=Path)
    p_val_c.add_argument("--version", type=int, default=None)
    p_val_c.add_argument("--json", action="store_true")
    p_mig = ct_sub.add_parser("migrate", help="Migrate Parquet between contract versions")
    p_mig.add_argument("contract")
    p_mig.add_argument("source", type=Path)
    p_mig.add_argument("destination", type=Path)
    p_mig.add_argument("--from-version", type=int, required=True)
    p_mig.add_argument("--to-version", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    # Bare package version (must not steal contracts --version N).
    if raw == ["--version"]:
        from football_analytics import __version__

        print(__version__)
        return 0
    parser = build_parser()
    args = parser.parse_args(raw)
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
    if args.command == "contracts":
        if args.contracts_command == "list":
            return cmd_contracts_list()
        if args.contracts_command == "show":
            return cmd_contracts_show(args.contract, args.version)
        if args.contracts_command == "fingerprint":
            return cmd_contracts_fingerprint(args.contract, args.version, as_json=bool(args.json))
        if args.contracts_command == "validate":
            return cmd_contracts_validate(
                args.contract, args.parquet_path, args.version, as_json=bool(args.json)
            )
        if args.contracts_command == "migrate":
            return cmd_contracts_migrate(
                args.contract,
                args.source,
                args.destination,
                args.from_version,
                args.to_version,
            )
        parser.parse_args(["contracts", "--help"])
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
