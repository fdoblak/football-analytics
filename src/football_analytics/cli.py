"""CLI: version, info, foundation, contracts, and Stage 2D project/cache helpers."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections.abc import Sequence
from pathlib import Path

_CACHE_KEY_RE = re.compile(r"^[a-f0-9]{64}$")


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


def cmd_project_check(*, profile: str, deep: bool, as_json: bool) -> int:
    from football_analytics.pipeline.project_check import run_project_checks

    report = run_project_checks(profile=profile, mode="deep" if deep else "quick", strict=False)
    payload = report.to_dict()
    if as_json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))
    else:
        for check in report.checks:
            print(f"{check.status:4} {check.id}: {check.message}")
        print(f"overall={report.overall_status} exit_code={report.exit_code}")
    return int(report.exit_code)


def _load_cache_roots() -> tuple[Path, object]:
    from football_analytics.pipeline.cache import load_cache_policy, resolve_cache_root

    root = _project_root()
    policy = load_cache_policy(root / "configs" / "system" / "cache_policy.yaml")
    cache_root = resolve_cache_root(root / "configs" / "system" / "paths.yaml")
    return cache_root, policy


def cmd_cache_inspect(cache_key: str) -> int:
    from football_analytics.core.redaction import redact_value
    from football_analytics.pipeline.cache import entry_dir

    if not _CACHE_KEY_RE.fullmatch(cache_key):
        print("error: cache-key must be 64 lowercase hex", file=sys.stderr)
        return 2
    try:
        cache_root, policy = _load_cache_roots()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    entry = entry_dir(cache_root, cache_key)
    if not entry.is_dir():
        print(f"error: cache entry missing: {cache_key[:12]}…", file=sys.stderr)
        return 1
    max_bytes = int(getattr(policy, "max_manifest_bytes", 1_048_576))
    payload: dict[str, object] = {"cache_key": cache_key, "entry_exists": True}
    for name in ("cache_manifest.json", "stage_result.json"):
        path = entry / name
        if not path.is_file() or path.is_symlink():
            payload[name] = None
            continue
        if path.stat().st_size > max_bytes:
            print(f"error: {name} too large", file=sys.stderr)
            return 3
        try:
            payload[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"error: failed to read {name}: {type(exc).__name__}", file=sys.stderr)
            return 3
    print(json.dumps(redact_value(payload), sort_keys=True, ensure_ascii=False, indent=2))
    return 0


def cmd_cache_verify(cache_key: str) -> int:
    from football_analytics.pipeline.artifacts import verify_artifact_on_disk
    from football_analytics.pipeline.cache import entry_dir
    from football_analytics.pipeline.types import ArtifactRef

    if not _CACHE_KEY_RE.fullmatch(cache_key):
        print("error: cache-key must be 64 lowercase hex", file=sys.stderr)
        return 2
    try:
        cache_root, policy = _load_cache_roots()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    entry = entry_dir(cache_root, cache_key)
    manifest_path = entry / "cache_manifest.json"
    arts_dir = entry / "artifacts"
    if not entry.is_dir() or not manifest_path.is_file() or not arts_dir.is_dir():
        print("error: cache entry incomplete", file=sys.stderr)
        return 3
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"error: manifest unreadable: {type(exc).__name__}", file=sys.stderr)
        return 3
    if manifest.get("cache_key") != cache_key:
        print("error: manifest cache_key mismatch", file=sys.stderr)
        return 3
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        print("error: manifest artifacts invalid", file=sys.stderr)
        return 3
    reject_hl = bool(getattr(policy, "reject_hardlinks", True))
    try:
        for item in artifacts:
            if not isinstance(item, dict):
                raise ValueError("artifact entry not object")
            ref = ArtifactRef(
                logical_name=str(item["logical_name"]),
                relative_path=str(item["relative_path"]),
                media_type=str(item["media_type"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
                contract_name=item.get("contract_name"),
                contract_version=item.get("contract_version"),
                schema_fingerprint=item.get("schema_fingerprint"),
                metadata=item.get("metadata") or {},
            )
            verify_artifact_on_disk(ref, root=arts_dir, reject_hardlinks=reject_hl)
    except Exception as exc:  # noqa: BLE001
        print(f"error: verify failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"cache_key": cache_key, "status": "PASS"}, sort_keys=True))
    return 0


def cmd_video_probe(
    *,
    source: Path,
    output_dir: Path,
    policy_path: Path,
    contain_root: Path | None,
) -> int:
    """Stage 3B: safe local FFprobe media validation (lazy imports)."""
    from football_analytics.video.contracts import default_repo_root, load_ingest_policy
    from football_analytics.video.probe_service import run_media_probe

    root = default_repo_root()
    pol = policy_path if policy_path.is_absolute() else root / policy_path
    try:
        policy = load_ingest_policy(pol)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(policy["ffprobe_policy"]["runtime_root"]))
    result = run_media_probe(
        source=str(source),
        output_dir=str(output_dir),
        policy=policy,
        contain_root=contain,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"receipt_status: {summary['receipt_status']}")
    print(f"output_dir: {summary['output_dir']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_video_normalize(
    *,
    source: Path,
    output: Path,
    policy_path: Path,
    expected_source_sha256: str | None,
    execute: bool,
    contain_root: Path | None,
    receipt_dir: Path | None,
) -> int:
    """Stage 3C: safe local FFmpeg normalization (default dry-run)."""
    from football_analytics.video.contracts import default_repo_root, load_ingest_policy
    from football_analytics.video.normalization_service import run_video_normalization

    root = default_repo_root()
    pol = policy_path if policy_path.is_absolute() else root / policy_path
    try:
        policy = load_ingest_policy(pol)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    if execute and not expected_source_sha256:
        print("error: --execute requires --expected-source-sha256", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(policy["ffmpeg_policy"]["runtime_root"]))
    result = run_video_normalization(
        source=str(source),
        output=str(output),
        policy=policy,
        expected_source_sha256=expected_source_sha256,
        execute=execute,
        contain_root=contain,
        receipt_dir=str(receipt_dir) if receipt_dir is not None else None,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"status: {summary['status']}")
    print(f"output_path: {summary['output_path']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_video_frames(
    *,
    source: Path,
    output_dir: Path,
    mode: str,
    policy_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
    expected_source_sha256: str | None,
    execute_materialize: bool,
    sample_every: int | None,
    normalization_receipt: Path | None,
) -> int:
    """Stage 3D: streaming frame timeline (+ optional materialize)."""
    from football_analytics.video.contracts import default_repo_root, load_ingest_policy
    from football_analytics.video.frame_timeline_service import run_frame_timeline
    from football_analytics.video.types import FrameTimelineMode

    root = default_repo_root()
    pol = policy_path if policy_path.is_absolute() else root / policy_path
    try:
        policy = load_ingest_policy(pol)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    try:
        timeline_mode = FrameTimelineMode(mode)
    except ValueError:
        print(f"error: invalid mode {mode!r}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(policy["frame_timeline_policy"]["runtime_root"]))
    result = run_frame_timeline(
        source=str(source),
        output_dir=str(output_dir),
        policy=policy,
        mode=timeline_mode,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
        expected_source_sha256=expected_source_sha256,
        execute_materialize=execute_materialize,
        sample_every=sample_every,
        normalization_receipt=(
            str(normalization_receipt) if normalization_receipt is not None else None
        ),
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"status: {summary['status']}")
    print(f"frames_parquet: {summary['frames_parquet']}")
    print(f"mapping_quality: {summary['mapping_quality']}")
    print(f"frame_count: {summary['frame_count']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_broadcast_shots_detect(
    *,
    source: Path,
    timeline: Path,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 4B: shot boundary detection baseline (lazy imports)."""
    from football_analytics.broadcast.shot_config import (
        default_shot_config_path,
        load_shot_boundary_config,
    )
    from football_analytics.broadcast.shot_service import run_shot_boundary_detection
    from football_analytics.data.registry import default_project_root

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_shot_config_path(repo_root=root)
    try:
        config = load_shot_boundary_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_shot_boundary_detection(
        source=str(source),
        timeline=str(timeline),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"boundary_count: {summary['boundary_count']}")
    print(f"segment_count: {summary['segment_count']}")
    print(f"boundaries_parquet: {summary['boundaries_parquet']}")
    print(f"segments_parquet: {summary['segments_parquet']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_broadcast_shots_evaluate(
    *,
    predictions: Path,
    ground_truth: Path,
    output: Path,
    config_path: Path,
    tolerance_us: int | None,
) -> int:
    """Stage 4B: evaluate predicted boundaries vs ground truth."""
    import json

    from football_analytics.broadcast.contracts import load_broadcast_contract
    from football_analytics.broadcast.shot_config import load_shot_boundary_config
    from football_analytics.broadcast.shot_evaluation import evaluate_from_rows
    from football_analytics.core.records import write_json_record
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import default_project_root

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    try:
        config = load_shot_boundary_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    tol = (
        int(tolerance_us)
        if tolerance_us is not None
        else int(config["evaluation"]["matching_tolerance_us"])
    )
    try:
        pred_table = read_contract_parquet(predictions, load_broadcast_contract("shot_boundaries"))
        pred_rows = pred_table.to_pylist()
        gt_path = Path(ground_truth)
        if gt_path.suffix.lower() == ".json":
            gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
            gt_rows = list(gt_payload.get("boundaries") or gt_payload.get("ground_truth") or [])
            duration_us = gt_payload.get("duration_us")
        else:
            gt_table = read_contract_parquet(gt_path, load_broadcast_contract("shot_boundaries"))
            gt_rows = gt_table.to_pylist()
            duration_us = None
        metrics = evaluate_from_rows(pred_rows, gt_rows, tolerance_us=tol, duration_us=duration_us)
        write_json_record(output, metrics.to_dict(), overwrite=False)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"f1: {metrics.f1}")
    print(f"precision: {metrics.precision}")
    print(f"recall: {metrics.recall}")
    print(f"true_positives: {metrics.true_positives}")
    print(f"false_positives: {metrics.false_positives}")
    print(f"false_negatives: {metrics.false_negatives}")
    print(f"output: {output}")
    return 0


def cmd_broadcast_camera_classify(
    *,
    source: Path,
    timeline: Path,
    shots: Path,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 4C: camera-view classification baseline (lazy imports)."""
    from football_analytics.broadcast.camera_config import (
        default_camera_config_path,
        load_camera_view_config,
    )
    from football_analytics.broadcast.camera_service import run_camera_view_classification
    from football_analytics.data.registry import default_project_root

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_camera_config_path(repo_root=root)
    try:
        config = load_camera_view_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_camera_view_classification(
        source=str(source),
        timeline=str(timeline),
        shots=str(shots),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"segment_count: {summary['segment_count']}")
    print(f"cameras_parquet: {summary['cameras_parquet']}")
    print(f"classification_receipt: {summary['classification_receipt']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_broadcast_integrate(
    *,
    timeline: Path,
    boundaries: Path,
    shots: Path,
    camera_views: Path,
    output_dir: Path,
    policy_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 4D: fuse shot/camera segments and route analysis windows."""
    from football_analytics.broadcast.broadcast_pipeline import run_broadcast_integrate
    from football_analytics.broadcast.playability import (
        default_routing_policy_path,
        load_routing_policy,
    )
    from football_analytics.data.registry import default_project_root

    root = default_project_root()
    pol_path = policy_path if policy_path.is_absolute() else root / policy_path
    if not pol_path.is_file():
        pol_path = default_routing_policy_path(repo_root=root)
    try:
        policy = load_routing_policy(pol_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(policy["runtime_root"]))
    result = run_broadcast_integrate(
        timeline=str(timeline),
        boundaries=str(boundaries),
        shots=str(shots),
        camera_views=str(camera_views),
        output_dir=str(output_dir),
        policy=policy,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"window_count: {summary['window_count']}")
    print(f"review_count: {summary['review_count']}")
    print(f"analysis_windows_parquet: {summary['analysis_windows_parquet']}")
    print(f"review_queue_json: {summary['review_queue_json']}")
    print(f"pipeline_receipt_json: {summary['pipeline_receipt_json']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_broadcast_camera_evaluate(
    *,
    predictions: Path,
    ground_truth: Path,
    output: Path,
    config_path: Path,
) -> int:
    """Stage 4C: evaluate predicted camera views vs ground truth."""
    import json

    from football_analytics.broadcast.camera_config import load_camera_view_config
    from football_analytics.broadcast.camera_evaluation import (
        combined_view_framing_macro_f1,
        evaluate_camera_predictions,
    )
    from football_analytics.broadcast.contracts import load_broadcast_contract
    from football_analytics.core.records import write_json_record
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import default_project_root

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    try:
        config = load_camera_view_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    try:
        pred_table = read_contract_parquet(
            predictions, load_broadcast_contract("camera_view_segments")
        )
        pred_rows = pred_table.to_pylist()
        gt_path = Path(ground_truth)
        if gt_path.suffix.lower() == ".json":
            gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
            if isinstance(gt_payload, list):
                gt_rows = list(gt_payload)
            else:
                gt_rows = list(
                    gt_payload.get("segments")
                    or gt_payload.get("ground_truth")
                    or gt_payload.get("labels")
                    or []
                )
                if not gt_rows and any(
                    k in gt_payload for k in ("view_family", "fixture_id", "name")
                ):
                    gt_rows = [gt_payload]
        else:
            gt_table = read_contract_parquet(
                gt_path, load_broadcast_contract("camera_view_segments")
            )
            gt_rows = gt_table.to_pylist()
        supported = {
            "view_family": list(config["supported_axes"]["view_family"]),
            "framing_scale": list(config["supported_axes"]["framing_scale"]),
            "camera_motion": list(config["supported_axes"]["camera_motion"]),
            "graphics_status": list(config["supported_axes"]["graphics_status"]),
            "playability": list(config["supported_axes"]["playability"]),
        }
        report = evaluate_camera_predictions(pred_rows, gt_rows, supported_labels=supported)
        payload = report.to_dict()
        payload["view_framing_macro_f1"] = combined_view_framing_macro_f1(report)
        write_json_record(output, payload, overwrite=False)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"status: {report.status}")
    print(f"n_pairs: {report.n_pairs}")
    print(f"view_framing_macro_f1: {payload.get('view_framing_macro_f1')}")
    print(f"graphics_macro_f1: {report.axes['graphics_status'].macro_f1}")
    print(f"motion_macro_f1: {report.axes['camera_motion'].macro_f1}")
    print(f"playability_macro_f1: {report.axes['playability'].macro_f1}")
    print(f"unsafe_playable_fp_rate: {report.unsafe_playable_false_positive_rate}")
    print(f"output: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-analytics",
        description="Broadcast football video analytics pipeline (Stage 2D CLI)",
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

    p_proj = sub.add_parser("project", help="Project validation helpers")
    proj_sub = p_proj.add_subparsers(dest="project_command")
    p_check = proj_sub.add_parser("check", help="Run unified project checks")
    p_check.add_argument("--profile", choices=("local", "ci"), default="local")
    depth = p_check.add_mutually_exclusive_group()
    depth.add_argument("--quick", action="store_true", default=False)
    depth.add_argument("--deep", action="store_true", default=False)
    p_check.add_argument("--json", action="store_true")

    p_cache = sub.add_parser("cache", help="Read-only cache inspect/verify")
    cache_sub = p_cache.add_subparsers(dest="cache_command")
    p_ins = cache_sub.add_parser("inspect", help="Print secret-safe cache entry JSON")
    p_ins.add_argument("cache_key")
    p_ver = cache_sub.add_parser("verify", help="Verify cache entry artifacts (read-only)")
    p_ver.add_argument("cache_key")

    p_video = sub.add_parser("video", help="Video ingest/probe helpers (Stage 3)")
    video_sub = p_video.add_subparsers(dest="video_command")
    p_probe = video_sub.add_parser("probe", help="Safe local FFprobe media validation")
    p_probe.add_argument("--source", type=Path, required=True, help="Absolute local video path")
    p_probe.add_argument("--output-dir", type=Path, required=True, help="Runtime output directory")
    p_probe.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/video/ingest_policy.yaml"),
        help="Ingest/ffprobe policy YAML",
    )
    p_probe.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: ffprobe_policy.runtime_root)",
    )
    p_norm = video_sub.add_parser("normalize", help="Safe local FFmpeg video normalization")
    p_norm.add_argument("--source", type=Path, required=True, help="Absolute local video path")
    p_norm.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Absolute normalized output path",
    )
    p_norm.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/video/ingest_policy.yaml"),
        help="Ingest/ffmpeg policy YAML",
    )
    p_norm.add_argument(
        "--expected-source-sha256",
        type=str,
        default=None,
        help="Expected source SHA-256 (required with --execute)",
    )
    p_norm.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute normalization (default: dry-run plan/skip receipt only)",
    )
    p_norm.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: ffmpeg_policy.runtime_root)",
    )
    p_norm.add_argument(
        "--receipt-dir",
        type=Path,
        default=None,
        help="Optional directory for normalization_receipt.json",
    )
    p_frames = video_sub.add_parser("frames", help="Frame timeline mapping (Stage 3D)")
    p_frames.add_argument("--source", type=Path, required=True, help="Absolute local video path")
    p_frames.add_argument("--output-dir", type=Path, required=True, help="Runtime output directory")
    p_frames.add_argument(
        "--mode",
        choices=("timeline_only", "sampled", "all_frames"),
        default="timeline_only",
        help="timeline_only (default) or materialize modes",
    )
    p_frames.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/video/ingest_policy.yaml"),
        help="Ingest/frame_timeline policy YAML",
    )
    p_frames.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: frame_timeline_policy.runtime_root)",
    )
    p_frames.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_frames.add_argument("--video-id", type=str, default=None, help="Optional video_id")
    p_frames.add_argument(
        "--expected-source-sha256",
        type=str,
        default=None,
        help="Optional expected source SHA-256",
    )
    p_frames.add_argument(
        "--execute-materialize",
        action="store_true",
        default=False,
        help="Required for sampled/all_frames image materialization",
    )
    p_frames.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Sample stride for sampled mode (default from policy)",
    )
    p_frames.add_argument(
        "--normalization-receipt",
        type=Path,
        default=None,
        help="Optional Stage 3C normalization receipt JSON",
    )

    p_broadcast = sub.add_parser("broadcast", help="Broadcast shot/camera helpers (Stage 4)")
    broadcast_sub = p_broadcast.add_subparsers(dest="broadcast_command")
    p_shots = broadcast_sub.add_parser("shots", help="Shot boundary detection / evaluation")
    shots_sub = p_shots.add_subparsers(dest="shots_command")
    p_detect = shots_sub.add_parser("detect", help="Detect shot boundaries (baseline)")
    p_detect.add_argument("--source", type=Path, required=True, help="Absolute local video path")
    p_detect.add_argument(
        "--timeline", type=Path, required=True, help="Absolute frames.parquet path"
    )
    p_detect.add_argument("--output-dir", type=Path, required=True, help="Runtime output directory")
    p_detect.add_argument(
        "--config",
        type=Path,
        default=Path("configs/broadcast/shot_boundary_baseline.yaml"),
        help="Shot boundary baseline config YAML",
    )
    p_detect.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: config.runtime_root)",
    )
    p_detect.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_detect.add_argument("--video-id", type=str, default=None, help="Optional video_id")
    p_eval = shots_sub.add_parser("evaluate", help="Evaluate predicted boundaries vs ground truth")
    p_eval.add_argument(
        "--predictions", type=Path, required=True, help="Predicted shot_boundaries.parquet"
    )
    p_eval.add_argument(
        "--ground-truth", type=Path, required=True, help="Ground-truth JSON or parquet"
    )
    p_eval.add_argument("--output", type=Path, required=True, help="Metrics JSON output path")
    p_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/broadcast/shot_boundary_baseline.yaml"),
        help="Config for default matching tolerance",
    )
    p_eval.add_argument(
        "--tolerance-us",
        type=int,
        default=None,
        help="Override matching tolerance microseconds",
    )
    p_camera = broadcast_sub.add_parser("camera", help="Camera-view classification / evaluation")
    camera_sub = p_camera.add_subparsers(dest="camera_command")
    p_classify = camera_sub.add_parser("classify", help="Classify camera views (baseline)")
    p_classify.add_argument("--source", type=Path, required=True, help="Absolute local video path")
    p_classify.add_argument(
        "--timeline", type=Path, required=True, help="Absolute frames.parquet path"
    )
    p_classify.add_argument(
        "--shots", type=Path, required=True, help="Absolute shot_segments.parquet path"
    )
    p_classify.add_argument(
        "--output-dir", type=Path, required=True, help="Runtime output directory"
    )
    p_classify.add_argument(
        "--config",
        type=Path,
        default=Path("configs/broadcast/camera_view_baseline.yaml"),
        help="Camera-view baseline config YAML",
    )
    p_classify.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: config.runtime_root)",
    )
    p_classify.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_classify.add_argument("--video-id", type=str, default=None, help="Optional video_id")
    p_cam_eval = camera_sub.add_parser(
        "evaluate", help="Evaluate predicted camera views vs ground truth"
    )
    p_cam_eval.add_argument(
        "--predictions", type=Path, required=True, help="Predicted camera_view_segments.parquet"
    )
    p_cam_eval.add_argument(
        "--ground-truth", type=Path, required=True, help="Ground-truth JSON or parquet"
    )
    p_cam_eval.add_argument("--output", type=Path, required=True, help="Metrics JSON output path")
    p_cam_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/broadcast/camera_view_baseline.yaml"),
        help="Camera-view baseline config YAML",
    )
    p_integrate = broadcast_sub.add_parser(
        "integrate", help="Fuse shot/camera segments and route analysis windows"
    )
    p_integrate.add_argument(
        "--timeline", type=Path, required=True, help="Absolute frames.parquet path"
    )
    p_integrate.add_argument(
        "--boundaries", type=Path, required=True, help="Absolute shot_boundaries.parquet path"
    )
    p_integrate.add_argument(
        "--shots", type=Path, required=True, help="Absolute shot_segments.parquet path"
    )
    p_integrate.add_argument(
        "--camera-views",
        type=Path,
        required=True,
        help="Absolute camera_view_segments.parquet path",
    )
    p_integrate.add_argument(
        "--output-dir", type=Path, required=True, help="Runtime output directory"
    )
    p_integrate.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/broadcast/broadcast_routing_policy.yaml"),
        help="Broadcast routing policy YAML",
    )
    p_integrate.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: policy.runtime_root)",
    )
    p_integrate.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_integrate.add_argument("--video-id", type=str, default=None, help="Optional video_id")
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
    if args.command == "project":
        if args.project_command == "check":
            return cmd_project_check(
                profile=str(args.profile),
                deep=bool(args.deep),
                as_json=bool(args.json),
            )
        parser.parse_args(["project", "--help"])
        return 2
    if args.command == "cache":
        if args.cache_command == "inspect":
            return cmd_cache_inspect(str(args.cache_key))
        if args.cache_command == "verify":
            return cmd_cache_verify(str(args.cache_key))
        parser.parse_args(["cache", "--help"])
        return 2
    if args.command == "video":
        if args.video_command == "probe":
            return cmd_video_probe(
                source=args.source,
                output_dir=args.output_dir,
                policy_path=args.policy,
                contain_root=args.contain_root,
            )
        if args.video_command == "normalize":
            return cmd_video_normalize(
                source=args.source,
                output=args.output,
                policy_path=args.policy,
                expected_source_sha256=args.expected_source_sha256,
                execute=bool(args.execute),
                contain_root=args.contain_root,
                receipt_dir=args.receipt_dir,
            )
        if args.video_command == "frames":
            return cmd_video_frames(
                source=args.source,
                output_dir=args.output_dir,
                mode=str(args.mode),
                policy_path=args.policy,
                contain_root=args.contain_root,
                run_id=args.run_id,
                video_id=args.video_id,
                expected_source_sha256=args.expected_source_sha256,
                execute_materialize=bool(args.execute_materialize),
                sample_every=args.sample_every,
                normalization_receipt=args.normalization_receipt,
            )
        parser.parse_args(["video", "--help"])
        return 2
    if args.command == "broadcast":
        if args.broadcast_command == "shots":
            if args.shots_command == "detect":
                return cmd_broadcast_shots_detect(
                    source=args.source,
                    timeline=args.timeline,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                )
            if args.shots_command == "evaluate":
                return cmd_broadcast_shots_evaluate(
                    predictions=args.predictions,
                    ground_truth=args.ground_truth,
                    output=args.output,
                    config_path=args.config,
                    tolerance_us=args.tolerance_us,
                )
            parser.parse_args(["broadcast", "shots", "--help"])
            return 2
        if args.broadcast_command == "camera":
            if args.camera_command == "classify":
                return cmd_broadcast_camera_classify(
                    source=args.source,
                    timeline=args.timeline,
                    shots=args.shots,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                )
            if args.camera_command == "evaluate":
                return cmd_broadcast_camera_evaluate(
                    predictions=args.predictions,
                    ground_truth=args.ground_truth,
                    output=args.output,
                    config_path=args.config,
                )
            parser.parse_args(["broadcast", "camera", "--help"])
            return 2
        if args.broadcast_command == "integrate":
            return cmd_broadcast_integrate(
                timeline=args.timeline,
                boundaries=args.boundaries,
                shots=args.shots,
                camera_views=args.camera_views,
                output_dir=args.output_dir,
                policy_path=args.policy,
                contain_root=args.contain_root,
                run_id=args.run_id,
                video_id=args.video_id,
            )
        parser.parse_args(["broadcast", "--help"])
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
