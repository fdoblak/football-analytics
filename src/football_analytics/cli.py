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


def cmd_perception_humans_detect(
    *,
    source: Path,
    timeline: Path,
    analysis_windows: Path,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 5B: human detection baseline (lazy imports)."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.detection_service import run_human_detection
    from football_analytics.perception.human_detector_config import (
        default_human_detector_config_path,
        load_human_detector_config,
    )

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_human_detector_config_path(repo_root=root)
    try:
        config = load_human_detector_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_human_detection(
        source=str(source),
        timeline=str(timeline),
        analysis_windows=str(analysis_windows),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
        project_root=root,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"detection_count: {summary['detection_count']}")
    print(f"human_detection_count: {summary['human_detection_count']}")
    print(f"ball_detection_count: {summary['ball_detection_count']}")
    print(f"detections_parquet: {summary['detections_parquet']}")
    print(f"frame_status_parquet: {summary['frame_status_parquet']}")
    print(f"attributes_parquet: {summary['attributes_parquet']}")
    print(f"receipt_json: {summary['receipt_json']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_perception_humans_evaluate(
    *,
    predictions: Path,
    ground_truth: Path,
    output: Path,
    config_path: Path,
) -> int:
    """Stage 5B: evaluate predicted human boxes vs ground truth."""
    import json

    from football_analytics.core.records import write_json_record
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.detection_evaluation import evaluate_from_rows
    from football_analytics.perception.human_detector_config import load_human_detector_config

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    try:
        config = load_human_detector_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    try:
        pred_table = read_contract_parquet(predictions, get_contract("detections", 1))
        pred_rows = pred_table.to_pylist()
        # Attach entity_type=human for Stage 5B preds when attributes absent.
        for r in pred_rows:
            r.setdefault("entity_type", "human")
        gt_path = Path(ground_truth)
        if gt_path.suffix.lower() == ".json":
            gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
            gt_rows = list(gt_payload.get("detections") or gt_payload.get("ground_truth") or [])
        else:
            gt_table = read_contract_parquet(gt_path, get_contract("detections", 1))
            gt_rows = gt_table.to_pylist()
            for r in gt_rows:
                r.setdefault("entity_type", "human")
                r.setdefault("is_reviewed_ground_truth", True)
        metrics = evaluate_from_rows(
            pred_rows,
            gt_rows,
            iou_threshold=0.5,
            iou_thresholds=list(config["evaluation_iou_thresholds"]),
        )
        write_json_record(output, metrics.to_dict(), overwrite=False)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"status: {metrics.status}")
    print(f"f1: {metrics.f1}")
    print(f"precision: {metrics.precision}")
    print(f"recall: {metrics.recall}")
    print(f"ap50: {metrics.ap50}")
    print(f"output: {output}")
    return 0


def cmd_perception_ball_detect(
    *,
    source: Path,
    timeline: Path,
    analysis_windows: Path,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 5C: ball detection baseline (lazy imports)."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.ball_detector_config import (
        default_ball_detector_config_path,
        load_ball_detector_config,
    )
    from football_analytics.perception.ball_service import run_ball_detection

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_ball_detector_config_path(repo_root=root)
    try:
        config = load_ball_detector_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_ball_detection(
        source=str(source),
        timeline=str(timeline),
        analysis_windows=str(analysis_windows),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
        project_root=root,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"detection_count: {summary['detection_count']}")
    print(f"human_detection_count: {summary['human_detection_count']}")
    print(f"ball_detection_count: {summary['ball_detection_count']}")
    print(f"detections_parquet: {summary['detections_parquet']}")
    print(f"frame_status_parquet: {summary['frame_status_parquet']}")
    print(f"attributes_parquet: {summary['attributes_parquet']}")
    print(f"receipt_json: {summary['receipt_json']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_perception_ball_evaluate(
    *,
    predictions: Path,
    ground_truth: Path,
    output: Path,
    config_path: Path,
) -> int:
    """Stage 5C: evaluate predicted ball boxes vs ground truth."""
    import json

    from football_analytics.core.records import write_json_record
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.ball_detector_config import load_ball_detector_config
    from football_analytics.perception.ball_evaluation import evaluate_ball_from_rows

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    try:
        config = load_ball_detector_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    try:
        pred_table = read_contract_parquet(predictions, get_contract("detections", 1))
        pred_rows = pred_table.to_pylist()
        for r in pred_rows:
            r.setdefault("entity_type", "ball")
        gt_path = Path(ground_truth)
        if gt_path.suffix.lower() == ".json":
            gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
            gt_rows = list(gt_payload.get("detections") or gt_payload.get("ground_truth") or [])
        else:
            gt_table = read_contract_parquet(gt_path, get_contract("detections", 1))
            gt_rows = gt_table.to_pylist()
            for r in gt_rows:
                r.setdefault("entity_type", "ball")
                r.setdefault("is_reviewed_ground_truth", True)
        metrics = evaluate_ball_from_rows(
            pred_rows,
            gt_rows,
            iou_threshold=0.5,
            iou_thresholds=list(config["evaluation_iou_thresholds"]),
        )
        write_json_record(output, metrics.to_dict(), overwrite=False)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"status: {metrics.status}")
    print(f"f1: {metrics.f1}")
    print(f"precision: {metrics.precision}")
    print(f"recall: {metrics.recall}")
    print(f"ap50: {metrics.ap50}")
    print(f"output: {output}")
    return 0


def cmd_perception_roles_classify(
    *,
    detections: Path,
    detection_attributes: Path,
    output_dir: Path,
    config_path: Path,
    detection_frame_status: Path | None,
    analysis_windows: Path | None,
    source: Path | None,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
    ground_truth: Path | None,
) -> int:
    """Stage 5D: conservative human role classification (lazy imports)."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.role_config import (
        default_human_role_config_path,
        load_human_role_config,
    )
    from football_analytics.perception.role_service import run_human_role_classification

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_human_role_config_path(repo_root=root)
    try:
        config = load_human_role_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_human_role_classification(
        detections=str(detections),
        detection_attributes=str(detection_attributes),
        detection_frame_status=(
            None if detection_frame_status is None else str(detection_frame_status)
        ),
        analysis_windows=None if analysis_windows is None else str(analysis_windows),
        source=None if source is None else str(source),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
        ground_truth=None if ground_truth is None else str(ground_truth),
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"config_fingerprint: {summary['config_fingerprint']}")
    if summary.get("attributes_parquet"):
        print(f"attributes_parquet: {summary['attributes_parquet']}")
    if summary.get("receipt_json"):
        print(f"receipt_json: {summary['receipt_json']}")
    if summary.get("evaluation_json"):
        print(f"evaluation_json: {summary['evaluation_json']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_perception_integrate(
    *,
    human_detections: Path,
    human_frame_status: Path,
    human_attributes: Path,
    human_receipt: Path,
    ball_detections: Path,
    ball_frame_status: Path,
    ball_attributes: Path,
    ball_receipt: Path,
    role_attributes: Path,
    role_receipt: Path,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    analysis_windows: Path | None,
    frames: Path | None,
    run_id: str | None,
    video_id: str | None,
    source_sha: str | None,
    timeline_fingerprint: str | None,
) -> int:
    """Stage 5E: fuse human/ball/role detection artifacts into one bundle."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.detection_pipeline import run_detection_integrate
    from football_analytics.perception.detection_pipeline_config import (
        default_detection_pipeline_config_path,
        load_detection_pipeline_config,
    )

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_detection_pipeline_config_path(repo_root=root)
    try:
        config = load_detection_pipeline_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_detection_integrate(
        human_detections=str(human_detections),
        human_frame_status=str(human_frame_status),
        human_attributes=str(human_attributes),
        human_receipt=str(human_receipt),
        ball_detections=str(ball_detections),
        ball_frame_status=str(ball_frame_status),
        ball_attributes=str(ball_attributes),
        ball_receipt=str(ball_receipt),
        role_attributes=str(role_attributes),
        role_receipt=str(role_receipt),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        analysis_windows=str(analysis_windows) if analysis_windows else None,
        frames=str(frames) if frames else None,
        run_id=run_id,
        video_id=video_id,
        expected_source_sha=source_sha,
        expected_timeline_fp=timeline_fingerprint,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"total_detection_count: {summary['total_detection_count']}")
    print(f"quality_status: {summary['quality_status']}")
    print(f"review_count: {summary['review_count']}")
    print(f"detections_parquet: {summary['detections_parquet']}")
    print(f"pipeline_receipt_json: {summary['pipeline_receipt_json']}")
    print(f"quality_report_json: {summary['quality_report_json']}")
    print(f"review_queue_json: {summary['review_queue_json']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_identity_contracts_validate(*, keep: bool, as_json: bool) -> int:
    """Run Stage 7A synthetic identity contract validator (no ReID inference)."""
    import runpy

    script = _project_root() / "scripts" / "check_identity_contracts.py"
    argv: list[str] = []
    if keep:
        argv.append("--keep")
    if as_json:
        argv.append("--json")
    ns = runpy.run_path(str(script), run_name="__not_main__")
    main_fn = ns.get("main")
    if not callable(main_fn):
        print("identity validator missing main()", file=sys.stderr)
        return 2
    return int(main_fn(argv))


def cmd_identity_target_validate(request_path: Path) -> int:
    """Validate target_player_request JSON against Stage 7A schema."""
    from football_analytics.identity.target_profile import validate_target_player_request

    if not request_path.is_file() or request_path.is_symlink():
        print(f"target request missing or symlink: {request_path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print("target request root must be object", file=sys.stderr)
            return 1
        validate_target_player_request(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"target_invalid: {exc}", file=sys.stderr)
        return 1
    print("target_valid: true")
    return 0


def cmd_identity_receipt_validate(receipt_path: Path) -> int:
    """Validate identity_run_receipt JSON against schema (no identity run)."""
    from football_analytics.identity.receipt import validate_receipt_payload

    if not receipt_path.is_file() or receipt_path.is_symlink():
        print(f"receipt missing or symlink: {receipt_path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt_payload(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"receipt_invalid: {exc}", file=sys.stderr)
        return 1
    print("receipt_valid: true")
    return 0


def cmd_identity_appearance_extract(
    *,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
    fixture: str,
) -> int:
    """Stage 7B: extract tracklet appearance profiles (synthetic fixture path)."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.identity.appearance_reid_config import (
        default_appearance_reid_config_path,
        load_appearance_reid_config,
    )
    from football_analytics.identity.appearance_reid_fixtures import all_core_fixtures
    from football_analytics.identity.appearance_reid_service import run_appearance_extract

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_appearance_reid_config_path(repo_root=root)
    try:
        config = load_appearance_reid_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    fixtures = all_core_fixtures()
    if fixture not in fixtures:
        print(f"unknown_fixture: {fixture}; choices={sorted(fixtures)}", file=sys.stderr)
        return 2
    bundle = fixtures[fixture]()
    contain = contain_root or Path(str(config["runtime_root"]))
    result = run_appearance_extract(
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
        in_memory_bundle=bundle,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"profiles_parquet: {summary.get('profiles_parquet')}")
    print(f"receipt_json: {summary.get('receipt_json')}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_identity_reid_candidates(
    *,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
    fixture: str,
) -> int:
    """Stage 7B: propose ReID candidate links + appearance evidence."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.identity.appearance_reid_config import (
        default_appearance_reid_config_path,
        load_appearance_reid_config,
    )
    from football_analytics.identity.appearance_reid_fixtures import all_core_fixtures
    from football_analytics.identity.appearance_reid_service import run_reid_candidates

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_appearance_reid_config_path(repo_root=root)
    try:
        config = load_appearance_reid_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    fixtures = all_core_fixtures()
    if fixture not in fixtures:
        print(f"unknown_fixture: {fixture}; choices={sorted(fixtures)}", file=sys.stderr)
        return 2
    bundle = fixtures[fixture]()
    contain = contain_root or Path(str(config["runtime_root"]))
    result = run_reid_candidates(
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
        in_memory_bundle=bundle,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"evidence_parquet: {summary.get('evidence_parquet')}")
    print(f"links_parquet: {summary.get('links_parquet')}")
    print(f"evaluation_json: {summary.get('evaluation_json')}")
    print(f"evaluation_status: {summary.get('evaluation_status')}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_identity_reid_evaluate(
    *,
    config_path: Path,
    links: Path | None,
    profiles: Path | None,
    ground_truth: Path | None,
) -> int:
    """Stage 7B: evaluate appearance ReID (NOT_EVALUATED without reviewed GT)."""
    import pyarrow.parquet as pq

    from football_analytics.data.registry import default_project_root
    from football_analytics.identity.appearance_reid_config import (
        appearance_reid_config_fingerprint,
        default_appearance_reid_config_path,
        load_appearance_reid_config,
    )
    from football_analytics.identity.appearance_reid_evaluation import NOT_EVALUATED_APPEARANCE_REID
    from football_analytics.identity.appearance_reid_service import run_reid_evaluate

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_appearance_reid_config_path(repo_root=root)
    try:
        config = load_appearance_reid_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    link_rows = None
    profile_rows = None
    if links is not None and links.is_file():
        link_rows = pq.read_table(links).to_pylist()
    if profiles is not None and profiles.is_file():
        profile_rows = pq.read_table(profiles).to_pylist()
    gt_rows = None
    has_gt = False
    if ground_truth is not None and ground_truth.is_file():
        gt_rows = json.loads(ground_truth.read_text(encoding="utf-8"))
        has_gt = bool(gt_rows)
    payload = run_reid_evaluate(
        links=link_rows,
        profiles=profile_rows,
        ground_truth=gt_rows if isinstance(gt_rows, list) else None,
        has_reviewed_ground_truth=has_gt,
        config_fingerprint=appearance_reid_config_fingerprint(config),
    )
    print(f"status: {payload['status']}")
    print(f"ground_truth_evaluation_status: {payload['ground_truth_evaluation_status']}")
    print(f"expected_code: {NOT_EVALUATED_APPEARANCE_REID}")
    return 0


def cmd_identity_appearance_validate(*, keep: bool, as_json: bool) -> int:
    """Run Stage 7B appearance ReID baseline validator."""
    import runpy

    script = _project_root() / "scripts" / "check_appearance_reid_baseline.py"
    argv: list[str] = []
    if keep:
        argv.append("--keep")
    if as_json:
        argv.append("--json")
    ns = runpy.run_path(str(script), run_name="__not_main__")
    main_fn = ns.get("main")
    if not callable(main_fn):
        print("appearance reid validator missing main()", file=sys.stderr)
        return 2
    return int(main_fn(argv))


def cmd_tracking_contracts_validate(*, keep: bool, as_json: bool) -> int:
    """Run Stage 6A synthetic tracking contract validator (no tracker algorithm)."""
    import runpy

    script = _project_root() / "scripts" / "check_tracking_contracts.py"
    argv: list[str] = []
    if keep:
        argv.append("--keep")
    if as_json:
        argv.append("--json")
    ns = runpy.run_path(str(script), run_name="__not_main__")
    main_fn = ns.get("main")
    if not callable(main_fn):
        print("tracking validator missing main()", file=sys.stderr)
        return 2
    return int(main_fn(argv))


def cmd_tracking_integrate(
    *,
    detections: Path,
    detection_attributes: Path,
    detection_receipt: Path,
    human_observations: Path,
    human_summaries: Path,
    human_lifecycle: Path,
    human_receipt: Path,
    ball_observations: Path,
    ball_summaries: Path,
    ball_lifecycle: Path,
    ball_receipt: Path,
    output_dir: Path,
    config_path: Path,
    contain_root: Path | None,
    frames: Path | None,
    analysis_windows: Path | None,
    ball_primary_sidecar: Path | None,
    run_id: str | None,
    video_id: str | None,
    source_sha: str | None,
    timeline_fingerprint: str | None,
    detection_fingerprint: str | None,
    analysis_window_fingerprint: str | None,
) -> int:
    """Stage 6D: fuse human/ball tracking artifacts into one bundle."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.tracking.tracking_pipeline import run_tracking_integrate
    from football_analytics.tracking.tracking_pipeline_config import (
        default_tracking_pipeline_config_path,
        load_tracking_pipeline_config,
    )

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_tracking_pipeline_config_path(repo_root=root)
    try:
        config = load_tracking_pipeline_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root
    if contain is None:
        contain = Path(str(config["runtime_root"]))
    result = run_tracking_integrate(
        detections=str(detections),
        detection_attributes=str(detection_attributes),
        detection_receipt=str(detection_receipt),
        human_observations=str(human_observations),
        human_summaries=str(human_summaries),
        human_lifecycle=str(human_lifecycle),
        human_receipt=str(human_receipt),
        ball_observations=str(ball_observations),
        ball_summaries=str(ball_summaries),
        ball_lifecycle=str(ball_lifecycle),
        ball_receipt=str(ball_receipt),
        output_dir=str(output_dir),
        config=config,
        contain_root=contain,
        frames=str(frames) if frames else None,
        analysis_windows=str(analysis_windows) if analysis_windows else None,
        ball_primary_sidecar=str(ball_primary_sidecar) if ball_primary_sidecar else None,
        run_id=run_id,
        video_id=video_id,
        expected_source_sha=source_sha,
        expected_timeline_fp=timeline_fingerprint,
        expected_detection_fp=detection_fingerprint,
        expected_analysis_window_fp=analysis_window_fingerprint,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"total_track_count: {summary['total_track_count']}")
    print(f"quality_status: {summary['quality_status']}")
    print(f"review_count: {summary['review_count']}")
    print(f"track_observations_parquet: {summary['track_observations_parquet']}")
    print(f"pipeline_receipt_json: {summary['pipeline_receipt_json']}")
    print(f"quality_report_json: {summary['quality_report_json']}")
    print(f"review_queue_json: {summary['review_queue_json']}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_tracking_validate(*, config_path: Path, frames: int, keep: bool) -> int:
    """Run Stage 6D synthetic tracking pipeline validator."""
    import runpy

    script = _project_root() / "scripts" / "check_tracking_pipeline.py"
    argv: list[str] = ["--config", str(config_path), "--frames", str(frames)]
    if keep:
        argv.append("--keep")
    ns = runpy.run_path(str(script), run_name="__not_main__")
    main_fn = ns.get("main")
    if not callable(main_fn):
        print("tracking pipeline validator missing main()", file=sys.stderr)
        return 2
    # Reconstruct argv for argparse inside main.
    old = sys.argv
    try:
        sys.argv = [str(script), *argv]
        return int(main_fn())
    finally:
        sys.argv = old


def cmd_tracking_receipt_validate(receipt_path: Path) -> int:
    """Validate tracking_run_receipt JSON against schema (no tracker run)."""
    from football_analytics.tracking.receipt import validate_receipt_payload

    if not receipt_path.is_file() or receipt_path.is_symlink():
        print(f"receipt missing or symlink: {receipt_path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print("receipt root must be object", file=sys.stderr)
            return 1
        validate_receipt_payload(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"receipt_invalid: {exc}", file=sys.stderr)
        return 1
    print("receipt_valid: true")
    return 0


def cmd_tracking_humans_run(
    *,
    detections: Path,
    frames: Path,
    analysis_windows: Path,
    output_dir: Path,
    config_path: Path,
    detection_attributes: Path | None,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 6B: human multi-object tracking baseline."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.tracking.human_tracking_config import (
        default_human_tracking_config_path,
        load_human_tracking_config,
    )
    from football_analytics.tracking.human_tracking_service import run_human_tracking

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_human_tracking_config_path(repo_root=root)
    try:
        config = load_human_tracking_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root or Path(str(config["runtime_root"]))
    result = run_human_tracking(
        detections=str(detections),
        frames=str(frames),
        analysis_windows=str(analysis_windows),
        output_dir=str(output_dir),
        config=config,
        detection_attributes=str(detection_attributes) if detection_attributes else None,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"observations_parquet: {summary['observations_parquet']}")
    print(f"summaries_parquet: {summary['summaries_parquet']}")
    print(f"lifecycle_parquet: {summary['lifecycle_parquet']}")
    print(f"receipt_json: {summary['receipt_json']}")
    print(f"evaluation_json: {summary['evaluation_json']}")
    print(f"evaluation_status: {summary.get('evaluation_status')}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_tracking_humans_evaluate(
    *,
    observations: Path,
    config_path: Path,
    ground_truth: Path | None,
) -> int:
    """Stage 6B: evaluate human tracks (not_evaluated without reviewed GT)."""
    import pyarrow.parquet as pq

    from football_analytics.data.registry import default_project_root
    from football_analytics.tracking.human_tracking_config import (
        default_human_tracking_config_path,
        human_tracking_config_fingerprint,
        load_human_tracking_config,
    )
    from football_analytics.tracking.human_tracking_evaluation import evaluate_human_tracking

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_human_tracking_config_path(repo_root=root)
    try:
        config = load_human_tracking_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    if not observations.is_file() or observations.is_symlink():
        print("observations missing or symlink", file=sys.stderr)
        return 2
    obs_rows = pq.read_table(observations).to_pylist()
    gt_rows = None
    reviewed = False
    if ground_truth is not None and ground_truth.is_file():
        payload = json.loads(ground_truth.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("reviewed") is True:
            gt_rows = list(payload.get("tracks") or payload.get("ground_truth") or [])
            reviewed = True
    report = evaluate_human_tracking(
        track_observations=obs_rows,
        ground_truth=gt_rows,
        has_reviewed_ground_truth=reviewed,
    )
    run_id = str(obs_rows[0]["run_id"]) if obs_rows else "run_unknown"
    video_id = str(obs_rows[0]["video_id"]) if obs_rows else "video_unknown"
    body = report.to_dict(
        run_id=run_id,
        video_id=video_id,
        config_fingerprint=human_tracking_config_fingerprint(config),
    )
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


def cmd_tracking_ball_run(
    *,
    detections: Path,
    frames: Path,
    analysis_windows: Path,
    output_dir: Path,
    config_path: Path,
    detection_attributes: Path | None,
    contain_root: Path | None,
    run_id: str | None,
    video_id: str | None,
) -> int:
    """Stage 6C: ball tracking baseline."""
    from football_analytics.data.registry import default_project_root
    from football_analytics.tracking.ball_tracking_config import (
        default_ball_tracking_config_path,
        load_ball_tracking_config,
    )
    from football_analytics.tracking.ball_tracking_service import run_ball_tracking

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_ball_tracking_config_path(repo_root=root)
    try:
        config = load_ball_tracking_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    contain = contain_root or Path(str(config["runtime_root"]))
    result = run_ball_tracking(
        detections=str(detections),
        frames=str(frames),
        analysis_windows=str(analysis_windows),
        output_dir=str(output_dir),
        config=config,
        detection_attributes=str(detection_attributes) if detection_attributes else None,
        contain_root=contain,
        run_id=run_id,
        video_id=video_id,
    )
    summary = result.to_summary()
    print(f"accepted: {summary['accepted']}")
    print(f"exit_code: {summary['exit_code']}")
    print(f"observations_parquet: {summary['observations_parquet']}")
    print(f"summaries_parquet: {summary['summaries_parquet']}")
    print(f"lifecycle_parquet: {summary['lifecycle_parquet']}")
    print(f"receipt_json: {summary['receipt_json']}")
    print(f"evaluation_json: {summary['evaluation_json']}")
    print(f"primary_sidecar_json: {summary.get('primary_sidecar_json')}")
    print(f"evaluation_status: {summary.get('evaluation_status')}")
    if summary.get("error_code"):
        print(f"error_code: {summary['error_code']}")
    return int(result.exit_code)


def cmd_tracking_ball_evaluate(
    *,
    observations: Path,
    config_path: Path,
    ground_truth: Path | None,
) -> int:
    """Stage 6C: evaluate ball tracks (not_evaluated without reviewed GT)."""
    import pyarrow.parquet as pq

    from football_analytics.data.registry import default_project_root
    from football_analytics.tracking.ball_tracking_config import (
        ball_tracking_config_fingerprint,
        default_ball_tracking_config_path,
        load_ball_tracking_config,
    )
    from football_analytics.tracking.ball_tracking_evaluation import evaluate_ball_tracking

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    if not cfg_path.is_file():
        cfg_path = default_ball_tracking_config_path(repo_root=root)
    try:
        config = load_ball_tracking_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    if not observations.is_file() or observations.is_symlink():
        print("observations missing or symlink", file=sys.stderr)
        return 2
    obs_rows = pq.read_table(observations).to_pylist()
    gt_rows = None
    reviewed = False
    if ground_truth is not None and ground_truth.is_file():
        payload = json.loads(ground_truth.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("reviewed") is True:
            gt_rows = list(payload.get("tracks") or payload.get("ground_truth") or [])
            reviewed = True
    report = evaluate_ball_tracking(
        track_observations=obs_rows,
        ground_truth=gt_rows,
        has_reviewed_ground_truth=reviewed,
    )
    run_id = str(obs_rows[0]["run_id"]) if obs_rows else "run_unknown"
    video_id = str(obs_rows[0]["video_id"]) if obs_rows else "video_unknown"
    body = report.to_dict(
        run_id=run_id,
        video_id=video_id,
        config_fingerprint=ball_tracking_config_fingerprint(config),
    )
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


def cmd_perception_validate(*, config_path: Path, frames: int, keep: bool) -> int:
    """Stage 5E: run detection pipeline validator script entry."""
    import runpy

    from football_analytics.data.registry import default_project_root

    root = default_project_root()
    script = root / "scripts" / "check_detection_pipeline.py"
    cfg = config_path if config_path.is_absolute() else root / config_path
    argv = [
        str(script),
        "--config",
        str(cfg),
        "--frames",
        str(frames),
    ]
    if keep:
        argv.append("--keep")
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    finally:
        sys.argv = old
    return 0


def cmd_perception_roles_evaluate(
    *,
    predictions: Path,
    ground_truth: Path,
    output: Path,
    config_path: Path,
) -> int:
    """Stage 5D: evaluate predicted roles vs reviewed ground truth."""
    import json

    from football_analytics.core.records import write_json_record
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import default_project_root
    from football_analytics.perception.role_config import load_human_role_config
    from football_analytics.perception.role_evaluation import evaluate_roles_from_rows

    root = default_project_root()
    cfg_path = config_path if config_path.is_absolute() else root / config_path
    try:
        load_human_role_config(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    try:
        pred_path = Path(predictions)
        if pred_path.suffix.lower() == ".json":
            pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
            pred_rows = list(pred_payload.get("roles") or pred_payload.get("predictions") or [])
        else:
            pred_table = read_contract_parquet(pred_path, get_contract("detection_attributes", 1))
            pred_rows = pred_table.to_pylist()
            for r in pred_rows:
                r.setdefault("assignment_status", "classified")
        gt_path = Path(ground_truth)
        if gt_path.suffix.lower() == ".json":
            gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
            gt_rows = list(gt_payload.get("roles") or gt_payload.get("ground_truth") or [])
        else:
            gt_table = read_contract_parquet(gt_path, get_contract("detection_attributes", 1))
            gt_rows = gt_table.to_pylist()
        metrics = evaluate_roles_from_rows(pred_rows, gt_rows)
        write_json_record(output, metrics.to_dict(), overwrite=False)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"status: {metrics.status}")
    print(f"macro_f1: {metrics.macro_f1}")
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

    p_perception = sub.add_parser("perception", help="Perception detection helpers (Stage 5)")
    perception_sub = p_perception.add_subparsers(dest="perception_command")
    p_humans = perception_sub.add_parser("humans", help="Human detection / evaluation (Stage 5B)")
    humans_sub = p_humans.add_subparsers(dest="humans_command")
    p_h_detect = humans_sub.add_parser("detect", help="Detect humans (baseline)")
    p_h_detect.add_argument("--source", type=Path, required=True, help="Absolute source video path")
    p_h_detect.add_argument(
        "--timeline", type=Path, required=True, help="Absolute frames.parquet path"
    )
    p_h_detect.add_argument(
        "--analysis-windows",
        type=Path,
        required=True,
        help="Absolute analysis_windows.parquet path",
    )
    p_h_detect.add_argument(
        "--output-dir", type=Path, required=True, help="Runtime output directory"
    )
    p_h_detect.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/human_detector_baseline.yaml"),
        help="Human detector baseline config YAML",
    )
    p_h_detect.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: config.runtime_root)",
    )
    p_h_detect.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_h_detect.add_argument("--video-id", type=str, default=None, help="Optional video_id")
    p_h_eval = humans_sub.add_parser("evaluate", help="Evaluate predicted humans vs ground truth")
    p_h_eval.add_argument(
        "--predictions", type=Path, required=True, help="Predicted detections.parquet"
    )
    p_h_eval.add_argument(
        "--ground-truth", type=Path, required=True, help="Ground-truth JSON or parquet"
    )
    p_h_eval.add_argument("--output", type=Path, required=True, help="evaluation.json output path")
    p_h_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/human_detector_baseline.yaml"),
        help="Human detector baseline config YAML",
    )
    p_ball = perception_sub.add_parser("ball", help="Ball detection / evaluation (Stage 5C)")
    ball_sub = p_ball.add_subparsers(dest="ball_command")
    p_b_detect = ball_sub.add_parser("detect", help="Detect balls (baseline)")
    p_b_detect.add_argument("--source", type=Path, required=True, help="Absolute source video path")
    p_b_detect.add_argument(
        "--timeline", type=Path, required=True, help="Absolute frames.parquet path"
    )
    p_b_detect.add_argument(
        "--analysis-windows",
        type=Path,
        required=True,
        help="Absolute analysis_windows.parquet path",
    )
    p_b_detect.add_argument(
        "--output-dir", type=Path, required=True, help="Runtime output directory"
    )
    p_b_detect.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/ball_detector_baseline.yaml"),
        help="Ball detector baseline config YAML",
    )
    p_b_detect.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: config.runtime_root)",
    )
    p_b_detect.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_b_detect.add_argument("--video-id", type=str, default=None, help="Optional video_id")
    p_b_eval = ball_sub.add_parser("evaluate", help="Evaluate predicted balls vs ground truth")
    p_b_eval.add_argument(
        "--predictions", type=Path, required=True, help="Predicted detections.parquet"
    )
    p_b_eval.add_argument(
        "--ground-truth", type=Path, required=True, help="Ground-truth JSON or parquet"
    )
    p_b_eval.add_argument("--output", type=Path, required=True, help="ball_evaluation.json output")
    p_b_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/ball_detector_baseline.yaml"),
        help="Ball detector baseline config YAML",
    )
    p_roles = perception_sub.add_parser(
        "roles", help="Human role classification / evaluation (Stage 5D)"
    )
    roles_sub = p_roles.add_subparsers(dest="roles_command")
    p_r_classify = roles_sub.add_parser("classify", help="Classify human roles (baseline)")
    p_r_classify.add_argument(
        "--detections", type=Path, required=True, help="Absolute detections.parquet path"
    )
    p_r_classify.add_argument(
        "--detection-attributes",
        type=Path,
        required=True,
        help="Absolute detection_attributes.parquet path",
    )
    p_r_classify.add_argument(
        "--output-dir", type=Path, required=True, help="Runtime output directory"
    )
    p_r_classify.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/human_role_baseline.yaml"),
        help="Human role baseline config YAML",
    )
    p_r_classify.add_argument(
        "--detection-frame-status",
        type=Path,
        default=None,
        help="Optional detection_frame_status.parquet",
    )
    p_r_classify.add_argument(
        "--analysis-windows",
        type=Path,
        default=None,
        help="Optional analysis_windows.parquet",
    )
    p_r_classify.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional source video for crops (crops not persisted)",
    )
    p_r_classify.add_argument(
        "--contain-root",
        type=Path,
        default=None,
        help="Containment root (default: config.runtime_root)",
    )
    p_r_classify.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    p_r_classify.add_argument("--video-id", type=str, default=None, help="Optional video_id")
    p_r_classify.add_argument(
        "--ground-truth", type=Path, default=None, help="Optional reviewed role GT"
    )
    p_r_eval = roles_sub.add_parser("evaluate", help="Evaluate predicted roles vs ground truth")
    p_r_eval.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predicted roles JSON or detection_attributes.parquet",
    )
    p_r_eval.add_argument(
        "--ground-truth", type=Path, required=True, help="Ground-truth JSON or parquet"
    )
    p_r_eval.add_argument("--output", type=Path, required=True, help="role_evaluation.json output")
    p_r_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/human_role_baseline.yaml"),
        help="Human role baseline config YAML",
    )
    p_integrate = perception_sub.add_parser(
        "integrate", help="Fuse human/ball/role detection artifacts (Stage 5E)"
    )
    p_integrate.add_argument("--human-detections", type=Path, required=True)
    p_integrate.add_argument("--human-frame-status", type=Path, required=True)
    p_integrate.add_argument("--human-attributes", type=Path, required=True)
    p_integrate.add_argument("--human-receipt", type=Path, required=True)
    p_integrate.add_argument("--ball-detections", type=Path, required=True)
    p_integrate.add_argument("--ball-frame-status", type=Path, required=True)
    p_integrate.add_argument("--ball-attributes", type=Path, required=True)
    p_integrate.add_argument("--ball-receipt", type=Path, required=True)
    p_integrate.add_argument("--role-attributes", type=Path, required=True)
    p_integrate.add_argument("--role-receipt", type=Path, required=True)
    p_integrate.add_argument("--output-dir", type=Path, required=True)
    p_integrate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/detection_pipeline.yaml"),
        help="Detection pipeline config YAML",
    )
    p_integrate.add_argument("--analysis-windows", type=Path, default=None)
    p_integrate.add_argument("--frames", type=Path, default=None)
    p_integrate.add_argument("--contain-root", type=Path, default=None)
    p_integrate.add_argument("--run-id", type=str, default=None)
    p_integrate.add_argument("--video-id", type=str, default=None)
    p_integrate.add_argument("--source-sha", type=str, default=None)
    p_integrate.add_argument("--timeline-fingerprint", type=str, default=None)
    p_p_validate = perception_sub.add_parser(
        "validate", help="Run detection pipeline validator (Stage 5E)"
    )
    p_p_validate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/perception/detection_pipeline.yaml"),
    )
    p_p_validate.add_argument("--frames", type=int, default=8, help="Synthetic frames (≤20)")
    p_p_validate.add_argument("--keep", action="store_true", help="Keep validator session dir")

    p_tracking = sub.add_parser(
        "tracking", help="Multi-object tracking helpers (Stage 6A/6B/6C/6D)"
    )
    tracking_sub = p_tracking.add_subparsers(dest="tracking_command")
    p_trk_contracts = tracking_sub.add_parser("contracts", help="Tracking contract helpers")
    trk_contracts_sub = p_trk_contracts.add_subparsers(dest="tracking_contracts_command")
    p_trk_c_val = trk_contracts_sub.add_parser(
        "validate", help="Validate tracking contracts (synthetic Stage 6A)"
    )
    p_trk_c_val.add_argument("--keep", action="store_true", help="Keep validator session dir")
    p_trk_c_val.add_argument("--json", action="store_true", help="Emit JSON report")
    p_trk_receipt = tracking_sub.add_parser("receipt", help="Tracking receipt helpers")
    trk_receipt_sub = p_trk_receipt.add_subparsers(dest="tracking_receipt_command")
    p_trk_r_val = trk_receipt_sub.add_parser(
        "validate", help="Validate a tracking_run_receipt JSON file"
    )
    p_trk_r_val.add_argument("receipt", type=Path, help="Path to tracking_run_receipt JSON")
    p_trk_humans = tracking_sub.add_parser("humans", help="Human MOT baseline (Stage 6B)")
    trk_humans_sub = p_trk_humans.add_subparsers(dest="tracking_humans_command")
    p_trk_h_run = trk_humans_sub.add_parser("run", help="Run human multi-object tracking")
    p_trk_h_run.add_argument("--detections", type=Path, required=True)
    p_trk_h_run.add_argument("--frames", type=Path, required=True)
    p_trk_h_run.add_argument("--analysis-windows", type=Path, required=True)
    p_trk_h_run.add_argument("--output-dir", type=Path, required=True)
    p_trk_h_run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/human_tracking_baseline.yaml"),
    )
    p_trk_h_run.add_argument("--detection-attributes", type=Path, default=None)
    p_trk_h_run.add_argument("--contain-root", type=Path, default=None)
    p_trk_h_run.add_argument("--run-id", type=str, default=None)
    p_trk_h_run.add_argument("--video-id", type=str, default=None)
    p_trk_h_eval = trk_humans_sub.add_parser("evaluate", help="Evaluate human tracks")
    p_trk_h_eval.add_argument("--observations", type=Path, required=True)
    p_trk_h_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/human_tracking_baseline.yaml"),
    )
    p_trk_h_eval.add_argument("--ground-truth", type=Path, default=None)
    p_trk_ball = tracking_sub.add_parser("ball", help="Ball tracking baseline (Stage 6C)")
    trk_ball_sub = p_trk_ball.add_subparsers(dest="tracking_ball_command")
    p_trk_b_run = trk_ball_sub.add_parser("run", help="Run ball tracking")
    p_trk_b_run.add_argument("--detections", type=Path, required=True)
    p_trk_b_run.add_argument("--frames", type=Path, required=True)
    p_trk_b_run.add_argument("--analysis-windows", type=Path, required=True)
    p_trk_b_run.add_argument("--output-dir", type=Path, required=True)
    p_trk_b_run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/ball_tracking_baseline.yaml"),
    )
    p_trk_b_run.add_argument("--detection-attributes", type=Path, default=None)
    p_trk_b_run.add_argument("--contain-root", type=Path, default=None)
    p_trk_b_run.add_argument("--run-id", type=str, default=None)
    p_trk_b_run.add_argument("--video-id", type=str, default=None)
    p_trk_b_eval = trk_ball_sub.add_parser("evaluate", help="Evaluate ball tracks")
    p_trk_b_eval.add_argument("--observations", type=Path, required=True)
    p_trk_b_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/ball_tracking_baseline.yaml"),
    )
    p_trk_b_eval.add_argument("--ground-truth", type=Path, default=None)
    p_trk_integrate = tracking_sub.add_parser(
        "integrate", help="Fuse human/ball tracking artifacts (Stage 6D)"
    )
    p_trk_integrate.add_argument("--detections", type=Path, required=True)
    p_trk_integrate.add_argument("--detection-attributes", type=Path, required=True)
    p_trk_integrate.add_argument("--detection-receipt", type=Path, required=True)
    p_trk_integrate.add_argument("--human-observations", type=Path, required=True)
    p_trk_integrate.add_argument("--human-summaries", type=Path, required=True)
    p_trk_integrate.add_argument("--human-lifecycle", type=Path, required=True)
    p_trk_integrate.add_argument("--human-receipt", type=Path, required=True)
    p_trk_integrate.add_argument("--ball-observations", type=Path, required=True)
    p_trk_integrate.add_argument("--ball-summaries", type=Path, required=True)
    p_trk_integrate.add_argument("--ball-lifecycle", type=Path, required=True)
    p_trk_integrate.add_argument("--ball-receipt", type=Path, required=True)
    p_trk_integrate.add_argument("--output-dir", type=Path, required=True)
    p_trk_integrate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/tracking_pipeline.yaml"),
        help="Tracking pipeline config YAML",
    )
    p_trk_integrate.add_argument("--frames", type=Path, default=None)
    p_trk_integrate.add_argument("--analysis-windows", type=Path, default=None)
    p_trk_integrate.add_argument("--ball-primary-sidecar", type=Path, default=None)
    p_trk_integrate.add_argument("--contain-root", type=Path, default=None)
    p_trk_integrate.add_argument("--run-id", type=str, default=None)
    p_trk_integrate.add_argument("--video-id", type=str, default=None)
    p_trk_integrate.add_argument("--source-sha", type=str, default=None)
    p_trk_integrate.add_argument("--timeline-fingerprint", type=str, default=None)
    p_trk_integrate.add_argument("--detection-fingerprint", type=str, default=None)
    p_trk_integrate.add_argument("--analysis-window-fingerprint", type=str, default=None)
    p_trk_validate = tracking_sub.add_parser(
        "validate", help="Run tracking pipeline validator (Stage 6D)"
    )
    p_trk_validate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/tracking_pipeline.yaml"),
    )
    p_trk_validate.add_argument("--frames", type=int, default=8, help="Synthetic frames (≤20)")
    p_trk_validate.add_argument("--keep", action="store_true", help="Keep validator session dir")

    p_identity = sub.add_parser("identity", help="ReID / identity / target-player (Stage 7A/7B)")
    identity_sub = p_identity.add_subparsers(dest="identity_command")
    p_id_contracts = identity_sub.add_parser("contracts", help="Identity contract helpers")
    id_contracts_sub = p_id_contracts.add_subparsers(dest="identity_contracts_command")
    p_id_c_val = id_contracts_sub.add_parser(
        "validate", help="Validate identity contracts (synthetic Stage 7A)"
    )
    p_id_c_val.add_argument("--keep", action="store_true", help="Keep validator session dir")
    p_id_c_val.add_argument("--json", action="store_true", help="Emit JSON report")
    p_id_target = identity_sub.add_parser("target", help="Target player request helpers")
    id_target_sub = p_id_target.add_subparsers(dest="identity_target_command")
    p_id_t_val = id_target_sub.add_parser(
        "validate", help="Validate a target_player_request JSON file"
    )
    p_id_t_val.add_argument("request", type=Path, help="Path to target_player_request JSON")
    p_id_receipt = identity_sub.add_parser("receipt", help="Identity receipt helpers")
    id_receipt_sub = p_id_receipt.add_subparsers(dest="identity_receipt_command")
    p_id_r_val = id_receipt_sub.add_parser(
        "validate", help="Validate an identity_run_receipt JSON file"
    )
    p_id_r_val.add_argument("receipt", type=Path, help="Path to identity_run_receipt JSON")

    p_id_appearance = identity_sub.add_parser(
        "appearance", help="Appearance embedding extract (Stage 7B)"
    )
    id_app_sub = p_id_appearance.add_subparsers(dest="identity_appearance_command")
    p_id_app_ext = id_app_sub.add_parser("extract", help="Extract tracklet appearance profiles")
    p_id_app_ext.add_argument("--output-dir", type=Path, required=True)
    p_id_app_ext.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identity/appearance_reid_baseline.yaml"),
    )
    p_id_app_ext.add_argument("--contain-root", type=Path, default=None)
    p_id_app_ext.add_argument("--run-id", type=str, default=None)
    p_id_app_ext.add_argument("--video-id", type=str, default=None)
    p_id_app_ext.add_argument(
        "--fixture",
        type=str,
        default="same_appearance",
        help="Synthetic fixture name (no real video required)",
    )
    p_id_app_val = id_app_sub.add_parser("validate", help="Run appearance ReID baseline validator")
    p_id_app_val.add_argument("--keep", action="store_true")
    p_id_app_val.add_argument("--json", action="store_true")

    p_id_reid = identity_sub.add_parser("reid", help="Tracklet ReID candidates/evaluate (Stage 7B)")
    id_reid_sub = p_id_reid.add_subparsers(dest="identity_reid_command")
    p_id_reid_cand = id_reid_sub.add_parser("candidates", help="Propose ReID candidate links")
    p_id_reid_cand.add_argument("--output-dir", type=Path, required=True)
    p_id_reid_cand.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identity/appearance_reid_baseline.yaml"),
    )
    p_id_reid_cand.add_argument("--contain-root", type=Path, default=None)
    p_id_reid_cand.add_argument("--run-id", type=str, default=None)
    p_id_reid_cand.add_argument("--video-id", type=str, default=None)
    p_id_reid_cand.add_argument("--fixture", type=str, default="same_appearance")
    p_id_reid_eval = id_reid_sub.add_parser("evaluate", help="Evaluate appearance ReID")
    p_id_reid_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/identity/appearance_reid_baseline.yaml"),
    )
    p_id_reid_eval.add_argument("--links", type=Path, default=None)
    p_id_reid_eval.add_argument("--profiles", type=Path, default=None)
    p_id_reid_eval.add_argument("--ground-truth", type=Path, default=None)

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
    if args.command == "perception":
        if args.perception_command == "humans":
            if args.humans_command == "detect":
                return cmd_perception_humans_detect(
                    source=args.source,
                    timeline=args.timeline,
                    analysis_windows=args.analysis_windows,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                )
            if args.humans_command == "evaluate":
                return cmd_perception_humans_evaluate(
                    predictions=args.predictions,
                    ground_truth=args.ground_truth,
                    output=args.output,
                    config_path=args.config,
                )
            parser.parse_args(["perception", "humans", "--help"])
            return 2
        if args.perception_command == "ball":
            if args.ball_command == "detect":
                return cmd_perception_ball_detect(
                    source=args.source,
                    timeline=args.timeline,
                    analysis_windows=args.analysis_windows,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                )
            if args.ball_command == "evaluate":
                return cmd_perception_ball_evaluate(
                    predictions=args.predictions,
                    ground_truth=args.ground_truth,
                    output=args.output,
                    config_path=args.config,
                )
            parser.parse_args(["perception", "ball", "--help"])
            return 2
        if args.perception_command == "roles":
            if args.roles_command == "classify":
                return cmd_perception_roles_classify(
                    detections=args.detections,
                    detection_attributes=args.detection_attributes,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    detection_frame_status=args.detection_frame_status,
                    analysis_windows=args.analysis_windows,
                    source=args.source,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                    ground_truth=args.ground_truth,
                )
            if args.roles_command == "evaluate":
                return cmd_perception_roles_evaluate(
                    predictions=args.predictions,
                    ground_truth=args.ground_truth,
                    output=args.output,
                    config_path=args.config,
                )
            parser.parse_args(["perception", "roles", "--help"])
            return 2
        if args.perception_command == "integrate":
            return cmd_perception_integrate(
                human_detections=args.human_detections,
                human_frame_status=args.human_frame_status,
                human_attributes=args.human_attributes,
                human_receipt=args.human_receipt,
                ball_detections=args.ball_detections,
                ball_frame_status=args.ball_frame_status,
                ball_attributes=args.ball_attributes,
                ball_receipt=args.ball_receipt,
                role_attributes=args.role_attributes,
                role_receipt=args.role_receipt,
                output_dir=args.output_dir,
                config_path=args.config,
                contain_root=args.contain_root,
                analysis_windows=args.analysis_windows,
                frames=args.frames,
                run_id=args.run_id,
                video_id=args.video_id,
                source_sha=args.source_sha,
                timeline_fingerprint=args.timeline_fingerprint,
            )
        if args.perception_command == "validate":
            return cmd_perception_validate(
                config_path=args.config,
                frames=int(args.frames),
                keep=bool(args.keep),
            )
        parser.parse_args(["perception", "--help"])
        return 2
    if args.command == "tracking":
        if args.tracking_command == "contracts":
            if args.tracking_contracts_command == "validate":
                return cmd_tracking_contracts_validate(
                    keep=bool(args.keep), as_json=bool(args.json)
                )
            parser.parse_args(["tracking", "contracts", "--help"])
            return 2
        if args.tracking_command == "receipt":
            if args.tracking_receipt_command == "validate":
                return cmd_tracking_receipt_validate(args.receipt)
            parser.parse_args(["tracking", "receipt", "--help"])
            return 2
        if args.tracking_command == "humans":
            if args.tracking_humans_command == "run":
                return cmd_tracking_humans_run(
                    detections=args.detections,
                    frames=args.frames,
                    analysis_windows=args.analysis_windows,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    detection_attributes=args.detection_attributes,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                )
            if args.tracking_humans_command == "evaluate":
                return cmd_tracking_humans_evaluate(
                    observations=args.observations,
                    config_path=args.config,
                    ground_truth=args.ground_truth,
                )
            parser.parse_args(["tracking", "humans", "--help"])
            return 2
        if args.tracking_command == "ball":
            if args.tracking_ball_command == "run":
                return cmd_tracking_ball_run(
                    detections=args.detections,
                    frames=args.frames,
                    analysis_windows=args.analysis_windows,
                    output_dir=args.output_dir,
                    config_path=args.config,
                    detection_attributes=args.detection_attributes,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                )
            if args.tracking_ball_command == "evaluate":
                return cmd_tracking_ball_evaluate(
                    observations=args.observations,
                    config_path=args.config,
                    ground_truth=args.ground_truth,
                )
            parser.parse_args(["tracking", "ball", "--help"])
            return 2
        if args.tracking_command == "integrate":
            return cmd_tracking_integrate(
                detections=args.detections,
                detection_attributes=args.detection_attributes,
                detection_receipt=args.detection_receipt,
                human_observations=args.human_observations,
                human_summaries=args.human_summaries,
                human_lifecycle=args.human_lifecycle,
                human_receipt=args.human_receipt,
                ball_observations=args.ball_observations,
                ball_summaries=args.ball_summaries,
                ball_lifecycle=args.ball_lifecycle,
                ball_receipt=args.ball_receipt,
                output_dir=args.output_dir,
                config_path=args.config,
                contain_root=args.contain_root,
                frames=args.frames,
                analysis_windows=args.analysis_windows,
                ball_primary_sidecar=args.ball_primary_sidecar,
                run_id=args.run_id,
                video_id=args.video_id,
                source_sha=args.source_sha,
                timeline_fingerprint=args.timeline_fingerprint,
                detection_fingerprint=args.detection_fingerprint,
                analysis_window_fingerprint=args.analysis_window_fingerprint,
            )
        if args.tracking_command == "validate":
            return cmd_tracking_validate(
                config_path=args.config,
                frames=int(args.frames),
                keep=bool(args.keep),
            )
        parser.parse_args(["tracking", "--help"])
        return 2
    if args.command == "identity":
        if args.identity_command == "contracts":
            if args.identity_contracts_command == "validate":
                return cmd_identity_contracts_validate(
                    keep=bool(args.keep), as_json=bool(args.json)
                )
            parser.parse_args(["identity", "contracts", "--help"])
            return 2
        if args.identity_command == "target":
            if args.identity_target_command == "validate":
                return cmd_identity_target_validate(args.request)
            parser.parse_args(["identity", "target", "--help"])
            return 2
        if args.identity_command == "receipt":
            if args.identity_receipt_command == "validate":
                return cmd_identity_receipt_validate(args.receipt)
            parser.parse_args(["identity", "receipt", "--help"])
            return 2
        if args.identity_command == "appearance":
            if args.identity_appearance_command == "extract":
                return cmd_identity_appearance_extract(
                    output_dir=args.output_dir,
                    config_path=args.config,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                    fixture=str(args.fixture),
                )
            if args.identity_appearance_command == "validate":
                return cmd_identity_appearance_validate(
                    keep=bool(args.keep), as_json=bool(args.json)
                )
            parser.parse_args(["identity", "appearance", "--help"])
            return 2
        if args.identity_command == "reid":
            if args.identity_reid_command == "candidates":
                return cmd_identity_reid_candidates(
                    output_dir=args.output_dir,
                    config_path=args.config,
                    contain_root=args.contain_root,
                    run_id=args.run_id,
                    video_id=args.video_id,
                    fixture=str(args.fixture),
                )
            if args.identity_reid_command == "evaluate":
                return cmd_identity_reid_evaluate(
                    config_path=args.config,
                    links=args.links,
                    profiles=args.profiles,
                    ground_truth=args.ground_truth,
                )
            parser.parse_args(["identity", "reid", "--help"])
            return 2
        parser.parse_args(["identity", "--help"])
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
