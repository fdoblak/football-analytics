"""Stage/cache validator core logic (Stage 2D).

Used by ``scripts/check_stage_cache.py`` and project checks.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/stage_cache_checks")
DEFAULT_POLICY = Path("configs/system/cache_policy.yaml")
DEFAULT_PATHS = Path("configs/system/paths.yaml")

_PIPELINE_SCHEMAS = (
    "schemas/pipeline/artifact_ref.schema.json",
    "schemas/pipeline/stage_request.schema.json",
    "schemas/pipeline/stage_result.schema.json",
    "schemas/pipeline/stage_execution_receipt.schema.json",
)
_CACHE_SCHEMAS = ("schemas/cache/cache_manifest.schema.json",)


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False) -> None:
        self.errors.append(msg)
        self.exit_code = EXIT_INTEGRITY if integrity else EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code == EXIT_INTEGRITY or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "extras": self.extras,
        }


def _fp(label: str) -> str:
    from football_analytics.core.hashing import hash_canonical_json

    return hash_canonical_json({"stage_cache_check": label})


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _check_import_smoke(result: Result) -> None:
    """Import stage interface without pulling pyarrow/torch."""
    before = {m for m in sys.modules if m == "torch" or m.startswith("torch.")}
    before |= {m for m in sys.modules if m == "pyarrow" or m.startswith("pyarrow.")}
    try:
        import football_analytics.pipeline  # noqa: F401
        from football_analytics.pipeline.cache_key import compute_cache_key  # noqa: F401
        from football_analytics.pipeline.stage import (  # noqa: F401
            StageRegistry,
            SyntheticEchoStage,
            make_stage_identity,
        )
        from football_analytics.pipeline.types import ArtifactRef, StageRequest  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        result.err(f"stage interface import failed: {type(exc).__name__}", integrity=True)
        return
    after = {m for m in sys.modules if m == "torch" or m.startswith("torch.")}
    after |= {m for m in sys.modules if m == "pyarrow" or m.startswith("pyarrow.")}
    newly = after - before
    if newly:
        result.err(
            f"pipeline import loaded heavy modules: {sorted(newly)[:8]}",
            integrity=True,
        )


def _check_key_determinism(result: Result) -> None:
    from football_analytics.pipeline.artifacts import build_artifact_ref
    from football_analytics.pipeline.cache_key import compute_cache_key
    from football_analytics.pipeline.stage import SyntheticEchoStage

    stage = SyntheticEchoStage()
    with __import__("tempfile").TemporaryDirectory() as tmp:
        root = Path(tmp)
        path_a = root / "a.bin"
        path_b = root / "b.bin"
        path_a.write_bytes(b"alpha")
        path_b.write_bytes(b"beta")
        ref_a = build_artifact_ref("a", path_a, root=root, media_type="application/octet-stream")
        ref_b = build_artifact_ref("b", path_b, root=root, media_type="application/octet-stream")
        cfg = _fp("cfg1")
        compat = _fp("compat1")
        k1 = compute_cache_key(
            stage=stage.identity,
            config_fingerprint=cfg,
            compatibility_fingerprint=compat,
            inputs={"a": ref_a, "b": ref_b},
        )
        k2 = compute_cache_key(
            stage=stage.identity,
            config_fingerprint=cfg,
            compatibility_fingerprint=compat,
            inputs={"b": ref_b, "a": ref_a},
        )
        if k1 != k2 or len(k1) != 64:
            result.err("cache key not deterministic / order-independent", integrity=True)
        k3 = compute_cache_key(
            stage=stage.identity,
            config_fingerprint=_fp("cfg2"),
            compatibility_fingerprint=compat,
            inputs={"a": ref_a, "b": ref_b},
        )
        if k3 == k1:
            result.err("config fingerprint change did not alter cache key", integrity=True)
        result.extras["sample_cache_key"] = k1


def _make_request(
    *,
    stage: Any,
    work: Path,
    out: Path,
    inputs: dict[str, Any],
    config_fingerprint: str,
    compat: str,
) -> Any:
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.pipeline.types import StageRequest

    return StageRequest(
        run_id=generate_run_id(),
        stage_identity=stage.identity,
        config_fingerprint=config_fingerprint,
        compatibility_fingerprint=compat,
        inputs=inputs,
        working_directory=work,
        output_directory=out,
        requested_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        cache_policy_enabled=True,
    )


def _run_synthetic(
    result: Result,
    *,
    policy_path: Path,
    fixture_root: Path,
) -> None:
    from football_analytics.pipeline.artifacts import build_artifact_ref
    from football_analytics.pipeline.cache import entry_dir, load_cache_policy
    from football_analytics.pipeline.execution import execute_stage
    from football_analytics.pipeline.stage import SyntheticEchoStage

    policy = load_cache_policy(policy_path)
    cache_root = fixture_root / "cache"
    quarantine_root = fixture_root / "quarantine"
    cache_root.mkdir(parents=True, mode=0o700)
    quarantine_root.mkdir(parents=True, mode=0o700)

    stage = SyntheticEchoStage()
    cfg = _fp("synthetic_cfg")
    compat = _fp("synthetic_compat")

    work1 = fixture_root / "work1"
    out1 = fixture_root / "out1"
    work1.mkdir(mode=0o700)
    out1.mkdir(mode=0o700)
    inp = work1 / "input.bin"
    inp.write_bytes(b"synthetic-echo-input-v1")
    ref = build_artifact_ref("input", inp, root=work1, media_type="application/octet-stream")
    req1 = _make_request(
        stage=stage,
        work=work1,
        out=out1,
        inputs={"input": ref},
        config_fingerprint=cfg,
        compat=compat,
    )
    r1 = execute_stage(
        stage, req1, cache_root=cache_root, policy=policy, quarantine_root=quarantine_root
    )
    if r1.status != "succeeded" or r1.cache_hit:
        result.err(
            f"first execute expected miss/succeeded got status={r1.status} hit={r1.cache_hit}",
            integrity=True,
        )
    if stage.executions != 1:
        result.err(f"executions after miss expected 1 got {stage.executions}", integrity=True)
    result.extras["first_cache_key"] = r1.cache_key

    work2 = fixture_root / "work2"
    out2 = fixture_root / "out2"
    work2.mkdir(mode=0o700)
    out2.mkdir(mode=0o700)
    inp2 = work2 / "input.bin"
    inp2.write_bytes(b"synthetic-echo-input-v1")
    ref2 = build_artifact_ref("input", inp2, root=work2, media_type="application/octet-stream")
    req2 = _make_request(
        stage=stage,
        work=work2,
        out=out2,
        inputs={"input": ref2},
        config_fingerprint=cfg,
        compat=compat,
    )
    r2 = execute_stage(
        stage, req2, cache_root=cache_root, policy=policy, quarantine_root=quarantine_root
    )
    if not r2.cache_hit or r2.status != "cache_hit":
        result.err(
            f"second execute expected cache_hit got status={r2.status} hit={r2.cache_hit}",
            integrity=True,
        )
    if stage.executions != 1:
        result.err(f"executions after hit expected 1 got {stage.executions}", integrity=True)

    work3 = fixture_root / "work3"
    out3 = fixture_root / "out3"
    work3.mkdir(mode=0o700)
    out3.mkdir(mode=0o700)
    inp3 = work3 / "input.bin"
    inp3.write_bytes(b"synthetic-echo-input-v1")
    ref3 = build_artifact_ref("input", inp3, root=work3, media_type="application/octet-stream")
    req3 = _make_request(
        stage=stage,
        work=work3,
        out=out3,
        inputs={"input": ref3},
        config_fingerprint=_fp("synthetic_cfg_changed"),
        compat=compat,
    )
    r3 = execute_stage(
        stage, req3, cache_root=cache_root, policy=policy, quarantine_root=quarantine_root
    )
    if r3.cache_hit or r3.status != "succeeded":
        result.err(
            f"config change expected miss got status={r3.status} hit={r3.cache_hit}",
            integrity=True,
        )
    if stage.executions != 2:
        result.err(
            f"executions after config miss expected 2 got {stage.executions}",
            integrity=True,
        )

    result.extras["synthetic"] = True
    result.extras["cache_root"] = str(cache_root)
    # Confirm entry dirs exist for published keys
    if not entry_dir(cache_root, r1.cache_key).is_dir():
        result.err("published cache entry missing after miss", integrity=True)


def _run_corruption_smoke(
    result: Result,
    *,
    policy_path: Path,
    fixture_root: Path,
) -> None:
    from football_analytics.pipeline.artifacts import build_artifact_ref
    from football_analytics.pipeline.cache import entry_dir, load_cache_policy
    from football_analytics.pipeline.execution import execute_stage
    from football_analytics.pipeline.stage import SyntheticEchoStage

    policy = load_cache_policy(policy_path)
    if not policy.quarantine_corrupt_entries:
        result.warn("quarantine_corrupt_entries disabled; corruption smoke limited")

    cache_root = fixture_root / "cache_corrupt"
    quarantine_root = fixture_root / "quarantine_corrupt"
    cache_root.mkdir(parents=True, mode=0o700)
    quarantine_root.mkdir(parents=True, mode=0o700)

    stage = SyntheticEchoStage()
    cfg = _fp("corrupt_cfg")
    compat = _fp("corrupt_compat")
    work = fixture_root / "work_c"
    out = fixture_root / "out_c"
    work.mkdir(mode=0o700)
    out.mkdir(mode=0o700)
    inp = work / "input.bin"
    inp.write_bytes(b"corrupt-smoke-input")
    ref = build_artifact_ref("input", inp, root=work, media_type="application/octet-stream")
    req = _make_request(
        stage=stage,
        work=work,
        out=out,
        inputs={"input": ref},
        config_fingerprint=cfg,
        compat=compat,
    )
    r1 = execute_stage(
        stage, req, cache_root=cache_root, policy=policy, quarantine_root=quarantine_root
    )
    if r1.status != "succeeded":
        result.err("corruption smoke: initial publish execute failed", integrity=True)
        return

    entry = entry_dir(cache_root, r1.cache_key)
    arts = list((entry / "artifacts").rglob("*"))
    files = [p for p in arts if p.is_file() and not p.is_symlink()]
    if not files:
        result.err("corruption smoke: no artifact file to tamper", integrity=True)
        return
    target = files[0]
    blob = bytearray(target.read_bytes())
    if not blob:
        blob = bytearray(b"x")
    else:
        blob[0] = (blob[0] + 1) % 256
    target.write_bytes(bytes(blob))

    work2 = fixture_root / "work_c2"
    out2 = fixture_root / "out_c2"
    work2.mkdir(mode=0o700)
    out2.mkdir(mode=0o700)
    inp2 = work2 / "input.bin"
    inp2.write_bytes(b"corrupt-smoke-input")
    ref2 = build_artifact_ref("input", inp2, root=work2, media_type="application/octet-stream")
    req2 = _make_request(
        stage=stage,
        work=work2,
        out=out2,
        inputs={"input": ref2},
        config_fingerprint=cfg,
        compat=compat,
    )
    r2 = execute_stage(
        stage, req2, cache_root=cache_root, policy=policy, quarantine_root=quarantine_root
    )
    # Corruption should prevent hit; stage re-executes (miss path).
    if r2.cache_hit:
        result.err("corruption smoke: corrupt entry served as cache hit", integrity=True)
    warnings = " ".join(r2.warnings).lower()
    if "corrupt" not in warnings and "quarantine" not in warnings:
        result.warn("corruption smoke: no corruption/quarantine warning on result")

    permanent_delete = False
    q_receipts = list(quarantine_root.rglob("quarantine_receipt.json"))
    if policy.quarantine_corrupt_entries:
        if not q_receipts and entry.exists():
            result.err(
                "corruption smoke: expected quarantine but entry still present",
                integrity=True,
            )
        for receipt_path in q_receipts:
            try:
                payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                result.err(f"quarantine receipt unreadable: {type(exc).__name__}", integrity=True)
                continue
            if payload.get("permanent_delete_performed") is not False:
                permanent_delete = True
                result.err(
                    "quarantine receipt permanent_delete_performed must be false",
                    integrity=True,
                )
    result.extras["corruption_smoke"] = True
    result.extras["permanent_delete_performed"] = permanent_delete
    result.extras["quarantine_receipts"] = len(q_receipts)


def _concurrency_publish_worker(payload: dict[str, Any]) -> str:
    """Multiprocessing worker: attempt publish_cache_entry once."""
    try:
        root = Path(payload["project_root"])
        src = str(root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from football_analytics.pipeline.artifacts import build_artifact_ref
        from football_analytics.pipeline.cache import load_cache_policy, publish_cache_entry
        from football_analytics.pipeline.stage import make_stage_identity
        from football_analytics.pipeline.types import ArtifactRef, StageResult

        policy = load_cache_policy(Path(payload["policy_path"]))
        cache_root = Path(payload["cache_root"])
        artifact_root = Path(payload["artifact_root"])
        stage = make_stage_identity(
            name=payload["stage_name"],
            version=int(payload["stage_version"]),
            code_fingerprint=payload["code_fingerprint"],
            deterministic=True,
            cacheable=True,
        )
        artifacts: dict[str, ArtifactRef] = {}
        for item in payload["artifacts"]:
            path = artifact_root / item["relative_path"]
            ref = build_artifact_ref(
                item["logical_name"],
                path,
                root=artifact_root,
                media_type=item["media_type"],
            )
            artifacts[ref.logical_name] = ref
        result = StageResult(
            run_id=payload["run_id"],
            stage_name=stage.name,
            stage_version=stage.version,
            status="succeeded",
            cache_key=payload["cache_key"],
            cache_hit=False,
            started_at_utc=payload["started_at_utc"],
            finished_at_utc=payload["finished_at_utc"],
            duration_ms=1,
            inputs={},
            outputs={k: v.to_dict() for k, v in artifacts.items()},
            metrics={},
            warnings=(),
            error=None,
            execution_fingerprint=payload["execution_fingerprint"],
        )
        publish_cache_entry(
            cache_root=cache_root,
            cache_key=payload["cache_key"],
            stage_identity=stage,
            config_fingerprint=payload["config_fingerprint"],
            artifacts=MappingProxyType(artifacts),
            artifact_root=artifact_root,
            stage_result=result,
            policy=policy,
            source_run_id=payload["run_id"],
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}"


def _run_concurrency_smoke(
    result: Result,
    *,
    policy_path: Path,
    fixture_root: Path,
) -> None:
    from football_analytics.core.hashing import hash_canonical_json
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.pipeline.artifacts import build_artifact_ref, verify_artifact_on_disk
    from football_analytics.pipeline.cache import entry_dir, load_cache_policy
    from football_analytics.pipeline.cache_key import compute_cache_key
    from football_analytics.pipeline.stage import make_stage_identity

    policy = load_cache_policy(policy_path)
    cache_root = fixture_root / "cache_race"
    cache_root.mkdir(parents=True, mode=0o700)

    code_fp = _fp("race_code")
    stage = make_stage_identity(
        name="synthetic_race",
        version=1,
        code_fingerprint=code_fp,
        deterministic=True,
        cacheable=True,
    )
    cfg = _fp("race_cfg")
    compat = _fp("race_compat")
    run_id = generate_run_id()

    # Two isolated artifact roots with identical content.
    payloads: list[dict[str, Any]] = []
    arts_meta: list[dict[str, str]] = []
    for idx in (1, 2):
        root = fixture_root / f"race_arts_{idx}"
        root.mkdir(mode=0o700)
        path = root / "out.bin"
        path.write_bytes(b"race-payload-identical")
        ref = build_artifact_ref("out", path, root=root, media_type="application/octet-stream")
        arts_meta = [
            {
                "logical_name": ref.logical_name,
                "relative_path": ref.relative_path,
                "media_type": ref.media_type,
            }
        ]
        cache_key = compute_cache_key(
            stage=stage,
            config_fingerprint=cfg,
            compatibility_fingerprint=compat,
            inputs={},
        )
        # Include output hash via empty inputs + stage identity only — but publish needs
        # a stable key. Use the same key for both workers.
        exec_fp = hash_canonical_json({"race": "concurrency", "key": cache_key})
        payloads.append(
            {
                "project_root": str(_project_root()),
                "policy_path": str(policy_path.resolve()),
                "cache_root": str(cache_root.resolve()),
                "artifact_root": str(root.resolve()),
                "stage_name": stage.name,
                "stage_version": stage.version,
                "code_fingerprint": stage.code_fingerprint,
                "cache_key": cache_key,
                "config_fingerprint": cfg,
                "artifacts": arts_meta,
                "run_id": run_id,
                "started_at_utc": "2026-01-01T00:00:00.000000Z",
                "finished_at_utc": "2026-01-01T00:00:01.000000Z",
                "execution_fingerprint": exec_fp,
            }
        )

    cache_key = payloads[0]["cache_key"]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        async_result = pool.map_async(_concurrency_publish_worker, payloads)
        try:
            outcomes = async_result.get(timeout=60)
        except Exception as exc:  # noqa: BLE001 — includes TimeoutError
            with contextlib.suppress(Exception):
                pool.terminate()
            result.err(
                f"concurrency smoke timed out or failed: {type(exc).__name__}",
                integrity=True,
            )
            return

    ok_count = sum(1 for o in outcomes if o == "ok")
    if ok_count < 1:
        result.err(f"concurrency smoke: no successful publish ({outcomes})", integrity=True)
        return

    entry = entry_dir(cache_root, cache_key)
    if not entry.is_dir():
        result.err("concurrency smoke: final entry missing", integrity=True)
        return
    arts_dir = entry / "artifacts"
    manifest_path = entry / "cache_manifest.json"
    if not manifest_path.is_file():
        result.err("concurrency smoke: manifest missing", integrity=True)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("artifacts", []):
        from football_analytics.pipeline.types import ArtifactRef

        ref = ArtifactRef(
            logical_name=str(item["logical_name"]),
            relative_path=str(item["relative_path"]),
            media_type=str(item["media_type"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
            metadata=item.get("metadata") or {},
        )
        try:
            verify_artifact_on_disk(ref, root=arts_dir, reject_hardlinks=policy.reject_hardlinks)
        except Exception as exc:  # noqa: BLE001
            result.err(
                f"concurrency smoke: final entry invalid: {type(exc).__name__}",
                integrity=True,
            )
    result.extras["concurrency_smoke"] = True
    result.extras["concurrency_outcomes"] = list(outcomes)


def run_stage_cache_checks(
    *,
    project_root: Path | None = None,
    config: Path | None = None,
    paths_config: Path | None = None,
    synthetic: bool = False,
    corruption_smoke: bool = False,
    concurrency_smoke: bool = False,
    strict: bool = False,
) -> Result:
    """Run stage/cache validation checks and return a Result."""
    root = project_root or _project_root()
    policy_path = Path(config) if config else root / DEFAULT_POLICY
    paths_path = Path(paths_config) if paths_config else root / DEFAULT_PATHS
    if not policy_path.is_absolute():
        policy_path = (root / policy_path).resolve()
    if not paths_path.is_absolute():
        paths_path = (root / paths_path).resolve()

    result = Result()
    if not policy_path.is_file():
        result.err(f"cache policy missing: {policy_path}")
        return result.finalize(strict=strict)

    try:
        from football_analytics.pipeline.cache import load_cache_policy

        policy = load_cache_policy(policy_path)
        if policy.automatic_purge:
            result.err("automatic_purge must be false", integrity=True)
        result.extras["cache_policy_enabled"] = policy.enabled
    except Exception as exc:  # noqa: BLE001
        result.err(f"cache policy load failed: {type(exc).__name__}", integrity=True)
        return result.finalize(strict=strict)

    for rel in _PIPELINE_SCHEMAS + _CACHE_SCHEMAS:
        path = root / rel
        if not path.is_file():
            result.err(f"schema missing: {rel}", integrity=True)

    _check_import_smoke(result)
    _check_key_determinism(result)

    need_fixture = synthetic or corruption_smoke or concurrency_smoke
    fixture_root: Path | None = None
    if need_fixture:
        RUNTIME_ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fixture_root = RUNTIME_ROOT / "fixtures" / f"run_{stamp}"
        fixture_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        result.extras["fixture_root"] = str(fixture_root)

    try:
        if synthetic and fixture_root is not None:
            _run_synthetic(result, policy_path=policy_path, fixture_root=fixture_root)
        if corruption_smoke and fixture_root is not None:
            _run_corruption_smoke(result, policy_path=policy_path, fixture_root=fixture_root)
        if concurrency_smoke and fixture_root is not None:
            _run_concurrency_smoke(result, policy_path=policy_path, fixture_root=fixture_root)
    except Exception as exc:  # noqa: BLE001
        result.err(f"runtime checks failed: {type(exc).__name__}: {exc}", integrity=True)
    finally:
        if fixture_root is not None and fixture_root.exists():
            shutil.rmtree(fixture_root, ignore_errors=False)
            if fixture_root.exists():
                result.err("fixture cleanup incomplete", integrity=True)
            else:
                result.extras["fixture_cleaned"] = True

    # paths.yaml is optional for containment note only (do not use real cache root).
    if paths_path.is_file():
        result.extras["paths_config_present"] = True
    else:
        result.warn(f"paths config missing: {paths_path}")

    return result.finalize(strict=strict)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2D stage/cache validator")
    p.add_argument("--config", default=str(DEFAULT_POLICY))
    p.add_argument("--paths-config", default=str(DEFAULT_PATHS))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--corruption-smoke", action="store_true")
    p.add_argument("--concurrency-smoke", action="store_true")
    p.add_argument("--json-out")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_stage_cache_checks(
            config=Path(args.config),
            paths_config=Path(args.paths_config),
            synthetic=bool(args.synthetic),
            corruption_smoke=bool(args.corruption_smoke),
            concurrency_smoke=bool(args.concurrency_smoke),
            strict=bool(args.strict),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"status=FAIL exit_code={EXIT_CONFIG} error={type(exc).__name__}", file=sys.stderr)
        return EXIT_CONFIG

    payload = result.to_dict()
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        for w in result.warnings:
            print(f"WARNING: {w}")
        for e in result.errors:
            print(f"ERROR: {e}")
        print(f"status={result.status} exit_code={result.exit_code}")
    return int(result.exit_code)
