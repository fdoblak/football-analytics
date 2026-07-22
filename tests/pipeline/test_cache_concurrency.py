#!/usr/bin/env python3
"""Cache concurrent publish tests (Stage 2D)."""

from __future__ import annotations

import multiprocessing as mp
import tempfile
import traceback
import unittest
from pathlib import Path
from types import MappingProxyType

from football_analytics.core.hashing import sha256_bytes
from football_analytics.core.run_id import generate_run_id
from football_analytics.pipeline.artifacts import build_artifact_ref
from football_analytics.pipeline.cache import (
    entry_dir,
    load_cache_policy,
    publish_cache_entry,
    verify_cache_entry,
)
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.stage import make_stage_identity
from football_analytics.pipeline.types import ArtifactRef, StageResult

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs" / "system" / "cache_policy.yaml"
FP = "a" * 64
FP2 = "b" * 64


def _worker(args: tuple) -> str:
    """Publish the same key from a child process; return 'ok' or error text."""
    cache_root_s, art_root_s, key, run_id, policy_path_s = args
    try:
        cache_root = Path(cache_root_s)
        art_root = Path(art_root_s)
        policy = load_cache_policy(Path(policy_path_s))
        ref = ArtifactRef(
            logical_name="Out",
            relative_path="out.bin",
            media_type="application/octet-stream",
            size_bytes=(art_root / "out.bin").stat().st_size,
            sha256=sha256_bytes((art_root / "out.bin").read_bytes()),
        )
        stage = make_stage_identity(
            name="conc_stage",
            version=1,
            code_fingerprint=FP,
            deterministic=True,
            cacheable=True,
        )
        result = StageResult(
            run_id=run_id,
            stage_name="conc_stage",
            stage_version=1,
            status="succeeded",
            cache_key=key,
            cache_hit=False,
            started_at_utc="2026-07-22T12:00:00.000000Z",
            finished_at_utc="2026-07-22T12:00:01.000000Z",
            duration_ms=1,
            inputs={},
            outputs={"Out": ref.to_dict()},
            metrics={},
            warnings=(),
            error=None,
            execution_fingerprint=FP2,
        )
        publish_cache_entry(
            cache_root=cache_root,
            cache_key=key,
            stage_identity=stage,
            config_fingerprint=FP,
            artifacts=MappingProxyType({"Out": ref}),
            artifact_root=art_root,
            stage_result=result,
            policy=policy,
            source_run_id=run_id,
        )
        return "ok"
    except Exception:  # noqa: BLE001
        return traceback.format_exc()


class CacheConcurrencyTests(unittest.TestCase):
    def test_01_two_processes_single_valid_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            art_root = root / "arts"
            art_root.mkdir()
            path = art_root / "out.bin"
            path.write_bytes(b"concurrent-payload")
            ref = build_artifact_ref(
                "Out", path, root=art_root, media_type="application/octet-stream"
            )
            stage = make_stage_identity(
                name="conc_stage",
                version=1,
                code_fingerprint=FP,
                deterministic=True,
                cacheable=True,
            )
            key = compute_cache_key(
                stage=stage,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Out": ref},
            )
            run_id = generate_run_id()
            payload = (
                str(cache_root),
                str(art_root),
                key,
                run_id,
                str(POLICY_PATH),
            )
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=2) as pool:
                results = pool.map(_worker, [payload, payload])
            self.assertTrue(all(r == "ok" for r in results), results)
            entry = entry_dir(cache_root, key)
            self.assertTrue(entry.is_dir())
            # Exactly one cache entry directory for this key
            shard = cache_root / "v1" / "sha256" / key[:2]
            siblings = [p for p in shard.iterdir() if p.name == key[2:]]
            self.assertEqual(len(siblings), 1)
            policy = load_cache_policy(POLICY_PATH)
            manifest, _ = verify_cache_entry(
                cache_root,
                key,
                expected_stage=stage,
                expected_config_fp=FP,
                expected_inputs={"Out": ref},
                expected_compatibility_fp=FP2,
                policy=policy,
            )
            self.assertEqual(manifest["cache_key"], key)
            art = entry / "artifacts" / "out.bin"
            self.assertEqual(art.read_bytes(), b"concurrent-payload")
            # No leftover tmp publish dirs
            leftovers = list(cache_root.glob(".tmp_publish_*"))
            self.assertEqual(leftovers, [])

    def test_02_threads_acquire_lock(self) -> None:
        import threading

        from football_analytics.pipeline.cache import acquire_key_lock

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            key = "c" * 64
            held = []
            errors = []

            def run(idx: int) -> None:
                try:
                    with acquire_key_lock(cache_root, key, timeout_seconds=5.0):
                        held.append(idx)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertEqual(errors, [])
            self.assertEqual(sorted(held), [0, 1])


if __name__ == "__main__":
    unittest.main()
