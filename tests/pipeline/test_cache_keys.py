#!/usr/bin/env python3
"""Cache key determinism and payload content tests (Stage 2D)."""

from __future__ import annotations

import unittest
from collections import OrderedDict

from football_analytics.pipeline.cache_key import CACHE_KEY_VERSION, compute_cache_key
from football_analytics.pipeline.exceptions import CacheError
from football_analytics.pipeline.stage import make_stage_identity
from football_analytics.pipeline.types import ArtifactRef, ContractRef

FP = "f" * 64
FP2 = "1" * 64
FP3 = "2" * 64


def _stage(**kwargs):
    base = dict(
        name="key_stage",
        version=1,
        code_fingerprint=FP,
        input_contracts=(),
        output_contracts=(),
        deterministic=True,
        cacheable=True,
    )
    base.update(kwargs)
    return make_stage_identity(**base)


def _ref(name: str, digest: str = FP2, path: str = "in.bin", size: int = 3) -> ArtifactRef:
    return ArtifactRef(
        logical_name=name,
        relative_path=path,
        media_type="application/octet-stream",
        size_bytes=size,
        sha256=digest,
    )


class CacheKeysTests(unittest.TestCase):
    def test_01_determinism(self) -> None:
        stage = _stage()
        inputs = {"Alpha": _ref("Alpha"), "Beta": _ref("Beta", digest=FP3, path="b.bin")}
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in k1))

    def test_02_mapping_order_independence(self) -> None:
        stage = _stage()
        a = _ref("Alpha")
        b = _ref("Beta", digest=FP3, path="b.bin")
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=OrderedDict([("Alpha", a), ("Beta", b)]),
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=OrderedDict([("Beta", b), ("Alpha", a)]),
        )
        self.assertEqual(k1, k2)

    def test_03_config_change_alters_key(self) -> None:
        stage = _stage()
        inputs = {"Alpha": _ref("Alpha")}
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP3,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        self.assertNotEqual(k1, k2)

    def test_04_input_change_alters_key(self) -> None:
        stage = _stage()
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs={"Alpha": _ref("Alpha", digest=FP2)},
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs={"Alpha": _ref("Alpha", digest=FP3)},
        )
        self.assertNotEqual(k1, k2)

    def test_05_code_fingerprint_change_alters_key(self) -> None:
        inputs = {"Alpha": _ref("Alpha")}
        k1 = compute_cache_key(
            stage=_stage(code_fingerprint=FP),
            config_fingerprint=FP2,
            compatibility_fingerprint=FP3,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=_stage(code_fingerprint=FP3),
            config_fingerprint=FP2,
            compatibility_fingerprint=FP3,
            inputs=inputs,
        )
        self.assertNotEqual(k1, k2)

    def test_06_compatibility_change_alters_key(self) -> None:
        stage = _stage()
        inputs = {"Alpha": _ref("Alpha")}
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP3,
            inputs=inputs,
        )
        self.assertNotEqual(k1, k2)

    def test_07_absolute_path_independence(self) -> None:
        """Same relative content yields same key regardless of absolute roots."""
        stage = _stage()
        # relative_path is what matters; absolute dirs are never in the payload
        ref_a = _ref("Alpha", path="subdir/file.bin")
        ref_b = ArtifactRef(
            logical_name="Alpha",
            relative_path="subdir/file.bin",
            media_type="application/octet-stream",
            size_bytes=3,
            sha256=FP2,
        )
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs={"Alpha": ref_a},
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs={"Alpha": ref_b},
        )
        self.assertEqual(k1, k2)

    def test_08_relative_path_change_alters_key(self) -> None:
        stage = _stage()
        k1 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs={"Alpha": _ref("Alpha", path="a.bin")},
        )
        k2 = compute_cache_key(
            stage=stage,
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs={"Alpha": _ref("Alpha", path="b.bin")},
        )
        self.assertNotEqual(k1, k2)

    def test_09_output_contracts_affect_key(self) -> None:
        inputs = {"Alpha": _ref("Alpha")}
        k1 = compute_cache_key(
            stage=_stage(output_contracts=()),
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=_stage(output_contracts=(ContractRef("events", 1),)),
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        self.assertNotEqual(k1, k2)

    def test_10_deterministic_flag_affects_key(self) -> None:
        inputs = {"Alpha": _ref("Alpha")}
        k1 = compute_cache_key(
            stage=_stage(deterministic=True),
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=_stage(deterministic=False),
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        self.assertNotEqual(k1, k2)

    def test_11_cacheable_flag_affects_key(self) -> None:
        inputs = {"Alpha": _ref("Alpha")}
        k1 = compute_cache_key(
            stage=_stage(cacheable=True),
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        k2 = compute_cache_key(
            stage=_stage(cacheable=False),
            config_fingerprint=FP,
            compatibility_fingerprint=FP2,
            inputs=inputs,
        )
        self.assertNotEqual(k1, k2)

    def test_12_version_constant(self) -> None:
        self.assertEqual(CACHE_KEY_VERSION, 1)

    def test_13_bad_config_fingerprint(self) -> None:
        with self.assertRaises(CacheError):
            compute_cache_key(
                stage=_stage(),
                config_fingerprint="short",
                compatibility_fingerprint=FP2,
                inputs={},
            )

    def test_14_bad_input_type(self) -> None:
        with self.assertRaises(CacheError):
            compute_cache_key(
                stage=_stage(),
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Alpha": "not-a-ref"},  # type: ignore[dict-item]
            )

    def test_15_no_timestamp_hostname_in_payload(self) -> None:
        from pathlib import Path

        import football_analytics.pipeline.cache_key as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        # Split off module docstring; inspect payload construction only.
        after_doc = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
        payload_block = after_doc.split("payload =", 1)[-1].split("return ", 1)[0]
        for forbidden in (
            "hostname",
            "requested_at",
            "timestamp",
            "username",
            "working_directory",
            "output_directory",
            "run_id",
        ):
            self.assertNotIn(f'"{forbidden}"', payload_block)


if __name__ == "__main__":
    unittest.main()
