#!/usr/bin/env python3
"""ArtifactRef build/verify/copy tests (Stage 2D)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.hashing import sha256_bytes, sha256_file
from football_analytics.pipeline.artifacts import (
    build_artifact_ref,
    copy_artifact_file,
    is_hardlinked,
    verify_artifact_on_disk,
)
from football_analytics.pipeline.exceptions import ArtifactError, PipelineError
from football_analytics.pipeline.types import ArtifactRef, validate_safe_relative_path

FP = "e" * 64


class ArtifactRefsTests(unittest.TestCase):
    def test_01_valid_ref_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.bin"
            path.write_bytes(b"abc")
            ref = build_artifact_ref("Data", path, root=root, media_type="application/octet-stream")
            self.assertEqual(ref.logical_name, "Data")
            self.assertEqual(ref.relative_path, "data.bin")
            self.assertEqual(ref.size_bytes, 3)
            self.assertEqual(ref.sha256, sha256_bytes(b"abc"))
            verify_artifact_on_disk(ref, root=root)

    def test_02_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.bin"
            path.write_bytes(b"abc")
            ref = build_artifact_ref("Data", path, root=root, media_type="application/octet-stream")
            path.write_bytes(b"xyz")
            with self.assertRaises(ArtifactError):
                verify_artifact_on_disk(ref, root=root)

    def test_03_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.bin"
            path.write_bytes(b"abc")
            bad = ArtifactRef(
                logical_name="Data",
                relative_path="data.bin",
                media_type="application/octet-stream",
                size_bytes=99,
                sha256=sha256_file(path),
            )
            with self.assertRaises(ArtifactError):
                verify_artifact_on_disk(bad, root=root)

    def test_04_path_traversal_rejected(self) -> None:
        with self.assertRaises(ArtifactError):
            validate_safe_relative_path("../x")
        with self.assertRaises(ArtifactError):
            validate_safe_relative_path("/abs")
        with self.assertRaises(ArtifactError):
            validate_safe_relative_path("a\\b")
        with self.assertRaises(ArtifactError):
            validate_safe_relative_path("..")

    def test_05_escape_root_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside.bin"
            root.mkdir()
            outside.write_bytes(b"x")
            with self.assertRaises(ArtifactError):
                build_artifact_ref(
                    "Data", outside, root=root, media_type="application/octet-stream"
                )

    def test_06_symlink_rejected_on_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.bin"
            link = root / "link.bin"
            real.write_bytes(b"abc")
            link.symlink_to(real)
            with self.assertRaises(ArtifactError):
                build_artifact_ref("Data", link, root=root, media_type="application/octet-stream")

    def test_07_symlink_escape_rejected_on_verify(self) -> None:
        # resolve() follows symlinks; escape outside root must still be rejected.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            root.mkdir()
            outside = base / "outside.bin"
            outside.write_bytes(b"abc")
            link = root / "link.bin"
            link.symlink_to(outside)
            ref = ArtifactRef(
                logical_name="Data",
                relative_path="link.bin",
                media_type="application/octet-stream",
                size_bytes=3,
                sha256=sha256_bytes(b"abc"),
            )
            with self.assertRaises(ArtifactError):
                verify_artifact_on_disk(ref, root=root)

    def test_08_noncanonical_hash_rejected(self) -> None:
        with self.assertRaises((ArtifactError, PipelineError)):
            ArtifactRef(
                logical_name="Data",
                relative_path="data.bin",
                media_type="application/octet-stream",
                size_bytes=1,
                sha256="A" * 64,
            )

    def test_09_secret_metadata_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.bin"
            path.write_bytes(b"abc")
            with self.assertRaises(ArtifactError):
                build_artifact_ref(
                    "Data",
                    path,
                    root=root,
                    media_type="application/octet-stream",
                    metadata={"api_token": "secret"},
                )

    def test_10_nested_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "sub" / "data.bin"
            nested.parent.mkdir()
            nested.write_bytes(b"nested")
            ref = build_artifact_ref(
                "Data", nested, root=root, media_type="application/octet-stream"
            )
            self.assertEqual(ref.relative_path, "sub/data.bin")
            verify_artifact_on_disk(ref, root=root)

    def test_11_copy_artifact_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src.bin"
            dst = root / "dst.bin"
            src.write_bytes(b"copy-me")
            copy_artifact_file(src, dst)
            self.assertEqual(dst.read_bytes(), b"copy-me")
            self.assertEqual(dst.stat().st_mode & 0o777, 0o600)

    def test_12_copy_rejects_existing_dest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src.bin"
            dst = root / "dst.bin"
            src.write_bytes(b"a")
            dst.write_bytes(b"b")
            with self.assertRaises(ArtifactError):
                copy_artifact_file(src, dst)

    def test_13_copy_rejects_symlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.bin"
            link = root / "link.bin"
            real.write_bytes(b"a")
            link.symlink_to(real)
            with self.assertRaises(ArtifactError):
                copy_artifact_file(link, root / "dst.bin")

    def test_14_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ArtifactError):
                build_artifact_ref(
                    "Data",
                    root / "missing.bin",
                    root=root,
                    media_type="application/octet-stream",
                )

    def test_15_parquet_requires_contract_fields(self) -> None:
        with self.assertRaises(ArtifactError):
            ArtifactRef(
                logical_name="Table",
                relative_path="t.parquet",
                media_type="application/vnd.apache.parquet",
                size_bytes=1,
                sha256=FP,
            )

    def test_16_parquet_with_contract_ok(self) -> None:
        ref = ArtifactRef(
            logical_name="Table",
            relative_path="t.parquet",
            media_type="application/vnd.apache.parquet",
            size_bytes=1,
            sha256=FP,
            contract_name="detections",
            contract_version=1,
            schema_fingerprint=FP,
        )
        self.assertEqual(ref.contract_name, "detections")

    def test_17_hardlink_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.bin"
            b = root / "b.bin"
            a.write_bytes(b"hardlink")
            try:
                os.link(a, b)
            except OSError:
                self.skipTest("hardlink not allowed on this filesystem")
            self.assertTrue(is_hardlinked(a))
            ref = build_artifact_ref("Data", a, root=root, media_type="application/octet-stream")
            with self.assertRaises(ArtifactError):
                verify_artifact_on_disk(ref, root=root, reject_hardlinks=True)

    def test_18_safe_relative_normalizes_dot(self) -> None:
        self.assertEqual(validate_safe_relative_path("./a/b"), "a/b")

    def test_19_control_chars_rejected(self) -> None:
        with self.assertRaises(ArtifactError):
            validate_safe_relative_path("a\nb")

    def test_20_to_dict_roundtrip_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.bin"
            path.write_bytes(b"z")
            ref = build_artifact_ref(
                "Data",
                path,
                root=root,
                media_type="application/octet-stream",
                metadata={"note": "ok"},
            )
            d = ref.to_dict()
            self.assertEqual(d["logical_name"], "Data")
            self.assertEqual(d["metadata"]["note"], "ok")
            self.assertNotIn("password", d["metadata"])


if __name__ == "__main__":
    unittest.main()
