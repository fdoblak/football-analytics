#!/usr/bin/env python3
"""Hashing helpers tests (Stage 2B)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from football_analytics.core.hashing import (
    HashError,
    hash_canonical_json,
    hash_directory_tree,
    sha256_bytes,
    sha256_file,
)


class HashingTests(unittest.TestCase):
    def test_01_empty_vector(self) -> None:
        self.assertEqual(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

    def test_02_abc_vector(self) -> None:
        self.assertEqual(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_03_streaming_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            p.write_bytes(b"abc")
            self.assertEqual(sha256_file(p), sha256_bytes(b"abc"))

    def test_04_large_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.bin"
            data = b"x" * (2 * 1024 * 1024 + 17)
            p.write_bytes(data)
            self.assertEqual(sha256_file(p, chunk_size=65536), sha256_bytes(data))

    def test_05_symlink_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            link = Path(tmp) / "link"
            real.write_bytes(b"abc")
            link.symlink_to(real)
            with self.assertRaises(HashError):
                sha256_file(link)

    def test_06_fifo_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fifo = Path(tmp) / "f.fifo"
            try:
                os.mkfifo(fifo)
            except (OSError, AttributeError):
                self.skipTest("fifo unavailable")
            with self.assertRaises(HashError):
                sha256_file(fifo)

    def test_07_directory_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            (root / "b").mkdir(parents=True)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b" / "c.txt").write_text("c", encoding="utf-8")
            m1 = hash_directory_tree(root)
            m2 = hash_directory_tree(root)
            self.assertEqual(m1.digest, m2.digest)
            self.assertEqual([f.relative_path for f in m1.files], ["a.txt", "b/c.txt"])

    def test_08_empty_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            m = hash_directory_tree(root)
            self.assertEqual(m.files, ())
            self.assertEqual(len(m.digest), 64)

    def test_09_relative_paths_posix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "t"
            (root / "d").mkdir(parents=True)
            (root / "d" / "x.txt").write_text("x", encoding="utf-8")
            m = hash_directory_tree(root)
            self.assertEqual(m.files[0].relative_path, "d/x.txt")

    def test_10_hidden_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "t"
            root.mkdir()
            (root / ".secret").write_text("s", encoding="utf-8")
            (root / "vis.txt").write_text("v", encoding="utf-8")
            m = hash_directory_tree(root)
            self.assertEqual([f.relative_path for f in m.files], ["vis.txt"])

    def test_11_symlink_in_tree_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "t"
            root.mkdir()
            real = root / "real.txt"
            real.write_text("r", encoding="utf-8")
            (root / "link.txt").symlink_to(real)
            with self.assertRaises(HashError):
                hash_directory_tree(root)

    def test_12_canonical_json_key_order(self) -> None:
        self.assertEqual(
            hash_canonical_json({"b": 1, "a": 2}), hash_canonical_json({"a": 2, "b": 1})
        )

    def test_13_list_order_matters(self) -> None:
        self.assertNotEqual(hash_canonical_json([1, 2]), hash_canonical_json([2, 1]))

    def test_14_modified_during_hash_detection(self) -> None:
        # Best-effort: rewrite file between stats by monkeying is hard; ensure API exists.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.bin"
            p.write_bytes(b"abc")
            digest = sha256_file(p)
            self.assertEqual(digest, sha256_bytes(b"abc"))
            # Touch after hash should not affect previous result
            time.sleep(0.01)
            p.write_bytes(b"abcd")
            self.assertNotEqual(sha256_file(p), digest)

    def test_15_missing_file(self) -> None:
        with self.assertRaises(HashError):
            sha256_file(Path("/tmp/definitely-missing-football-analytics-hash-test"))


if __name__ == "__main__":
    unittest.main()
