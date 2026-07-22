"""Path/security and validation tests for Stage 3A."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.hashing import sha256_bytes
from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.types import (
    IngestMode,
    IngestRequest,
    ProvenanceInfo,
    SourceKind,
    VideoSource,
    VideoSourceError,
)
from football_analytics.video.validation import (
    assert_extension_allowed,
    assert_request_source_compatibility,
    assert_safe_output_root,
    assert_safe_source_path,
    reject_unsafe_path_string,
    verify_source_integrity,
)

REPO = default_repo_root()
RUNTIME = Path("/home/fdoblak/workspace/video_contract_checks")


class VideoSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(REPO / "configs/video/ingest_policy.yaml")
        RUNTIME.mkdir(parents=True, exist_ok=True)

    def test_reject_traversal_home_env_url_null(self) -> None:
        for bad in (
            "/home/fdoblak/workspace/video_contract_checks/../secret.mp4",
            "~/clip.mp4",
            "/tmp/${USER}/clip.mp4",
            "https://example.com/a.mp4",
            "/tmp/a\x00.mp4",
        ):
            with self.assertRaises(VideoSourceError):
                reject_unsafe_path_string(bad, label="path")

    def test_symlink_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as tmp:
            root = Path(tmp)
            target = root / "real.mp4"
            target.write_bytes(b"abc")
            link = root / "link.mp4"
            link.symlink_to(target)
            with self.assertRaises(VideoSourceError):
                assert_safe_source_path(str(link), contain_root=str(root), policy=self.policy)

    def test_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as tmp:
            root = Path(tmp)
            d = root / "notvideo"
            d.mkdir()
            with self.assertRaises(VideoSourceError):
                assert_safe_source_path(str(d), contain_root=str(root), policy=self.policy)

    def test_fifo_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as tmp:
            root = Path(tmp)
            fifo = root / "pipe.mp4"
            os.mkfifo(fifo)
            try:
                with self.assertRaises(VideoSourceError):
                    assert_safe_source_path(str(fifo), contain_root=str(root), policy=self.policy)
            finally:
                fifo.unlink(missing_ok=True)

    def test_source_output_collision_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as tmp:
            root = Path(tmp)
            src = root / "a.mp4"
            src.write_bytes(b"data")
            with self.assertRaises(VideoSourceError):
                assert_safe_output_root(
                    str(src),
                    contain_root=str(root),
                    source_path=str(src),
                    overwrite_allowed=False,
                )
            with self.assertRaises(VideoSourceError):
                assert_safe_output_root(
                    str(root / "out"),
                    contain_root=str(root),
                    source_path=str(src),
                    overwrite_allowed=True,
                )

    def test_hash_size_and_mutation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as tmp:
            root = Path(tmp)
            path = root / "clip.mp4"
            data = b"hello-stage3a"
            path.write_bytes(data)
            digest = sha256_bytes(data)
            ok = verify_source_integrity(
                path, expected_sha256=digest, expected_size_bytes=len(data)
            )
            self.assertEqual(ok.sha256, digest)
            with self.assertRaises(VideoSourceError):
                verify_source_integrity(
                    path, expected_sha256="0" * 64, expected_size_bytes=len(data)
                )
            with self.assertRaises(VideoSourceError):
                verify_source_integrity(path, expected_sha256=digest, expected_size_bytes=1)

    def test_extension_allowlist(self) -> None:
        path = Path("/home/fdoblak/workspace/video_contract_checks/x.mp4")
        self.assertEqual(assert_extension_allowed(path, self.policy), ".mp4")
        with self.assertRaises(VideoSourceError):
            assert_extension_allowed(Path("x.exe"), self.policy)

    def test_fixture_mode_not_on_user_source(self) -> None:
        sha = "e" * 64
        source = VideoSource(
            source_id="src_user_one",
            source_kind=SourceKind.USER_LOCAL_VIDEO,
            original_filename="match.mp4",
            source_path="/home/fdoblak/workspace/video_contract_checks/match.mp4",
            source_size_bytes=10,
            source_sha256=sha,
            media_type="video/mp4",
            container_hint="mp4",
            created_at_utc="2026-07-22T21:00:00Z",
            registered_at_utc="2026-07-22T21:00:01Z",
            immutability_policy="immutable_source",
            provenance=ProvenanceInfo(origin="local_file", label="user"),
        )
        request = IngestRequest(
            request_id="req_user_one",
            run_id="run_20260722T210000000000Z_aaaaaaaaaaaa",
            source_id="src_user_one",
            source_path=source.source_path,
            requested_at_utc="2026-07-22T21:00:02Z",
            ingest_mode=IngestMode.VALIDATE_ONLY,
            policy_version=self.policy["policy_version"],
            probe_requested=False,
            normalization_requested=False,
            expected_source_sha256=sha,
            expected_source_size_bytes=10,
            output_root="/home/fdoblak/workspace/video_contract_checks/out",
            fixture_mode=True,
        )
        with self.assertRaises(VideoSourceError):
            assert_request_source_compatibility(request, source, self.policy)

    def test_regular_file_under_root_ok(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(RUNTIME)) as tmp:
            root = Path(tmp)
            path = root / "ok.mp4"
            path.write_bytes(b"ok")
            resolved = assert_safe_source_path(
                str(path), contain_root=str(root), policy=self.policy
            )
            self.assertTrue(stat.S_ISREG(resolved.stat().st_mode))


if __name__ == "__main__":
    unittest.main()
