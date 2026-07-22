"""Additional validation / policy / receipt tests for Stage 3A."""

from __future__ import annotations

import unittest

from football_analytics.video.contracts import (
    default_repo_root,
    load_ingest_policy,
)
from football_analytics.video.types import (
    ContractFingerprints,
    IngestReceipt,
    Issue,
    ReceiptProvenance,
    ReceiptStatus,
    VideoContractError,
    VideoPolicyError,
)
from football_analytics.video.validation import (
    assert_dimensions_within_policy,
    assert_duration_within_policy,
    assert_size_within_policy,
)


class VideoValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(default_repo_root() / "configs/video/ingest_policy.yaml")

    def test_policy_size_bounds(self) -> None:
        assert_size_within_policy(1, self.policy)
        with self.assertRaises(VideoPolicyError):
            assert_size_within_policy(
                int(self.policy["maximum_source_size_bytes"]) + 1, self.policy
            )

    def test_policy_dimensions(self) -> None:
        assert_dimensions_within_policy(1920, 1080, self.policy)
        with self.assertRaises(VideoPolicyError):
            assert_dimensions_within_policy(10, 10, self.policy)

    def test_unknown_duration_allowed(self) -> None:
        assert_duration_within_policy(None, self.policy)
        assert_duration_within_policy(1_000_000, self.policy)
        with self.assertRaises(VideoPolicyError):
            assert_duration_within_policy(1, self.policy)

    def test_receipt_warning_error_separation(self) -> None:
        receipt = IngestReceipt(
            receipt_id="rcpt_val_one",
            request_id="req_val_one",
            run_id="run_20260722T210000000000Z_bbbbbbbbbbbb",
            source_id="src_val_one",
            source_sha256="a" * 64,
            source_size_bytes=1,
            status=ReceiptStatus.REJECTED,
            started_at_utc="2026-07-22T21:00:00Z",
            completed_at_utc="2026-07-22T21:00:01Z",
            probe_record_ref=None,
            normalize_plan_ref=None,
            artifact_refs=(),
            policy_version=self.policy["policy_version"],
            contract_fingerprints=ContractFingerprints(source="a" * 64, request="b" * 64),
            warnings=(Issue(code="soft", message="warn"),),
            errors=(Issue(code="hard", message="fail"),),
            provenance=ReceiptProvenance(stage="3A", label="reject"),
        )
        payload = receipt.to_dict()
        self.assertEqual(len(payload["warnings"]), 1)
        self.assertEqual(len(payload["errors"]), 1)
        self.assertNotEqual(payload["status"], "succeeded")

    def test_validated_cannot_carry_errors(self) -> None:
        with self.assertRaises(VideoContractError):
            IngestReceipt(
                receipt_id="rcpt_val_two",
                request_id="req_val_two",
                run_id="run_20260722T210000000000Z_cccccccccccc",
                source_id="src_val_two",
                source_sha256="a" * 64,
                source_size_bytes=1,
                status=ReceiptStatus.VALIDATED,
                started_at_utc="2026-07-22T21:00:00Z",
                completed_at_utc="2026-07-22T21:00:01Z",
                probe_record_ref=None,
                normalize_plan_ref=None,
                artifact_refs=(),
                policy_version=self.policy["policy_version"],
                contract_fingerprints=ContractFingerprints(source="a" * 64, request="b" * 64),
                warnings=(),
                errors=(Issue(code="x", message="y"),),
                provenance=ReceiptProvenance(stage="3A", label="bad"),
            )


if __name__ == "__main__":
    unittest.main()
