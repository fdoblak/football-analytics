"""Unit tests for Stage 3D / 3D-F1 time mapping taxonomy."""

from __future__ import annotations

import hashlib
import unittest

import jsonschema

from football_analytics.core.run_id import generate_run_id
from football_analytics.video.contracts import (
    default_repo_root,
    load_all_video_schemas,
    validate_payload_against_schema,
)
from football_analytics.video.time_mapping import (
    MappingEvidence,
    MappingStats,
    classify_mapping_quality,
    duration_ts_to_us,
    empty_mapping_evidence,
    pts_to_video_time_us,
)
from football_analytics.video.types import (
    FRAME_TIMELINE_RECEIPT_SCHEMA_VERSION,
    FrameRateMode,
    FrameTimelineCleanup,
    FrameTimelineMode,
    FrameTimelineProvenance,
    FrameTimelineReceipt,
    FrameTimelineStatus,
    MappingQuality,
    Rational,
    VideoContractError,
    coerce_mapping_quality,
    normalize_legacy_receipt_payload,
)

REPO = default_repo_root()
FRAMES_JSON = REPO / "schemas/data/v1/frames.json"
# Locked at video-ingest-v0.3.0; must not change for 3D-F1.
FRAMES_JSON_SHA256 = "8fd233af820aa6242d575a2005f6552e43b37dd535140f26858782cc116d0437"


def _clean_stats(**kwargs: int | bool) -> MappingStats:
    base = MappingStats(frame_count=10, ok_count=10)
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def _evidence(**kwargs: object) -> MappingEvidence:
    base = dict(
        has_normalization_receipt=True,
        normalization_status="succeeded",
        frame_rate_conversion_performed=False,
        frame_rate_conversion_source_mode="cfr",
        frame_rate_conversion_target_mode="cfr",
        requires_stage3d_mapping=False,
        duration_drift_us=0,
        applied_transforms=("transcode_video",),
        constant_offset_us=None,
        identity_proven=False,
    )
    base.update(kwargs)
    return MappingEvidence(**base)  # type: ignore[arg-type]


def _minimal_receipt_dict(
    *,
    schema_version: int = FRAME_TIMELINE_RECEIPT_SCHEMA_VERSION,
    mapping_quality: str = "uncertain",
    status: str = "succeeded",
) -> dict:
    sha = "a" * 64
    errors: list[dict[str, str]] = []
    if status in {"failed", "rejected"}:
        errors = [{"code": "X", "message": "failed"}]
    return {
        "schema_version": schema_version,
        "receipt_id": "ftl_test_one",
        "run_id": generate_run_id(),
        "video_id": "vid_test_one",
        "source_path": "/home/fdoblak/workspace/frame_timeline_checks/a.mp4",
        "source_sha256": sha,
        "normalization_receipt_path": None,
        "mode": "timeline_only",
        "status": status,
        "started_at_utc": "2026-07-23T08:00:00Z",
        "completed_at_utc": "2026-07-23T08:00:01Z",
        "ffprobe_path": "/usr/bin/ffprobe",
        "ffprobe_version": "4.4.2",
        "video_stream_index": 0,
        "time_base": {"numerator": 1, "denominator": 12800},
        "frame_rate_mode": "cfr",
        "frames_parquet": (
            None
            if status != "succeeded"
            else "/home/fdoblak/workspace/frame_timeline_checks/frames.parquet"
        ),
        "frames_parquet_sha256": None if status != "succeeded" else sha,
        "frame_count": 0 if status != "succeeded" else 10,
        "ok_count": 0 if status != "succeeded" else 10,
        "skipped_count": 0,
        "failed_count": 0,
        "unknown_count": 0,
        "missing_pts_count": 0,
        "duplicate_pts_count": 0,
        "non_monotonic_pts_count": 0,
        "mapping_quality": mapping_quality,
        "sample_every": None,
        "materialized": False,
        "materialized_frame_count": None,
        "artifact_manifest": None,
        "warnings": [],
        "errors": errors,
        "cleanup": {"temp_removed": True},
        "provenance": {"stage": "3D", "label": "frame_timeline", "notes": None},
    }


class TimeMappingTests(unittest.TestCase):
    def test_rational_pts_mapping(self) -> None:
        tb = Rational(1, 12800)
        self.assertEqual(pts_to_video_time_us(512, tb), 40000)
        self.assertEqual(pts_to_video_time_us(0, tb), 0)

    def test_duration_ts(self) -> None:
        tb = Rational(1, 12800)
        self.assertEqual(duration_ts_to_us(512, tb), 40000)
        self.assertIsNone(duration_ts_to_us(None, tb))
        self.assertIsNone(duration_ts_to_us(0, tb))

    def test_rejects_negative_pts(self) -> None:
        with self.assertRaises(VideoContractError):
            pts_to_video_time_us(-1, Rational(1, 25))

    def test_quality_exact_identity(self) -> None:
        stats = _clean_stats()
        ev = _evidence(
            normalization_status="skipped",
            applied_transforms=(),
            identity_proven=True,
            duration_drift_us=0,
        )
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.EXACT_IDENTITY,
        )

    def test_quality_timestamp_preserved(self) -> None:
        stats = _clean_stats()
        ev = _evidence(identity_proven=False, frame_rate_conversion_performed=False)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.TIMESTAMP_PRESERVED,
        )

    def test_quality_constant_positive_offset(self) -> None:
        stats = _clean_stats()
        ev = _evidence(constant_offset_us=40_000, identity_proven=False)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.DERIVED_WITH_CONSTANT_OFFSET,
        )

    def test_quality_constant_negative_offset(self) -> None:
        stats = _clean_stats()
        ev = _evidence(constant_offset_us=-1_000, identity_proven=False)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.DERIVED_WITH_CONSTANT_OFFSET,
        )

    def test_quality_resampling_when_conversion_performed(self) -> None:
        stats = _clean_stats()
        ev = _evidence(
            frame_rate_conversion_performed=True,
            applied_transforms=("force_cfr",),
            requires_stage3d_mapping=True,
        )
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.DERIVED_WITH_RESAMPLING,
        )

    def test_quality_vfr_to_cfr_resampling(self) -> None:
        stats = _clean_stats()
        ev = _evidence(
            frame_rate_conversion_performed=True,
            frame_rate_conversion_source_mode="vfr",
            frame_rate_conversion_target_mode="cfr",
            requires_stage3d_mapping=True,
            applied_transforms=("force_cfr",),
        )
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.DERIVED_WITH_RESAMPLING,
        )
        # VFR mode alone never claims exact_identity / timestamp_preserved
        self.assertNotIn(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.VFR, evidence=ev),
            {MappingQuality.EXACT_IDENTITY, MappingQuality.TIMESTAMP_PRESERVED},
        )

    def test_quality_missing_norm_receipt_uncertain(self) -> None:
        stats = _clean_stats()
        self.assertEqual(
            classify_mapping_quality(
                stats, frame_rate_mode=FrameRateMode.CFR, evidence=empty_mapping_evidence()
            ),
            MappingQuality.UNCERTAIN,
        )

    def test_quality_conflicting_metadata_uncertain(self) -> None:
        stats = _clean_stats()
        ev = _evidence(
            identity_proven=True,
            frame_rate_conversion_performed=False,
            duration_drift_us=50_000,
        )
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.UNCERTAIN,
        )
        ev2 = _evidence(
            frame_rate_conversion_performed=False,
            frame_rate_conversion_target_mode="cfr",
            identity_proven=False,
        )
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.VFR, evidence=ev2),
            MappingQuality.UNCERTAIN,
        )

    def test_quality_not_available_on_invention(self) -> None:
        stats = MappingStats(frame_count=5, ok_count=5, invented_from_index_or_fps=True)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR),
            MappingQuality.NOT_AVAILABLE,
        )

    def test_quality_not_available_no_frames(self) -> None:
        stats = MappingStats(frame_count=0)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR),
            MappingQuality.NOT_AVAILABLE,
        )

    def test_quality_incomplete_pts_uncertain(self) -> None:
        stats = MappingStats(frame_count=10, ok_count=9, skipped_count=1, missing_pts_count=1)
        ev = _evidence()
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR, evidence=ev),
            MappingQuality.UNCERTAIN,
        )

    def test_quality_significant_non_monotonic_resampling(self) -> None:
        stats = MappingStats(frame_count=10, ok_count=10, non_monotonic_pts_count=3)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR),
            MappingQuality.DERIVED_WITH_RESAMPLING,
        )

    def test_failed_execution_quality_is_not_available_enum(self) -> None:
        self.assertEqual(MappingQuality.NOT_AVAILABLE.value, "not_available")
        self.assertNotIn("failed", {q.value for q in MappingQuality})
        receipt = FrameTimelineReceipt.from_dict(
            _minimal_receipt_dict(
                mapping_quality="not_available",
                status="failed",
            )
        )
        self.assertEqual(receipt.status, FrameTimelineStatus.FAILED)
        self.assertEqual(receipt.mapping_quality, MappingQuality.NOT_AVAILABLE)

    def test_legacy_coercion(self) -> None:
        for legacy in ("exact", "good", "degraded", "unreliable"):
            self.assertEqual(
                coerce_mapping_quality(legacy, schema_version=1),
                MappingQuality.UNCERTAIN,
            )
        self.assertEqual(
            coerce_mapping_quality("failed", schema_version=1),
            MappingQuality.NOT_AVAILABLE,
        )

    def test_v2_unknown_rejected(self) -> None:
        with self.assertRaises(VideoContractError):
            coerce_mapping_quality("exact", schema_version=2)
        with self.assertRaises(VideoContractError):
            coerce_mapping_quality("not_a_quality", schema_version=2)

    def test_schema_type_roundtrip_v2(self) -> None:
        schemas = load_all_video_schemas(REPO / "schemas/video")
        payload = _minimal_receipt_dict(mapping_quality="timestamp_preserved")
        validate_payload_against_schema(payload, schemas["frame_timeline_receipt.schema.json"])
        receipt = FrameTimelineReceipt.from_dict(payload)
        self.assertEqual(receipt.schema_version, 2)
        again = receipt.to_dict()
        self.assertEqual(again["schema_version"], 2)
        self.assertEqual(again["mapping_quality"], "timestamp_preserved")
        validate_payload_against_schema(again, schemas["frame_timeline_receipt.schema.json"])
        FrameTimelineReceipt.from_dict(again)

    def test_legacy_v030_payload_read(self) -> None:
        legacy = _minimal_receipt_dict(schema_version=1, mapping_quality="exact")
        receipt = FrameTimelineReceipt.from_dict(legacy)
        self.assertEqual(receipt.mapping_quality, MappingQuality.UNCERTAIN)
        self.assertEqual(receipt.schema_version, 2)
        self.assertEqual(receipt.to_dict()["schema_version"], 2)
        normalized = normalize_legacy_receipt_payload(legacy)
        self.assertEqual(normalized["schema_version"], 2)
        self.assertEqual(normalized["mapping_quality"], "uncertain")
        schemas = load_all_video_schemas(REPO / "schemas/video")
        validate_payload_against_schema(normalized, schemas["frame_timeline_receipt.schema.json"])

    def test_new_payload_validation_rejects_legacy_quality(self) -> None:
        schemas = load_all_video_schemas(REPO / "schemas/video")
        payload = _minimal_receipt_dict(mapping_quality="exact")
        with self.assertRaises(jsonschema.ValidationError):
            validate_payload_against_schema(payload, schemas["frame_timeline_receipt.schema.json"])

    def test_frames_json_unchanged(self) -> None:
        digest = hashlib.sha256(FRAMES_JSON.read_bytes()).hexdigest()
        self.assertEqual(digest, FRAMES_JSON_SHA256)
        data = FRAMES_JSON.read_text(encoding="utf-8")
        self.assertIn('"contract_name": "frames"', data)
        self.assertIn('"version": 1', data)

    def test_receipt_dataclass_construction(self) -> None:
        receipt = FrameTimelineReceipt(
            receipt_id="ftl_build_one",
            run_id=generate_run_id(),
            video_id="vid_build_one",
            source_path="/home/fdoblak/workspace/frame_timeline_checks/a.mp4",
            source_sha256="b" * 64,
            mode=FrameTimelineMode.TIMELINE_ONLY,
            status=FrameTimelineStatus.SUCCEEDED,
            started_at_utc="2026-07-23T08:00:00Z",
            completed_at_utc="2026-07-23T08:00:01Z",
            ffprobe_path="/usr/bin/ffprobe",
            ffprobe_version="4.4.2",
            video_stream_index=0,
            time_base=Rational(1, 25),
            frame_rate_mode=FrameRateMode.CFR,
            frames_parquet="/home/fdoblak/workspace/frame_timeline_checks/frames.parquet",
            frames_parquet_sha256="b" * 64,
            frame_count=1,
            ok_count=1,
            skipped_count=0,
            failed_count=0,
            unknown_count=0,
            missing_pts_count=0,
            duplicate_pts_count=0,
            non_monotonic_pts_count=0,
            mapping_quality=MappingQuality.EXACT_IDENTITY,
            materialized=False,
            artifact_manifest=None,
            warnings=(),
            errors=(),
            cleanup=FrameTimelineCleanup(temp_removed=True),
            provenance=FrameTimelineProvenance(stage="3D", label="frame_timeline"),
        )
        self.assertEqual(receipt.schema_version, FRAME_TIMELINE_RECEIPT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
