"""Schema and contract integration tests for Stage 3A."""

from __future__ import annotations

import json
import unittest

import jsonschema

from football_analytics.core.run_id import generate_run_id
from football_analytics.video.contracts import (
    SCHEMA_FILES,
    build_normalize_plan,
    default_repo_root,
    load_all_video_schemas,
    load_ingest_policy,
    validate_payload_against_schema,
)
from football_analytics.video.fixtures import metadata_fixture
from football_analytics.video.types import (
    IngestRequest,
    Rational,
    VideoSource,
)

REPO = default_repo_root()


class VideoContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = load_all_video_schemas(REPO / "schemas/video")
        cls.policy = load_ingest_policy(REPO / "configs/video/ingest_policy.yaml")

    def test_five_schemas_parse(self) -> None:
        self.assertEqual(set(self.schemas), set(SCHEMA_FILES))
        for _name, schema in self.schemas.items():
            self.assertEqual(schema.get("schema_version", {}).get("const", 1), 1)
            self.assertIn("$id", schema)
            self.assertFalse(schema.get("additionalProperties", True))

    def test_schema_ids_unique(self) -> None:
        ids = [s["$id"] for s in self.schemas.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_valid_source_and_request(self) -> None:
        sha = "a" * 64
        source = {
            "schema_version": 1,
            "source_id": "src_contract_one",
            "source_kind": "synthetic_fixture",
            "original_filename": "a.mp4",
            "source_path": "/home/fdoblak/workspace/video_contract_checks/a.mp4",
            "source_size_bytes": 10,
            "source_sha256": sha,
            "media_type": "video/mp4",
            "container_hint": "mp4",
            "created_at_utc": "2026-07-22T21:00:00Z",
            "registered_at_utc": "2026-07-22T21:00:01Z",
            "immutability_policy": "immutable_source",
            "provenance": {
                "origin": "synthetic_generated",
                "label": "t",
                "notes": None,
            },
        }
        validate_payload_against_schema(source, self.schemas["video_source.schema.json"])
        VideoSource.from_dict(source)
        request = {
            "schema_version": 1,
            "request_id": "req_contract_one",
            "run_id": generate_run_id(),
            "source_id": "src_contract_one",
            "source_path": source["source_path"],
            "requested_at_utc": "2026-07-22T21:00:02Z",
            "ingest_mode": "plan_only",
            "policy_version": self.policy["policy_version"],
            "probe_requested": True,
            "normalization_requested": False,
            "expected_source_sha256": sha,
            "expected_source_size_bytes": 10,
            "output_root": "/home/fdoblak/workspace/video_contract_checks/out",
            "fixture_mode": True,
        }
        validate_payload_against_schema(request, self.schemas["ingest_request.schema.json"])
        IngestRequest.from_dict(request)

    def test_required_field_rejection(self) -> None:
        with self.assertRaises(jsonschema.ValidationError):
            validate_payload_against_schema(
                {"schema_version": 1}, self.schemas["video_source.schema.json"]
            )

    def test_unknown_property_rejection(self) -> None:
        sha = "b" * 64
        payload = {
            "schema_version": 1,
            "source_id": "src_contract_two",
            "source_kind": "project_fixture",
            "original_filename": "a.mp4",
            "source_path": "/home/fdoblak/workspace/video_contract_checks/a.mp4",
            "source_size_bytes": 10,
            "source_sha256": sha,
            "media_type": "video/mp4",
            "container_hint": None,
            "created_at_utc": "2026-07-22T21:00:00Z",
            "registered_at_utc": "2026-07-22T21:00:01Z",
            "immutability_policy": "detect_mutation",
            "provenance": {
                "origin": "project_fixture",
                "label": "t",
                "notes": None,
            },
            "extra_field": True,
        }
        with self.assertRaises(jsonschema.ValidationError):
            validate_payload_against_schema(payload, self.schemas["video_source.schema.json"])

    def test_schema_version_rejection(self) -> None:
        payload = metadata_fixture("rotation_metadata", source_sha256="c" * 64)
        payload["schema_version"] = 99
        with self.assertRaises(jsonschema.ValidationError):
            validate_payload_against_schema(payload, self.schemas["video_probe.schema.json"])

    def test_probe_and_plan_and_receipt_cross_refs(self) -> None:
        sha = "d" * 64
        probe = metadata_fixture("vfr_metadata", source_sha256=sha)
        validate_payload_against_schema(probe, self.schemas["video_probe.schema.json"])
        plan = build_normalize_plan(
            plan_id="plan_contract_one",
            source_id="src_vfr",
            source_sha256=sha,
            policy_version=self.policy["policy_version"],
            required=True,
            reasons=("vfr_requires_explicit_decision",),
            target_container="mp4",
            target_video_codec="h264",
            target_audio_policy="drop",
            target_pixel_format="yuv420p",
            target_width=160,
            target_height=120,
            resize_policy="fit_within_keep_aspect",
            target_frame_rate=Rational(25, 1),
            frame_rate_policy="force_cfr",
            target_time_base=Rational(1, 90_000),
            rotation_policy="apply_metadata_once",
            sar_policy="preserve",
            audio_policy="drop",
            copy_metadata_policy="copy_timing_only",
            estimated_output_path="/home/fdoblak/workspace/video_contract_checks/out/x.mp4",
        )
        validate_payload_against_schema(plan.to_dict(), self.schemas["normalize_plan.schema.json"])
        # fingerprint stable
        self.assertEqual(plan.plan_fingerprint, plan.compute_fingerprint())

    def test_policy_defaults_safe(self) -> None:
        self.assertFalse(self.policy["network_sources_allowed"])
        self.assertFalse(self.policy["symlinks_allowed"])
        self.assertFalse(self.policy["overwrite_allowed"])
        self.assertEqual(self.policy["canonical_time_unit"], "microseconds")

    def test_schema_files_are_json(self) -> None:
        for name in SCHEMA_FILES:
            path = REPO / "schemas/video" / name
            json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
