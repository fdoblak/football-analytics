"""Stage 5D human role classification baseline unit tests."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import default_project_root
from football_analytics.perception.contracts import detection_schema_fingerprints
from football_analytics.perception.role_classification import AssignmentStatus
from football_analytics.perception.role_clustering import cluster_kit_colors
from football_analytics.perception.role_config import (
    default_human_role_config_path,
    human_role_config_fingerprint,
    load_human_role_config,
)
from football_analytics.perception.role_evaluation import (
    NOT_EVALUATED_ROLE,
    evaluate_role_assignments,
    evaluate_roles_from_rows,
)
from football_analytics.perception.role_features import (
    RoleFeatureError,
    color_l1_distance,
    extract_synthetic_features,
    map_user_other_to_staff,
)
from football_analytics.perception.role_fixtures import (
    FROZEN_ROLE_FIXTURES,
    assert_runtime_root,
    make_analysis_window_row,
    make_detection_row,
    make_frame_status_row,
    make_human_attribute_row,
)
from football_analytics.perception.role_service import (
    classify_from_synthetic_humans,
    run_human_role_classification,
)
from football_analytics.perception.taxonomy import map_model_class

REPO = default_project_root()
PY = sys.executable
EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"


def _write_table(rows: list, path: Path, contract_name: str, contain: Path) -> None:
    contract = get_contract(contract_name, 1)
    table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(contract))
    write_contract_parquet(table, path, contract, contain_root=contain)


class ConfigTests(unittest.TestCase):
    def test_config_fingerprint_stable(self) -> None:
        cfg = load_human_role_config(default_human_role_config_path())
        a = human_role_config_fingerprint(cfg)
        b = human_role_config_fingerprint(cfg)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertIn("staff", " ".join(cfg["notes"]).lower())
        self.assertIn("other", " ".join(cfg["notes"]).lower())


class MappingRulesTests(unittest.TestCase):
    def test_non_human_not_classified_as_player(self) -> None:
        tax = map_model_class(32, "sports_ball")
        self.assertEqual(tax.entity_type.value, "ball")
        self.assertNotEqual(tax.role_label.value, "player")

    def test_generic_person_never_auto_player(self) -> None:
        m = map_model_class(0, "person")
        self.assertEqual(m.entity_type.value, "human")
        self.assertEqual(m.role_label.value, "unknown")
        self.assertEqual(map_user_other_to_staff("other"), "staff")


class FeatureClusterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_human_role_config(default_human_role_config_path())

    def test_color_features_deterministic(self) -> None:
        a = extract_synthetic_features(
            detection_id=1,
            frame_index=0,
            bbox_xyxy=[10, 10, 40, 70],
            frame_width=200,
            frame_height=120,
            config=self.cfg,
            kit_hue=0.25,
            kit_saturation=0.6,
            kit_value=0.7,
        )
        b = extract_synthetic_features(
            detection_id=1,
            frame_index=0,
            bbox_xyxy=[10, 10, 40, 70],
            frame_width=200,
            frame_height=120,
            config=self.cfg,
            kit_hue=0.25,
            kit_saturation=0.6,
            kit_value=0.7,
        )
        self.assertEqual(a.color_signature, b.color_signature)
        self.assertEqual(a.upper_hist, b.upper_hist)
        self.assertTrue(0.0 <= a.crop_quality <= 1.0)
        self.assertTrue(math.isfinite(color_l1_distance(a.color_signature, b.color_signature)))

    def test_cluster_sort_deterministic_no_team_id(self) -> None:
        scenario = FROZEN_ROLE_FIXTURES["two_kits_players"]
        fw = 200.0
        fh = 120.0
        feats = []
        for h in scenario["humans"]:
            feats.append(
                extract_synthetic_features(
                    detection_id=int(h["detection_id"]),
                    frame_index=0,
                    bbox_xyxy=list(h["bbox"]),
                    frame_width=fw,
                    frame_height=fh,
                    config=self.cfg,
                    kit_hue=h["kit_hue"],
                    kit_saturation=h["kit_saturation"],
                    kit_value=h["kit_value"],
                )
            )
        c1, _ = cluster_kit_colors(feats, config=self.cfg)
        c2, _ = cluster_kit_colors(list(reversed(feats)), config=self.cfg)
        self.assertEqual([c.centroid for c in c1], [c.centroid for c in c2])
        self.assertLessEqual(len(c1), 2)
        for c in c1:
            self.assertIsNone(c.to_dict()["team_id"])
            self.assertIsNone(c.to_dict()["team_name"])


class AssignmentPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_human_role_config(default_human_role_config_path())
        self.fp = human_role_config_fingerprint(self.cfg)

    def test_tiny_crop_abstains(self) -> None:
        scenario = FROZEN_ROLE_FIXTURES["tiny_crop_abstain"]
        assigned = classify_from_synthetic_humans(
            run_id=generate_run_id(),
            video_id="video_01",
            frame_index=0,
            humans=scenario["humans"],
            frame_width=float(scenario["frame_width"]),
            frame_height=float(scenario["frame_height"]),
            config=self.cfg,
            config_fingerprint=self.fp,
        )
        self.assertEqual(assigned[0].assignment_status, AssignmentStatus.ABSTAINED)
        self.assertEqual(assigned[0].role_label.value, "unknown")

    def test_gk_color_alone_insufficient(self) -> None:
        scenario = FROZEN_ROLE_FIXTURES["gk_needs_extra_evidence"]
        assigned = classify_from_synthetic_humans(
            run_id=generate_run_id(),
            video_id="video_01",
            frame_index=0,
            humans=scenario["humans"],
            frame_width=float(scenario["frame_width"]),
            frame_height=float(scenario["frame_height"]),
            config=self.cfg,
            config_fingerprint=self.fp,
        )
        by_id = {a.detection_id: a for a in assigned}
        color_only = by_id[int(scenario["color_only_detection_id"])]
        self.assertNotEqual(color_only.role_label.value, "goalkeeper")
        gk = by_id[int(scenario["gk_candidate_detection_id"])]
        self.assertEqual(gk.role_label.value, "goalkeeper")

    def test_referee_not_from_dark_in_outfield_conflict(self) -> None:
        scenario = FROZEN_ROLE_FIXTURES["referee_dark"]
        assigned = classify_from_synthetic_humans(
            run_id=generate_run_id(),
            video_id="video_01",
            frame_index=0,
            humans=scenario["humans"],
            frame_width=float(scenario["frame_width"]),
            frame_height=float(scenario["frame_height"]),
            config=self.cfg,
            config_fingerprint=self.fp,
        )
        by_id = {a.detection_id: a for a in assigned}
        ref = by_id[int(scenario["ref_detection_id"])]
        self.assertEqual(ref.role_label.value, "referee")
        prov = json.loads(ref.to_attribute_row()["provenance_json"])
        self.assertIsNone(prov.get("team_id"))

    def test_conflict_unknown(self) -> None:
        scenario = FROZEN_ROLE_FIXTURES["conflict"]
        assigned = classify_from_synthetic_humans(
            run_id=generate_run_id(),
            video_id="video_01",
            frame_index=0,
            humans=scenario["humans"],
            frame_width=float(scenario["frame_width"]),
            frame_height=float(scenario["frame_height"]),
            config=self.cfg,
            config_fingerprint=self.fp,
        )
        # At least one abstain/unknown from conflict path or residual.
        self.assertTrue(any(a.role_label.value == "unknown" for a in assigned))

    def test_players_from_stable_clusters(self) -> None:
        scenario = FROZEN_ROLE_FIXTURES["two_kits_players"]
        assigned = classify_from_synthetic_humans(
            run_id=generate_run_id(),
            video_id="video_01",
            frame_index=0,
            humans=scenario["humans"],
            frame_width=float(scenario["frame_width"]),
            frame_height=float(scenario["frame_height"]),
            config=self.cfg,
            config_fingerprint=self.fp,
        )
        n_player = sum(1 for a in assigned if a.role_label.value == "player")
        self.assertGreaterEqual(n_player, int(scenario["expect_player_min"]))
        for a in assigned:
            prov = json.loads(a.to_attribute_row()["provenance_json"])
            self.assertIsNone(prov.get("team_id"))
            self.assertEqual(prov.get("other_maps_to"), "staff")
            self.assertIsNone(a.role_score)


class EvaluationTests(unittest.TestCase):
    def test_no_reviewed_gt_code(self) -> None:
        m = evaluate_roles_from_rows(
            [{"frame_index": 0, "detection_id": 0, "role_label": "player"}],
            None,
        )
        self.assertEqual(m.status, NOT_EVALUATED_ROLE)
        m2 = evaluate_roles_from_rows(
            [{"frame_index": 0, "detection_id": 0, "role_label": "player"}],
            [{"frame_index": 0, "detection_id": 0, "role_label": "player"}],
        )
        self.assertEqual(m2.status, NOT_EVALUATED_ROLE)

    def test_empty_denominator_null(self) -> None:
        m = evaluate_role_assignments(
            [],
            [
                {
                    "frame_index": 0,
                    "detection_id": 0,
                    "role_label": "player",
                    "is_reviewed_ground_truth": True,
                }
            ],
            synthetic_fixture_only=True,
        )
        self.assertIsNone(m.selective_accuracy)
        self.assertIsNone(m.per_role_precision["goalkeeper"])

    def test_synthetic_vs_real_claim_separated(self) -> None:
        m = evaluate_role_assignments(
            [{"frame_index": 0, "detection_id": 0, "role_label": "player"}],
            [
                {
                    "frame_index": 0,
                    "detection_id": 0,
                    "role_label": "player",
                    "is_reviewed_ground_truth": True,
                }
            ],
            synthetic_fixture_only=True,
        )
        d = m.to_dict()
        self.assertTrue(d["synthetic_fixture_only"])
        self.assertFalse(d["real_football_accuracy_claimed"])


class ServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_human_role_config(default_human_role_config_path())
        self.runtime = assert_runtime_root()

    def _write_bundle(self, tmp: Path, *, playability: str = "playable") -> dict[str, Path]:
        rid = generate_run_id()
        vid = "video_01"
        scenario = FROZEN_ROLE_FIXTURES["two_kits_players"]
        det_rows = [
            make_detection_row(
                rid, vid, frame_index=0, detection_id=int(h["detection_id"]), bbox=list(h["bbox"])
            )
            for h in scenario["humans"]
        ]
        # Add a ball detection that must be skipped.
        det_rows.append(
            make_detection_row(
                rid,
                vid,
                frame_index=0,
                detection_id=99,
                bbox=[1, 1, 5, 5],
                class_id=32,
                class_name="sports ball",
            )
        )
        attr_rows = [
            make_human_attribute_row(rid, vid, frame_index=0, detection_id=int(h["detection_id"]))
            for h in scenario["humans"]
        ]
        attr_rows.append(
            {
                "run_id": rid,
                "video_id": vid,
                "frame_index": 0,
                "detection_id": 99,
                "entity_type": "ball",
                "role_label": "unknown",
                "role_source": "detector_native",
                "role_score": None,
                "occlusion": None,
                "truncation": None,
                "visibility": None,
                "review_status": "unreviewed",
                "attribute_source_ref": "ball",
                "provenance_json": '{"stage":"5C"}',
                "contract_version": 1,
            }
        )
        status_rows = [
            make_frame_status_row(rid, vid, frame_index=0, human_count=len(scenario["humans"]))
        ]
        win_rows = [
            make_analysis_window_row(
                rid, vid, n_frames=1, playability=playability, tracking="eligible"
            )
        ]
        paths = {
            "detections": tmp / "detections.parquet",
            "attributes": tmp / "in_attributes.parquet",
            "status": tmp / "detection_frame_status.parquet",
            "windows": tmp / "analysis_windows.parquet",
            "out": tmp / "out",
        }
        paths["out"].mkdir(parents=True, exist_ok=True)
        _write_table(det_rows, paths["detections"], "detections", self.runtime)
        _write_table(attr_rows, paths["attributes"], "detection_attributes", self.runtime)
        _write_table(status_rows, paths["status"], "detection_frame_status", self.runtime)
        _write_table(win_rows, paths["windows"], "analysis_windows", self.runtime)
        return paths

    def test_non_human_skipped_and_fk_valid(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="role5d_", dir=str(self.runtime)))
        try:
            paths = self._write_bundle(tmp)
            result = run_human_role_classification(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["attributes"]),
                detection_frame_status=str(paths["status"]),
                analysis_windows=str(paths["windows"]),
                output_dir=str(paths["out"]),
                config=self.cfg,
                contain_root=self.runtime,
                allow_synthetic_without_video=True,
                synthetic_frame_size=(200.0, 120.0),
            )
            self.assertTrue(result.accepted, result.error_code)
            receipt = json.loads(Path(result.receipt_json).read_text(encoding="utf-8"))
            self.assertGreater(receipt["assignment_counts"]["skipped"], 0)
            self.assertEqual(
                sum(receipt["assignment_counts"].values()),
                receipt["assignment_counts"]["classified"]
                + receipt["assignment_counts"]["abstained"]
                + receipt["assignment_counts"]["not_eligible"]
                + receipt["assignment_counts"]["skipped"]
                + receipt["assignment_counts"]["failed"],
            )
            self.assertEqual(receipt["ground_truth_evaluation_status"], NOT_EVALUATED_ROLE)
            self.assertFalse(receipt["crops_persisted"])
            self.assertIsNone(receipt["team_id"])
            self.assertFalse((paths["out"] / "crops").exists())
            eval_body = json.loads(Path(result.evaluation_json).read_text(encoding="utf-8"))
            self.assertEqual(eval_body["status"], NOT_EVALUATED_ROLE)
            fp2 = human_role_config_fingerprint(self.cfg)
            self.assertEqual(result.config_fingerprint, fp2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_non_playable_not_eligible(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="role5d_np_", dir=str(self.runtime)))
        try:
            paths = self._write_bundle(tmp, playability="non_playable")
            result = run_human_role_classification(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["attributes"]),
                detection_frame_status=str(paths["status"]),
                analysis_windows=str(paths["windows"]),
                output_dir=str(paths["out"]),
                config=self.cfg,
                contain_root=self.runtime,
            )
            self.assertTrue(result.accepted)
            receipt = json.loads(Path(result.receipt_json).read_text(encoding="utf-8"))
            self.assertGreater(receipt["assignment_counts"]["not_eligible"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_overwrite_forbidden_no_false_success(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="role5d_ow_", dir=str(self.runtime)))
        try:
            paths = self._write_bundle(tmp)
            r1 = run_human_role_classification(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["attributes"]),
                detection_frame_status=str(paths["status"]),
                analysis_windows=str(paths["windows"]),
                output_dir=str(paths["out"]),
                config=self.cfg,
                contain_root=self.runtime,
                allow_synthetic_without_video=True,
                synthetic_frame_size=(200.0, 120.0),
            )
            self.assertTrue(r1.accepted, r1.error_code)
            r2 = run_human_role_classification(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["attributes"]),
                detection_frame_status=str(paths["status"]),
                analysis_windows=str(paths["windows"]),
                output_dir=str(paths["out"]),
                config=self.cfg,
                contain_root=self.runtime,
                allow_synthetic_without_video=True,
                synthetic_frame_size=(200.0, 120.0),
            )
            self.assertFalse(r2.accepted)
            self.assertEqual(r2.error_code, "OVERWRITE_FORBIDDEN")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_nan_inf_rejected(self) -> None:
        with self.assertRaises(RoleFeatureError):
            extract_synthetic_features(
                detection_id=0,
                frame_index=0,
                bbox_xyxy=[0, 0, 10, 20],
                frame_width=100,
                frame_height=100,
                config=self.cfg,
                kit_hue=float("nan"),
            )

    def test_detections_fingerprint_unchanged(self) -> None:
        fps = detection_schema_fingerprints()
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)

    def test_cli_help(self) -> None:
        import subprocess

        proc = subprocess.run(
            [PY, "-m", "football_analytics", "perception", "roles", "--help"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("classify", proc.stdout.lower() + proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
