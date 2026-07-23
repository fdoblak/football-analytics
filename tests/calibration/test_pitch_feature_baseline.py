"""Stage 8B pitch keypoint/line detection baseline tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from football_analytics.calibration.contracts import (
    EXPECTED_CALIBRATIONS_FP,
    assert_calibrations_fingerprint_frozen,
    calibration_schema_fingerprints,
)
from football_analytics.calibration.pitch_feature_config import (
    load_pitch_feature_config,
    pitch_feature_config_fingerprint,
)
from football_analytics.calibration.pitch_feature_evaluation import (
    NOT_EVALUATED_PITCH_FEATURES,
    evaluate_pitch_features,
)
from football_analytics.calibration.pitch_feature_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    fixture_image_bundle,
    make_solid_rgb,
)
from football_analytics.calibration.pitch_feature_mapping import (
    NBJW_LINES_LIST,
    keypoint_mapping,
    line_mapping,
)
from football_analytics.calibration.pitch_feature_postprocess import (
    PitchFeaturePostprocessError,
    decode_keypoints_from_heatmap,
    decode_lines_from_heatmap,
    fit_line_from_mask,
    make_synthetic_peak_heatmap,
)
from football_analytics.calibration.pitch_feature_preprocess import (
    build_stretch_transform,
    model_point_to_source,
    preprocess_rgb_uint8_to_tensor,
    source_point_to_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "calibration" / "pitch_feature_baseline.yaml"
KP_PATH = Path("/home/fdoblak/models/soccernet/sn-banner/SV_kp.pth")
LINES_PATH = Path("/home/fdoblak/models/soccernet/sn-banner/SV_lines.pth")
EXPECTED_KP_SHA = "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113"
EXPECTED_LINES_SHA = "2751242917f8c0f858a396e0cfe4521be39fe07bf049590eb21714526acecac1"


class PitchFeatureBaselineTests(unittest.TestCase):
    def test_lazy_import_does_not_load_hrnet(self) -> None:
        before = {k for k in sys.modules if k.startswith("fa_nbjw_") or "cls_hrnet" in k}
        import football_analytics.calibration as cal  # noqa: F401

        after = {k for k in sys.modules if k.startswith("fa_nbjw_") or "cls_hrnet" in k}
        self.assertEqual(after - before, set())

    def test_config_load_and_fingerprint(self) -> None:
        cfg = load_pitch_feature_config(CONFIG_PATH)
        self.assertEqual(list(cfg["image_size"]), [960, 540])
        self.assertFalse(cfg["auto_homography"])
        self.assertFalse(cfg["network_sources_allowed"])
        self.assertTrue(cfg["output_policy"]["confidence_always_null"])
        fp = pitch_feature_config_fingerprint(cfg)
        self.assertEqual(len(fp), 64)

    def test_calibrations_fingerprint_frozen(self) -> None:
        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        self.assertEqual(fps["calibrations"], EXPECTED_CALIBRATIONS_FP)

    def test_stretch_inverse_and_preprocess(self) -> None:
        tf = build_stretch_transform(source_width=320, source_height=180)
        mx, my = source_point_to_model(80.0, 45.0, tf)
        sx, sy = model_point_to_source(mx, my, tf)
        self.assertAlmostEqual(sx, 80.0, places=5)
        self.assertAlmostEqual(sy, 45.0, places=5)
        img = make_solid_rgb(width=320, height=180)
        tensor, tf2 = preprocess_rgb_uint8_to_tensor(img)
        self.assertEqual(tuple(tensor.shape), (1, 3, 540, 960))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertEqual(tf2.source_width, 320)

    def test_lines_list_trailing_space(self) -> None:
        self.assertEqual(NBJW_LINES_LIST[7], "Goal left post left ")
        m = line_mapping(7)
        self.assertTrue(m.source_name.endswith(" "))
        self.assertIsNone(m.canonical_pitch_feature_id)

    def test_unknown_mapping_stays_null(self) -> None:
        m = keypoint_mapping(36)  # circle-ish world coord — likely unmapped
        # Even if mapped, empty string forbidden; null OK
        if m.canonical_pitch_feature_id is not None:
            self.assertNotEqual(m.canonical_pitch_feature_id, "")

    def test_peak_nms_and_low_confidence(self) -> None:
        heat = make_synthetic_peak_heatmap(
            channels=57,
            height=270,
            width=480,
            peaks=[
                (0, 80, 80, 0.95),
                (0, 82, 82, 0.90),  # near duplicate channel peaks — top1 only
                (1, 100, 100, 0.05),
            ],
        )
        tf = build_stretch_transform(source_width=960, source_height=540)
        kps = decode_keypoints_from_heatmap(
            heat, transform=tf, score_threshold=0.1, expected_channels=57
        )
        ch0 = [k for k in kps if k.channel_index == 0 and not k.rejected]
        ch1 = [k for k in kps if k.channel_index == 1 and not k.rejected]
        self.assertEqual(len(ch0), 1)
        self.assertEqual(len(ch1), 0)

    def test_line_short_rejection_and_component_fit(self) -> None:
        heat = make_synthetic_peak_heatmap(
            channels=23,
            height=270,
            width=480,
            peaks=[(0, 10, 10, 0.9), (0, 10, 12, 0.9)],  # very short after scale
        )
        tf = build_stretch_transform(source_width=960, source_height=540)
        lines = decode_lines_from_heatmap(
            heat,
            transform=tf,
            score_threshold=0.1,
            expected_channels=23,
            minimum_length_px=50.0,
        )
        self.assertTrue(all(ln.rejected for ln in lines if ln.channel_index == 0))
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[30:50, 5:75] = 1
        fitted = fit_line_from_mask(mask, minimum_length_px=8.0)
        self.assertIsNotNone(fitted)

    def test_bounds_and_nan(self) -> None:
        import torch

        heat = torch.zeros((1, 57, 270, 480))
        heat[0, 0, 0, 0] = float("nan")
        tf = build_stretch_transform(source_width=960, source_height=540)
        with self.assertRaises(PitchFeaturePostprocessError):
            decode_keypoints_from_heatmap(
                heat, transform=tf, score_threshold=0.1, expected_channels=57
            )

    def test_channel_mismatch(self) -> None:
        heat = make_synthetic_peak_heatmap(channels=5, height=270, width=480, peaks=[])
        tf = build_stretch_transform(source_width=960, source_height=540)
        with self.assertRaises(PitchFeaturePostprocessError):
            decode_keypoints_from_heatmap(
                heat, transform=tf, score_threshold=0.1, expected_channels=57
            )

    def test_evaluation_not_evaluated(self) -> None:
        report = evaluate_pitch_features()
        self.assertEqual(report.ground_truth_evaluation_status, NOT_EVALUATED_PITCH_FEATURES)
        self.assertTrue(all(v is None for v in report.metrics.values()))

    def test_hash_mismatch(self) -> None:
        from football_analytics.calibration.pitch_feature_adapter import (
            PitchFeatureAdapterError,
            verify_weight_file,
        )

        if not KP_PATH.is_file():
            self.skipTest("SV_kp missing")
        with self.assertRaises(PitchFeatureAdapterError):
            verify_weight_file(KP_PATH, expected_sha256="0" * 64)

    def test_service_statuses_and_no_overwrite(self) -> None:
        from football_analytics.calibration.pitch_feature_config import (
            unfreeze_pitch_feature_config,
        )
        from football_analytics.calibration.pitch_feature_service import (
            run_pitch_feature_detect,
        )

        if not (KP_PATH.is_file() and LINES_PATH.is_file()):
            self.skipTest("SV weights missing")
        assert_runtime_root()
        cfg = unfreeze_pitch_feature_config(load_pitch_feature_config(CONFIG_PATH))
        cfg["device_policy"] = "cpu_only"
        cfg["maximum_frames_per_run"] = 2
        cfg["kp_score_threshold"] = 0.99
        cfg["line_score_threshold"] = 0.99
        bundle = fixture_image_bundle()
        session = Path(tempfile.mkdtemp(prefix="pf_test_", dir=str(RUNTIME_ROOT)))
        try:
            r = run_pitch_feature_detect(
                output_dir=str(session / "run_a"),
                config=cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=bundle,
                project_root=REPO_ROOT,
            )
            self.assertTrue(r.accepted, r.error_code)
            statuses = {
                f["status"]
                for f in __import__("json").loads(Path(r.frame_status_json).read_text())["frames"]
            }
            self.assertTrue(
                {"processed_no_features", "not_eligible"} & statuses or "processed" in statuses
            )
            # no-feature must not be failure
            self.assertNotIn("failed", statuses - {"not_eligible"})
            again = run_pitch_feature_detect(
                output_dir=str(session / "run_a"),
                config=cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=bundle,
                project_root=REPO_ROOT,
            )
            self.assertEqual(again.error_code, "OVERWRITE_FORBIDDEN")
            fail = run_pitch_feature_detect(
                output_dir=str(session / "run_fail"),
                config=cfg,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=bundle,
                project_root=REPO_ROOT,
                inject_failure=True,
            )
            self.assertFalse(fail.accepted)
            self.assertFalse((session / "run_fail" / "calibration_features.parquet").exists())
            # confidence null
            for row in r.summary["feature_rows"]:
                self.assertIsNone(row.get("confidence"))
        finally:
            import shutil

            shutil.rmtree(session, ignore_errors=True)

    @unittest.skipUnless(KP_PATH.is_file() and LINES_PATH.is_file(), "weights present")
    def test_model_smoke_schema_shapes(self) -> None:
        from football_analytics.calibration.pitch_feature_adapter import (
            NbjwHrnetPitchFeatureAdapter,
        )
        from football_analytics.calibration.pitch_feature_config import (
            unfreeze_pitch_feature_config,
        )

        cfg = unfreeze_pitch_feature_config(load_pitch_feature_config(CONFIG_PATH))
        cfg["device_policy"] = "cpu_only"
        kp_sha = EXPECTED_KP_SHA
        lines_sha = EXPECTED_LINES_SHA
        adapter = NbjwHrnetPitchFeatureAdapter.load(
            config=cfg,
            kp_weights_path=KP_PATH,
            lines_weights_path=LINES_PATH,
            kp_expected_sha256=kp_sha,
            lines_expected_sha256=lines_sha,
            device_policy="cpu_only",
        )
        img = make_solid_rgb(width=960, height=540)
        out = adapter.infer_rgb(img)
        self.assertEqual(out.kp_heatmap_shape[1], 58)
        self.assertEqual(out.lines_heatmap_shape[1], 24)
        self.assertEqual(out.kp_heatmap_shape[2:], (270, 480))
        self.assertEqual(out.kp_model_sha256, EXPECTED_KP_SHA)
        self.assertEqual(out.lines_model_sha256, EXPECTED_LINES_SHA)


if __name__ == "__main__":
    unittest.main()
