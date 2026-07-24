#!/usr/bin/env python3
"""Stage 9E physical metric pipeline fusion tests."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from football_analytics.physical.pipeline_config import (
    load_pipeline_config,
    pipeline_config_fingerprint,
)
from football_analytics.physical.pipeline_evaluation import NOT_EVALUATED_PIPELINE
from football_analytics.physical.pipeline_fixtures import run_consistent_chain
from football_analytics.physical.pipeline_service import integrate_physical_metrics


class PhysicalMetricPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_pipeline_config()
        self.fp = pipeline_config_fingerprint(self.cfg)
        self.tmp = Path(tempfile.mkdtemp(prefix="pipe9e_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_config(self) -> None:
        self.assertEqual(self.cfg["stage"], "9E")
        self.assertTrue(self.cfg["require_confirmed_identity"])
        self.assertFalse(self.cfg["output_policy"]["write_final_customer_visual"])

    def test_02_fuse_consistent(self) -> None:
        chain = run_consistent_chain(self.tmp / "chain")
        motion_sum = json.loads(Path(str(chain["motion"].summary_json)).read_text())
        spatial_sum = json.loads(Path(str(chain["spatial"].summary_json)).read_text())
        res = integrate_physical_metrics(
            output_dir=self.tmp / "fuse",
            identity=chain["identity"],
            motion_summary=motion_sum,
            motion_receipt=json.loads(Path(str(chain["motion"].receipt_json)).read_text()),
            spatial_summary=spatial_sum,
            spatial_receipt=json.loads(Path(str(chain["spatial"].receipt_json)).read_text()),
            recounted_distance_m=motion_sum.get("measured_distance_m"),
            config=self.cfg,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.summary["evaluation_status"], NOT_EVALUATED_PIPELINE)
        self.assertFalse(res.summary["final_customer_visual_created"])
        self.assertTrue(Path(str(res.receipt_json)).is_file())
        self.assertTrue(Path(str(res.quality_json)).is_file())

    def test_03_revoked_not_evaluable(self) -> None:
        chain = run_consistent_chain(self.tmp / "chain2")
        motion_sum = json.loads(Path(str(chain["motion"].summary_json)).read_text())
        ident = dict(chain["identity"])
        ident["identity_status"] = "revoked"
        ident["assignment_revoked"] = True
        res = integrate_physical_metrics(
            output_dir=self.tmp / "rev",
            identity=ident,
            motion_summary=motion_sum,
            motion_receipt=json.loads(Path(str(chain["motion"].receipt_json)).read_text()),
            config=self.cfg,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.summary["overall_physical_analysis_status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
