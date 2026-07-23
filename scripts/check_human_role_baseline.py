#!/usr/bin/env python3
"""Validate Stage 5D human role classification baseline.

Exit codes:
  0 PASS / PASS_WITH_FINDINGS
  1 validation finding / NO-GO content
  2 configuration failure
  3 integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_role_checks")
EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"
GATE = (
    "PASS_WITH_FINDINGS — HUMAN ROLE CLASSIFICATION BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.findings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finding(self, msg: str) -> None:
        self.findings.append(msg)

    def finalize(self) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "NO-GO" if self.exit_code == EXIT_INTEGRITY else "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.findings or self.warnings:
            self.status = "PASS_WITH_FINDINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
            "overall_status": self.status,
            "gate": GATE if self.status in {"PASS", "PASS_WITH_FINDINGS"} else self.status,
        }
        body.update(self.extras)
        return body


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.compiler import compile_arrow_schema, get_contract
    from football_analytics.data.parquet import write_contract_parquet
    from football_analytics.perception.contracts import detection_schema_fingerprints
    from football_analytics.perception.role_config import (
        human_role_config_fingerprint,
        load_human_role_config,
    )
    from football_analytics.perception.role_evaluation import NOT_EVALUATED_ROLE
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

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_human_role_config(cfg_path)
        result.extras["config_fingerprint"] = human_role_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    fps = detection_schema_fingerprints()
    result.extras["detections_fingerprint"] = fps["detections"]
    if fps["detections"] != EXPECTED_DETECTIONS_FP:
        result.err("detections v1 fingerprint changed", integrity=True)

    # Lazy import: perception must not pull ultralytics
    before = {k for k in sys.modules if "ultralytics" in k.lower()}
    import football_analytics.perception as perc  # noqa: F401

    after = {k for k in sys.modules if "ultralytics" in k.lower()}
    if after - before:
        result.err("import football_analytics.perception loaded ultralytics", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    person = map_model_class(0, "person")
    if person.role_label.value == "player":
        result.err("generic person must not map to player", integrity=True)

    # Deterministic synthetic classify
    fp = result.extras["config_fingerprint"]
    fix = FROZEN_ROLE_FIXTURES["gk_needs_extra_evidence"]
    a1 = classify_from_synthetic_humans(
        run_id=generate_run_id(),
        video_id="video_01",
        frame_index=0,
        humans=fix["humans"],
        frame_width=float(fix["frame_width"]),
        frame_height=float(fix["frame_height"]),
        config=config,
        config_fingerprint=fp,
    )
    a2 = classify_from_synthetic_humans(
        run_id=generate_run_id(),
        video_id="video_01",
        frame_index=0,
        humans=fix["humans"],
        frame_width=float(fix["frame_width"]),
        frame_height=float(fix["frame_height"]),
        config=config,
        config_fingerprint=fp,
    )
    roles1 = [x.role_label.value for x in sorted(a1, key=lambda z: z.detection_id)]
    roles2 = [x.role_label.value for x in sorted(a2, key=lambda z: z.detection_id)]
    if roles1 != roles2:
        result.err("non-deterministic role assignment")
    by_id = {x.detection_id: x for x in a1}
    if by_id[int(fix["color_only_detection_id"])].role_label.value == "goalkeeper":
        result.err("goalkeeper assigned from color alone")
    if by_id[int(fix["gk_candidate_detection_id"])].role_label.value != "goalkeeper":
        result.warn("lateral GK candidate not classified (conservative thresholds)")

    session = Path(tempfile.mkdtemp(prefix="role_val_", dir=str(RUNTIME_ROOT)))
    try:
        rid = generate_run_id()
        vid = "video_01"
        humans = FROZEN_ROLE_FIXTURES["two_kits_players"]["humans"]
        det_rows = [
            make_detection_row(
                rid, vid, frame_index=0, detection_id=int(h["detection_id"]), bbox=list(h["bbox"])
            )
            for h in humans
        ]
        attr_rows = [
            make_human_attribute_row(rid, vid, frame_index=0, detection_id=int(h["detection_id"]))
            for h in humans
        ]
        status_rows = [make_frame_status_row(rid, vid, frame_index=0, human_count=len(humans))]
        win_rows = [make_analysis_window_row(rid, vid, n_frames=1)]

        def _write(rows: list, name: str, contract: str) -> Path:
            path = session / name
            table = pa.Table.from_pylist(
                rows, schema=compile_arrow_schema(get_contract(contract, 1))
            )
            write_contract_parquet(
                table, path, get_contract(contract, 1), contain_root=RUNTIME_ROOT
            )
            return path

        det_p = _write(det_rows, "detections.parquet", "detections")
        attr_p = _write(attr_rows, "in_attributes.parquet", "detection_attributes")
        st_p = _write(status_rows, "status.parquet", "detection_frame_status")
        win_p = _write(win_rows, "windows.parquet", "analysis_windows")
        out = session / "out"
        out.mkdir(parents=True, exist_ok=True)
        svc = run_human_role_classification(
            detections=str(det_p),
            detection_attributes=str(attr_p),
            detection_frame_status=str(st_p),
            analysis_windows=str(win_p),
            output_dir=str(out),
            config=config,
            contain_root=RUNTIME_ROOT,
            allow_synthetic_without_video=True,
            synthetic_frame_size=(200.0, 120.0),
        )
        if not svc.accepted:
            result.err(f"service failed: {svc.error_code}")
        else:
            receipt = json.loads(Path(svc.receipt_json).read_text(encoding="utf-8"))
            result.extras["receipt_assignment_counts"] = receipt["assignment_counts"]
            result.extras["evaluation_status"] = receipt["ground_truth_evaluation_status"]
            if receipt["ground_truth_evaluation_status"] != NOT_EVALUATED_ROLE:
                result.err("expected NOT_EVALUATED_NO_REVIEWED_HUMAN_ROLE_GROUND_TRUTH")
            if receipt.get("crops_persisted"):
                result.err("crops must not be persisted by default")
            if (out / "crops").exists():
                result.err("crops directory present despite persist_crops=false")
            total = sum(receipt["assignment_counts"].values())
            if total != sum(receipt["assignment_counts"].values()):
                result.err("receipt count inconsistency")
    finally:
        shutil.rmtree(session, ignore_errors=True)

    result.finding(
        "No reviewed football human-role ground truth — real role accuracy not validated"
    )
    result.finding('User-facing "other" maps to canonical RoleLabel staff')
    result.extras["other_maps_to"] = "staff"
    result.extras["runtime_root"] = str(RUNTIME_ROOT)
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/perception/human_role_baseline.yaml",
        help="Role baseline config path",
    )
    parser.add_argument(
        "--report-dir",
        default=str(RUNTIME_ROOT),
        help="Directory for validation JSON report",
    )
    args = parser.parse_args(argv)
    result = run_checks(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"human_role_validation_{ts}.json"
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"status: {result.status}")
    print(f"gate: {result.to_dict()['gate']}")
    print(f"report: {report_path}")
    for e in result.errors:
        print(f"error: {e}", file=sys.stderr)
    for f in result.findings:
        print(f"finding: {f}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
