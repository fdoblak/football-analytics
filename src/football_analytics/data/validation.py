"""Structural and semantic table validation."""

from __future__ import annotations

import json
import math
from typing import Any

from football_analytics.core.run_id import RunIdError, validate_run_id
from football_analytics.data import DataContractError
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.fingerprint import contract_fingerprint, verify_schema_fingerprint
from football_analytics.data.types import (
    META_CONTRACT,
    META_VERSION,
    SAFE_ID_RE,
    SHA256_RE,
    ContractSpec,
    ValidationResult,
    assert_safe_identifier,
)


def _pa() -> Any:
    import pyarrow as pa

    return pa


def _types_compatible(a: Any, b: Any) -> bool:
    """Compare Arrow types ignoring list child field naming (item vs element)."""
    pa = _pa()
    if a.equals(b):
        return True
    if pa.types.is_list(a) and pa.types.is_list(b):
        if (pa.types.is_fixed_size_list(a) or pa.types.is_fixed_size_list(b)) and not (
            pa.types.is_fixed_size_list(a)
            and pa.types.is_fixed_size_list(b)
            and a.list_size == b.list_size
        ):
            return False
        return _types_compatible(a.value_type, b.value_type)
    if pa.types.is_struct(a) and pa.types.is_struct(b):
        if len(a) != len(b):
            return False
        return all(
            a.field(i).name == b.field(i).name
            and a.field(i).nullable == b.field(i).nullable
            and _types_compatible(a.field(i).type, b.field(i).type)
            for i in range(len(a))
        )
    return False


def validate_schema(schema: Any, contract: ContractSpec) -> ValidationResult:
    result = ValidationResult(contract=contract.contract_name, version=contract.version)
    expected = compile_arrow_schema(contract)
    if len(schema) != len(expected):
        result.err("column count mismatch")
    else:
        for a, b in zip(schema, expected, strict=True):
            if a.name != b.name:
                result.err(f"column order/name mismatch: {a.name} != {b.name}")
            elif not _types_compatible(a.type, b.type):
                result.err(f"type mismatch for {a.name}: {a.type} != {b.type}")
            elif a.nullable != b.nullable:
                result.err(f"nullability mismatch for {a.name}")
    meta = schema.metadata or {}
    if meta.get(META_CONTRACT.encode()) != contract.contract_name.encode():
        result.err("metadata contract_name mismatch")
    if meta.get(META_VERSION.encode()) != str(contract.version).encode():
        result.err("metadata contract_version mismatch")
    fp = contract_fingerprint(contract)
    if not verify_schema_fingerprint(schema, fp):
        result.err("metadata schema_fingerprint mismatch")
    return result.finalize()


def _col(table: Any, name: str) -> Any:
    return table.column(name)


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _finite_float(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    return False


def _check_pk(table: Any, pk: tuple[str, ...], result: ValidationResult) -> None:
    if table.num_rows == 0:
        return
    seen: set[tuple[Any, ...]] = set()
    arrays = [_col(table, c).to_pylist() for c in pk]
    for i in range(table.num_rows):
        key = tuple(a[i] for a in arrays)
        if any(v is None for v in key):
            result.err(f"null primary key at row {i}")
            return
        if key in seen:
            result.err(f"duplicate primary key at row {i}")
            return
        seen.add(key)


def _apply_rule(table: Any, rule: dict[str, Any], result: ValidationResult) -> None:
    rid = rule.get("rule")
    if rid == "run_id_stage2b":
        for i, v in enumerate(_col(table, "run_id").to_pylist()):
            try:
                validate_run_id(v)
            except (RunIdError, TypeError, DataContractError):
                result.err(f"invalid run_id at row {i}")
                return
        return
    if rid == "safe_identifier":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            try:
                assert_safe_identifier(v)
            except DataContractError:
                result.err(f"unsafe identifier {field} at row {i}")
                return
        return
    if rid == "safe_identifier_or_null":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                continue
            try:
                assert_safe_identifier(v)
            except DataContractError:
                result.err(f"unsafe identifier {field} at row {i}")
                return
        return
    if rid == "sha256_hex64":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if not isinstance(v, str) or not SHA256_RE.fullmatch(v):
                result.err(f"invalid sha256 at row {i}")
                return
        return
    if rid == "nonempty_string":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if not isinstance(v, str) or not v:
                result.err(f"empty string {field} at row {i}")
                return
        return
    if rid == "no_path_like":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                continue
            if (
                not isinstance(v, str)
                or "/" in v
                or "\\" in v
                or ".." in v
                or v.startswith("~")
                or "$HOME" in v
                or not SAFE_ID_RE.fullmatch(v)
            ):
                result.err(f"path-like {field} at row {i}")
                return
        return
    if rid in {"int_gt", "int_ge", "float_gt", "float_ge"}:
        fields = rule.get("fields") or [rule["field"]]
        value = rule["value"]
        for f in fields:
            for i, v in enumerate(_col(table, f).to_pylist()):
                if v is None:
                    result.err(f"null {f} at row {i}")
                    return
                if not _finite_float(v):
                    result.err(f"non-finite {f} at row {i}")
                    return
                ok = {
                    "int_gt": v > value,
                    "int_ge": v >= value,
                    "float_gt": float(v) > value,
                    "float_ge": float(v) >= value,
                }[rid]
                if not ok:
                    result.err(f"{f} violates {rid} at row {i}")
                    return
        return
    if rid in {"int_ge_or_null", "int_gt_or_null", "float_ge_or_null"}:
        field = rule["field"]
        value = rule["value"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                continue
            if not _finite_float(v):
                result.err(f"non-finite {field} at row {i}")
                return
            ok = {
                "int_ge_or_null": v >= value,
                "int_gt_or_null": v > value,
                "float_ge_or_null": float(v) >= value,
            }[rid]
            if not ok:
                result.err(f"{field} violates {rid} at row {i}")
                return
        return
    if rid == "int_between_or_null":
        field = rule["field"]
        lo, hi = rule["min"], rule["max"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                continue
            if not (lo <= v <= hi):
                result.err(f"{field} out of range at row {i}")
                return
        return
    if rid == "int_le_pair":
        left, right = rule["left"], rule["right"]
        L = _col(table, left).to_pylist()
        R = _col(table, right).to_pylist()
        for i, (a, b) in enumerate(zip(L, R, strict=True)):
            if a > b:
                result.err(f"{left}>{right} at row {i}")
                return
        return
    if rid == "enum":
        field = rule["field"]
        values = set(rule["values"])
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v not in values:
                result.err(f"invalid enum {field} at row {i}")
                return
        return
    if rid == "float_unit_interval":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None or not _finite_float(v) or not (0.0 <= float(v) <= 1.0):
                result.err(f"confidence {field} out of range at row {i}")
                return
        return
    if rid == "float_unit_interval_or_null":
        fields = rule.get("fields") or [rule["field"]]
        for f in fields:
            for i, v in enumerate(_col(table, f).to_pylist()):
                if v is None:
                    continue
                if not _finite_float(v) or not (0.0 <= float(v) <= 1.0):
                    result.err(f"confidence {f} out of range at row {i}")
                    return
        return
    if rid == "bbox_xyxy_half_open":
        cols = {c: _col(table, c).to_pylist() for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")}
        for i in range(table.num_rows):
            x1, y1, x2, y2 = (cols[c][i] for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
            vals = (x1, y1, x2, y2)
            if any(not _finite_float(v) or v is None for v in vals):
                result.err(f"non-finite bbox at row {i}")
                return
            if not (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1):
                result.err(f"invalid bbox xyxy at row {i}")
                return
        return
    if rid == "list_no_null_items":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                result.err(f"null list {field} at row {i}")
                return
            if any(item is None for item in v):
                result.err(f"null list item {field} at row {i}")
                return
        return
    if rid == "monotonic_video_time":
        # per (run_id, video_id) ordered by frame_index
        rows = table.to_pylist()
        groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for r in rows:
            key = (r["run_id"], r["video_id"])
            groups.setdefault(key, []).append((r["frame_index"], r["video_time_us"]))
        for key, items in groups.items():
            items.sort(key=lambda x: x[0])
            prev_t = -1
            for _fi, t in items:
                if t < prev_t:
                    result.err(f"non-monotonic video_time for {key}")
                    return
                prev_t = t
        return
    if rid == "track_summary_counts":
        for i, r in enumerate(table.to_pylist()):
            if r["observed_count"] + r["predicted_count"] > r["observation_count"]:
                result.err(f"track summary count inconsistency at row {i}")
                return
            if r["observation_count"] < 1:
                result.err(f"observation_count < 1 at row {i}")
                return
        return
    if rid == "calibration_valid_homography":
        for i, r in enumerate(table.to_pylist()):
            if r["is_valid"] and r["homography_image_to_pitch"] is None:
                result.err(f"valid calibration missing homography at row {i}")
                return
        return
    if rid == "fixed_list_finite_or_null":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                continue
            if any(not _finite_float(x) or x is None for x in v):
                result.err(f"non-finite fixed list {field} at row {i}")
                return
        return
    if rid == "jersey_number_policy":
        for i, r in enumerate(table.to_pylist()):
            num = r["normalized_number"]
            digits = r["digit_count"]
            if num is None:
                continue
            if not (0 <= int(num) <= 99):
                result.err(f"jersey number out of range at row {i}")
                return
            expected_digits = 1 if num < 10 else 2
            if num == 0:
                expected_digits = 1
            if digits is not None and digits not in (0, 1, 2):
                result.err(f"digit_count invalid at row {i}")
                return
            if digits is not None and digits != expected_digits:
                result.err(f"digit_count inconsistent at row {i}")
                return
        return
    if rid == "time_interval_or_null":
        for i, r in enumerate(table.to_pylist()):
            s, e = r["start_time_us"], r["end_time_us"]
            if s is None and e is None:
                continue
            if s is None or e is None:
                result.err(f"partial time interval at row {i}")
                return
            if s > e or s < 0 or e < 0:
                result.err(f"invalid time interval at row {i}")
                return
        return
    if rid == "actor_track_ids":
        for i, v in enumerate(_col(table, "actor_track_ids").to_pylist()):
            if v is None:
                result.err(f"null actor_track_ids at row {i}")
                return
            if any(x is None or x < 0 for x in v):
                result.err(f"invalid actor id at row {i}")
                return
            if len(v) != len(set(v)):
                result.err(f"duplicate actor ids at row {i}")
                return
        return
    if rid == "canonical_json_object_or_null":
        field = rule["field"]
        for i, v in enumerate(_col(table, field).to_pylist()):
            if v is None:
                continue
            try:
                obj = json.loads(v)
            except Exception:
                result.err(f"invalid attributes_json at row {i}")
                return
            if not isinstance(obj, dict):
                result.err(f"attributes_json not object at row {i}")
                return
        return
    result.err(f"unknown semantic rule: {rid}")


def _reject_nan_inf(table: Any, result: ValidationResult) -> None:
    pa = _pa()
    for name in table.column_names:
        col = table.column(name)
        t = col.type
        if pa.types.is_floating(t):
            for i, v in enumerate(col.to_pylist()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    result.err(f"NaN/Infinity in {name} at row {i}")
                    return
        if pa.types.is_list(t) and pa.types.is_floating(t.value_type):
            for i, v in enumerate(col.to_pylist()):
                if v is None:
                    continue
                if any(isinstance(x, float) and (math.isnan(x) or math.isinf(x)) for x in v):
                    result.err(f"NaN/Infinity in list {name} at row {i}")
                    return


def validate_table(
    table: Any, contract: ContractSpec, *, check_semantics: bool = True
) -> ValidationResult:
    """Strict structural (+ optional semantic) validation. Does not mutate table."""
    result = ValidationResult(
        contract=contract.contract_name, version=contract.version, row_count=table.num_rows
    )
    expected = compile_arrow_schema(contract)
    # structural schema (ignore fingerprint metadata differences on bare tables)
    if table.column_names != [f.name for f in expected]:
        # missing/extra/order
        if set(table.column_names) != {f.name for f in expected}:
            missing = sorted(set(f.name for f in expected) - set(table.column_names))
            extra = sorted(set(table.column_names) - set(f.name for f in expected))
            if missing:
                result.err(f"missing columns: {missing}")
            if extra:
                result.err(f"extra columns: {extra}")
        else:
            result.err("column order mismatch")
        return result.finalize()
    for name, field in zip(table.column_names, expected, strict=True):
        col = table.column(name)
        if not _types_compatible(col.type, field.type):
            result.err(f"type mismatch for {name}: {col.type} != {field.type}")
        # nullability: if field non-nullable, column must not contain nulls
        if not field.nullable and col.null_count > 0:
            result.err(f"nulls in non-nullable column {name}")
    if result.errors:
        return result.finalize()
    _reject_nan_inf(table, result)
    _check_pk(table, contract.primary_key, result)
    if check_semantics and not result.errors:
        for rule in contract.semantic_rules:
            _apply_rule(table, rule, result)
            if len(result.errors) >= 50:
                break
    return result.finalize()
