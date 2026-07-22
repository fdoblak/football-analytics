"""Explicit schema migration framework."""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics import __version__
from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import RecordError, write_json_record
from football_analytics.data import DataContractError
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.registry import SchemaRegistry
from football_analytics.data.types import ContractSpec
from football_analytics.data.validation import validate_table

MigrationFn = Callable[[Any, ContractSpec, ContractSpec], Any]

_REGISTRY: dict[tuple[str, int, int], MigrationFn] = {}


def register_migration(contract: str, frm: int, to: int, fn: MigrationFn) -> None:
    key = (contract, frm, to)
    if key in _REGISTRY:
        raise DataContractError(f"migration already registered: {key}")
    _REGISTRY[key] = fn


def plan_migration(
    registry: SchemaRegistry, contract: str, frm: int, to: int
) -> list[tuple[int, int, str]]:
    if frm == to:
        return []
    if to < frm:
        raise DataContractError("downgrade migrations are rejected by default")
    entry = registry.get_entry(contract)
    graph: dict[int, list[tuple[int, str]]] = {}
    for e in entry.edges:
        graph.setdefault(e.frm, []).append((e.to, e.migration_id))
    # BFS unique path
    q: deque[tuple[int, list[tuple[int, int, str]]]] = deque([(frm, [])])
    visited = {frm}
    found: list[tuple[int, int, str]] | None = None
    while q:
        node, path = q.popleft()
        for nxt, mid in graph.get(node, []):
            if nxt in visited:
                continue
            new_path = path + [(node, nxt, mid)]
            if nxt == to:
                if found is not None:
                    raise DataContractError("ambiguous migration path")
                found = new_path
            visited.add(nxt)
            q.append((nxt, new_path))
    if found is None:
        raise DataContractError(f"no migration path {contract} {frm}->{to}")
    return found


def _migrate_detections_0_to_1(table: Any, src: ContractSpec, dst: ContractSpec) -> Any:
    pa = __import__("pyarrow")
    from football_analytics.data.compiler import compile_arrow_schema

    rows = table.to_pylist()
    out = []
    class_map = {"ball": 0, "player": 1, "referee": 2, "goalkeeper": 3}
    for r in rows:
        w, h = r["bbox_width"], r["bbox_height"]
        if w is None or h is None or w <= 0 or h <= 0:
            raise DataContractError("invalid bbox width/height in v0->v1")
        for k in ("bbox_x", "bbox_y", "bbox_width", "bbox_height", "confidence"):
            v = r[k]
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                raise DataContractError("NaN/Infinity in v0 detection")
        cname = r["class_name"]
        out.append(
            {
                "run_id": r["run_id"],
                "video_id": r["video_id"],
                "frame_index": r["frame_index"],
                "detection_id": r["detection_id"],
                "class_id": int(class_map.get(cname, -1)),
                "class_name": cname,
                "confidence": r["confidence"],
                "bbox_x1": r["bbox_x"],
                "bbox_y1": r["bbox_y"],
                "bbox_x2": r["bbox_x"] + w,
                "bbox_y2": r["bbox_y"] + h,
                "model_id": r["model_id"],
                "is_interpolated": False,
                "quality_flags": [],
            }
        )
    schema = compile_arrow_schema(dst)
    return pa.Table.from_pylist(out, schema=schema)


# register built-in
register_migration("detections", 0, 1, _migrate_detections_0_to_1)


def migrate_table(
    table: Any,
    *,
    registry: SchemaRegistry,
    contract: str,
    from_version: int,
    to_version: int,
) -> Any:
    if from_version == to_version:
        return table
    steps = plan_migration(registry, contract, from_version, to_version)
    current = table
    cur_ver = from_version
    for frm, to, mid in steps:
        key = (contract, frm, to)
        if key not in _REGISTRY:
            raise DataContractError(f"unsupported migration step {mid}")
        src = registry.load_contract(contract, frm)
        dst = registry.load_contract(contract, to)
        vr = validate_table(current, src)
        if vr.status == "FAIL":
            raise DataContractError(f"source validation failed before {mid}: {vr.errors[:3]}")
        current = _REGISTRY[key](current, src, dst)
        vr2 = validate_table(current, dst)
        if vr2.status == "FAIL":
            raise DataContractError(f"target validation failed after {mid}: {vr2.errors[:3]}")
        cur_ver = to
    if cur_ver != to_version:
        raise DataContractError("migration did not reach target version")
    return current


def migrate_parquet(
    source: Path,
    destination: Path,
    *,
    registry: SchemaRegistry,
    contract: str,
    from_version: int,
    to_version: int,
    receipt_path: Path,
    contain_root: Path | None = None,
) -> dict[str, Any]:
    src = Path(source)
    dst = Path(destination)
    receipt = Path(receipt_path)
    if dst.exists():
        raise DataContractError("destination exists (no overwrite)")
    if receipt.exists():
        raise DataContractError("migration receipt already exists (no overwrite)")
    if src.is_symlink() or dst.is_symlink():
        raise DataContractError("symlink source/destination rejected")
    src_spec = registry.load_contract(contract, from_version)
    dst_spec = registry.load_contract(contract, to_version)
    source_hash_before = sha256_file(src)
    table = read_contract_parquet(src, src_spec, contain_root=contain_root)
    src_rows = table.num_rows
    src_fp = contract_fingerprint(src_spec)
    steps = plan_migration(registry, contract, from_version, to_version)
    mid = steps[0][2] if steps else f"{contract}_{from_version}_noop_{to_version}"
    try:
        migrated = migrate_table(
            table,
            registry=registry,
            contract=contract,
            from_version=from_version,
            to_version=to_version,
        )
        # PK / order preservation check for detections 0->1
        if contract == "detections" and from_version == 0 and to_version == 1:
            before = [
                (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"])
                for r in table.to_pylist()
            ]
            after = [
                (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"])
                for r in migrated.to_pylist()
            ]
            if before != after:
                raise DataContractError("primary key/order not preserved")
        write_contract_parquet(migrated, dst, dst_spec, contain_root=contain_root, overwrite=False)
        source_hash_after = sha256_file(src)
        if source_hash_after != source_hash_before:
            raise DataContractError("source file changed during migration")
        dest_hash = sha256_file(dst)
        receipt_payload = {
            "schema_version": 1,
            "migration_id": mid if len(steps) == 1 else "+".join(s[2] for s in steps),
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "contract_name": contract,
            "from_version": from_version,
            "to_version": to_version,
            "source_path": src.name,
            "destination_path": dst.name,
            "source_file_sha256": source_hash_before,
            "destination_file_sha256": dest_hash,
            "source_schema_fingerprint": src_fp,
            "destination_schema_fingerprint": contract_fingerprint(dst_spec),
            "source_row_count": src_rows,
            "destination_row_count": migrated.num_rows,
            "primary_key_preserved": True,
            "row_order_preserved": True,
            "lossy": False,
            "steps": [s[2] for s in steps] or ["noop"],
            "status": "success",
            "tool_version": __version__,
            "error": None,
        }
        try:
            write_json_record(receipt, receipt_payload, contain_root=contain_root, overwrite=False)
        except RecordError as exc:
            raise DataContractError(str(exc)) from exc
        return receipt_payload
    except Exception:
        if dst.exists():
            with contextlib.suppress(OSError):
                dst.unlink()
        raise
