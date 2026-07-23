"""Atomic Parquet I/O for contract tables."""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.data import DataContractError
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.fingerprint import contract_fingerprint, verify_schema_fingerprint
from football_analytics.data.types import (
    META_CONTRACT,
    META_CREATED_BY,
    META_FINGERPRINT,
    META_VERSION,
    ContractSpec,
)
from football_analytics.data.validation import validate_schema, validate_table


def _pa() -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def _reject_symlink_target(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise DataContractError("parquet path must not be a symlink")
    if path.parent.exists() and path.parent.is_symlink():
        raise DataContractError("parquet parent must not be a symlink")


def write_contract_parquet(
    table: Any,
    path: Path | str,
    contract: ContractSpec,
    *,
    contain_root: Path | None = None,
    overwrite: bool = False,
    compression: str = "zstd",
) -> Path:
    pa, pq = _pa()
    target = Path(path)
    _reject_symlink_target(target)
    if contain_root is not None:
        parent = target.parent
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            parent.resolve().relative_to(contain_root.resolve())
            (parent / target.name).resolve().relative_to(contain_root.resolve())
        except ValueError as exc:
            raise DataContractError("parquet path escapes containment") from exc
    if target.exists() and not overwrite:
        raise DataContractError(f"parquet target exists: {target}")
    if target.exists():
        mode = target.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise DataContractError("refusing non-regular parquet target")

    vr = validate_table(table, contract)
    if vr.status == "FAIL":
        raise DataContractError(f"table validation failed: {vr.errors[:3]}")

    schema = compile_arrow_schema(contract)
    # cast/replace schema metadata onto table
    casted = table.cast(schema)
    # strip pandas metadata if present
    md = dict(casted.schema.metadata or {})
    md.pop(b"pandas", None)
    casted = casted.replace_schema_metadata(md)

    parent = target.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".parquet.tmp", dir=str(parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(
            casted,
            tmp_path,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=min(max(casted.num_rows, 1), 65536),
        )
        if target.exists() and not overwrite:
            raise DataContractError(f"parquet target exists: {target}")
        os.replace(str(tmp_path), str(target))
        with contextlib.suppress(OSError):
            os.chmod(target, 0o600)
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise
    return target


def read_contract_parquet(
    path: Path | str,
    contract: ContractSpec,
    *,
    contain_root: Path | None = None,
) -> Any:
    pa, pq = _pa()
    target = Path(path)
    if target.is_symlink():
        raise DataContractError("parquet path must not be a symlink")
    if not target.is_file():
        raise DataContractError(f"parquet missing: {target}")
    if contain_root is not None:
        try:
            target.resolve().relative_to(contain_root.resolve())
        except ValueError as exc:
            raise DataContractError("parquet path escapes containment") from exc
    try:
        table = pq.read_table(target)
    except Exception as exc:  # noqa: BLE001
        raise DataContractError(f"corrupt/unreadable parquet: {type(exc).__name__}") from exc
    meta = table.schema.metadata or {}
    if meta.get(META_CONTRACT.encode()) != contract.contract_name.encode():
        raise DataContractError("parquet contract_name mismatch")
    if meta.get(META_VERSION.encode()) != str(contract.version).encode():
        raise DataContractError("parquet contract_version mismatch")
    fp = contract_fingerprint(contract)
    if not verify_schema_fingerprint(table.schema, fp):
        raise DataContractError("parquet schema_fingerprint mismatch/tamper")
    sr = validate_schema(table.schema, contract)
    if sr.status == "FAIL":
        raise DataContractError(f"parquet schema invalid: {sr.errors[:3]}")
    vr = validate_table(table, contract)
    if vr.status == "FAIL":
        raise DataContractError(f"parquet table invalid: {vr.errors[:3]}")
    return table


def inspect_contract_parquet(path: Path | str) -> dict[str, Any]:
    _, pq = _pa()
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise DataContractError("inspect requires regular non-symlink file")
    pf = pq.ParquetFile(target)
    meta = pf.schema_arrow.metadata or {}
    return {
        "path": str(target),
        "num_rows": pf.metadata.num_rows if pf.metadata else None,
        "contract_name": meta.get(META_CONTRACT.encode(), b"").decode(),
        "contract_version": meta.get(META_VERSION.encode(), b"").decode(),
        "schema_fingerprint": meta.get(META_FINGERPRINT.encode(), b"").decode(),
        "created_by_version": meta.get(META_CREATED_BY.encode(), b"").decode(),
        "file_sha256": sha256_file(target),
        "columns": [f.name for f in pf.schema_arrow],
    }


def write_contract_parquet_streaming(
    batches: Any,
    path: Path | str,
    contract: ContractSpec,
    *,
    contain_root: Path | None = None,
    overwrite: bool = False,
    compression: str = "zstd",
    check_semantics: bool = True,
) -> Path:
    """Stream Arrow RecordBatches/Tables into a contract Parquet file.

    Validates each batch against the contract, writes with ``pq.ParquetWriter``
    (zstd by default), then atomically renames temp → final. Does not materialize
    the full table as a Python list.
    """
    from collections.abc import Iterable, Iterator

    pa, pq = _pa()
    target = Path(path)
    _reject_symlink_target(target)
    if contain_root is not None:
        parent = target.parent
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            parent.resolve().relative_to(contain_root.resolve())
            (parent / target.name).resolve().relative_to(contain_root.resolve())
        except ValueError as exc:
            raise DataContractError("parquet path escapes containment") from exc
    if target.exists() and not overwrite:
        raise DataContractError(f"parquet target exists: {target}")
    if target.exists():
        mode = target.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise DataContractError("refusing non-regular parquet target")

    schema = compile_arrow_schema(contract)
    parent = target.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".parquet.tmp", dir=str(parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    writer: Any = None
    rows_written = 0
    try:
        if not isinstance(batches, (Iterable, Iterator)) or isinstance(batches, (str, bytes)):
            raise DataContractError("batches must be an iterator/iterable of RecordBatch/Table")
        for item in batches:
            if isinstance(item, pa.Table):
                table = item
            elif isinstance(item, pa.RecordBatch):
                table = pa.Table.from_batches([item])
            else:
                raise DataContractError(
                    f"batch must be RecordBatch or Table, got {type(item).__name__}"
                )
            vr = validate_table(table, contract, check_semantics=check_semantics)
            if vr.status == "FAIL":
                raise DataContractError(f"table validation failed: {vr.errors[:3]}")
            casted = table.cast(schema)
            md = dict(casted.schema.metadata or {})
            md.pop(b"pandas", None)
            casted = casted.replace_schema_metadata(md)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema=casted.schema,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            for batch in casted.to_batches():
                writer.write_batch(batch)
                rows_written += batch.num_rows
        if writer is None:
            # Empty stream: still publish a valid empty contract parquet.
            empty = pa.Table.from_batches([], schema=schema)
            pq.write_table(
                empty,
                tmp_path,
                compression=compression,
                use_dictionary=True,
                write_statistics=True,
                row_group_size=1,
            )
        else:
            writer.close()
            writer = None
        if target.exists() and not overwrite:
            raise DataContractError(f"parquet target exists: {target}")
        os.replace(str(tmp_path), str(target))
        with contextlib.suppress(OSError):
            os.chmod(target, 0o600)
    except Exception:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise
    _ = rows_written  # retained for clarity / future telemetry
    return target
