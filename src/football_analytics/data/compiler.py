"""Compile contract specs to PyArrow schemas (lazy pyarrow import)."""

from __future__ import annotations

from typing import Any

from football_analytics import __version__
from football_analytics.data import DataContractError
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.types import (
    META_CONTRACT,
    META_CREATED_BY,
    META_FINGERPRINT,
    META_VERSION,
    ContractSpec,
    FieldSpec,
)


def _pa() -> Any:
    import pyarrow as pa

    return pa


def _scalar_type(pa: Any, field: FieldSpec) -> Any:
    mapping = {
        "bool": pa.bool_(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "string": pa.string(),
        "large_string": pa.large_string(),
        "binary": pa.binary(),
    }
    if field.type_name in mapping:
        return mapping[field.type_name]
    if field.type_name == "fixed_size_binary":
        assert field.byte_width is not None
        return pa.binary(field.byte_width)
    if field.type_name == "timestamp":
        assert field.unit is not None
        return pa.timestamp(field.unit, tz=field.tz)
    if field.type_name == "duration":
        assert field.unit is not None
        return pa.duration(field.unit)
    raise DataContractError(f"cannot compile scalar type {field.type_name}")


def _compile_type(pa: Any, field: FieldSpec) -> Any:
    if field.type_name == "list":
        assert field.value_type is not None
        child = FieldSpec(name="element", type_name=field.value_type, nullable=False)
        return pa.list_(pa.field("element", _scalar_type(pa, child), nullable=False))
    if field.type_name == "fixed_size_list":
        assert field.value_type is not None and field.list_size is not None
        child = FieldSpec(name="element", type_name=field.value_type, nullable=False)
        return pa.list_(
            pa.field("element", _scalar_type(pa, child), nullable=False),
            field.list_size,
        )
    if field.type_name == "struct":
        children = [
            pa.field(c.name, _compile_type(pa, c), nullable=c.nullable) for c in field.fields
        ]
        return pa.struct(children)
    return _scalar_type(pa, field)


def compile_arrow_schema(spec: ContractSpec, *, attach_fingerprint: bool = True) -> Any:
    """Compile ContractSpec to pyarrow.Schema with FA metadata."""
    pa = _pa()
    fields = [pa.field(f.name, _compile_type(pa, f), nullable=f.nullable) for f in spec.fields]
    meta: dict[bytes, bytes] = {
        META_CONTRACT.encode("utf-8"): spec.contract_name.encode("utf-8"),
        META_VERSION.encode("utf-8"): str(spec.version).encode("utf-8"),
        META_CREATED_BY.encode("utf-8"): __version__.encode("utf-8"),
    }
    for k, v in sorted(spec.table_metadata.items()):
        meta[f"football_analytics.meta.{k}".encode()] = v.encode("utf-8")
    schema = pa.schema(fields, metadata=meta)
    if attach_fingerprint:
        # fingerprint from contract spec, not from schema-with-fingerprint (no loop)
        digest = contract_fingerprint(spec)
        meta2 = dict(schema.metadata or {})
        meta2[META_FINGERPRINT.encode("utf-8")] = digest.encode("utf-8")
        schema = schema.with_metadata(meta2)
    return schema


def get_contract(name: str, version: int | None = None, *, registry: Any = None) -> ContractSpec:
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    reg = registry or load_schema_registry(
        default_registry_path(), project_root=default_project_root()
    )
    return reg.load_contract(name, version)


def list_contracts(*, registry: Any = None) -> list[str]:
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    reg = registry or load_schema_registry(
        default_registry_path(), project_root=default_project_root()
    )
    return reg.list_contracts()
