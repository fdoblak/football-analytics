"""Load and parse contract JSON specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.data import DataContractError
from football_analytics.data.types import (
    ALLOWED_NESTED_TYPES,
    ALLOWED_SCALAR_TYPES,
    FIELD_NAME_RE,
    MAX_FIXED_LIST_SIZE,
    MAX_NEST_DEPTH,
    MAX_SPEC_BYTES,
    TIMESTAMP_UNITS,
    ContractSpec,
    FieldSpec,
    ForeignKeySpec,
)

ALLOWED_TOP_KEYS = frozenset(
    {
        "contract_name",
        "version",
        "description",
        "fields",
        "primary_key",
        "foreign_keys",
        "partition_by",
        "sort_by",
        "semantic_rules",
        "table_metadata",
    }
)


def _parse_field(raw: dict[str, Any], *, depth: int) -> FieldSpec:
    if depth > MAX_NEST_DEPTH:
        raise DataContractError("field nesting depth exceeded")
    if not isinstance(raw, dict):
        raise DataContractError("field must be object")
    name = raw.get("name")
    typ = raw.get("type")
    nullable = raw.get("nullable")
    if not isinstance(name, str) or not FIELD_NAME_RE.fullmatch(name):
        raise DataContractError(f"invalid field name: {name!r}")
    if not isinstance(typ, str):
        raise DataContractError(f"invalid type for {name}")
    if not isinstance(nullable, bool):
        raise DataContractError(f"nullable must be bool for {name}")
    if typ in ALLOWED_SCALAR_TYPES:
        unit = raw.get("unit")
        tz = raw.get("tz")
        byte_width = raw.get("byte_width")
        if typ == "timestamp":
            if unit not in TIMESTAMP_UNITS:
                raise DataContractError(f"invalid timestamp unit for {name}")
            if tz not in (None, "UTC"):
                raise DataContractError(f"invalid timestamp tz for {name}")
        if typ == "duration" and unit not in TIMESTAMP_UNITS:
            raise DataContractError(f"invalid duration unit for {name}")
        if typ == "fixed_size_binary" and (
            not isinstance(byte_width, int) or byte_width <= 0 or byte_width > 4096
        ):
            raise DataContractError(f"invalid byte_width for {name}")
        return FieldSpec(
            name=name,
            type_name=typ,
            nullable=nullable,
            unit=unit if isinstance(unit, str) else None,
            tz=tz if isinstance(tz, str) else None,
            byte_width=byte_width if isinstance(byte_width, int) else None,
        )
    if typ == "list":
        vt = raw.get("value_type")
        if not isinstance(vt, str) or vt not in ALLOWED_SCALAR_TYPES:
            raise DataContractError(f"invalid list value_type for {name}")
        return FieldSpec(name=name, type_name=typ, nullable=nullable, value_type=vt)
    if typ == "fixed_size_list":
        vt = raw.get("value_type")
        size = raw.get("list_size")
        if not isinstance(vt, str) or vt not in ALLOWED_SCALAR_TYPES:
            raise DataContractError(f"invalid fixed_size_list value_type for {name}")
        if not isinstance(size, int) or size <= 0 or size > MAX_FIXED_LIST_SIZE:
            raise DataContractError(f"invalid list_size for {name}")
        return FieldSpec(name=name, type_name=typ, nullable=nullable, value_type=vt, list_size=size)
    if typ == "struct":
        children = raw.get("fields")
        if not isinstance(children, list) or not children:
            raise DataContractError(f"struct {name} needs fields")
        parsed = tuple(_parse_field(c, depth=depth + 1) for c in children)
        names = [c.name for c in parsed]
        if len(names) != len(set(names)):
            raise DataContractError(f"duplicate nested field in {name}")
        return FieldSpec(name=name, type_name=typ, nullable=nullable, fields=parsed)
    if typ in ALLOWED_NESTED_TYPES:
        raise DataContractError(f"incomplete nested type for {name}")
    raise DataContractError(f"unknown type {typ!r} for {name}")


def parse_contract_dict(data: dict[str, Any], *, source_path: str | None = None) -> ContractSpec:
    unknown = set(data.keys()) - ALLOWED_TOP_KEYS
    if unknown:
        raise DataContractError(f"unknown contract keys: {sorted(unknown)}")
    for key in (
        "contract_name",
        "version",
        "description",
        "fields",
        "primary_key",
        "foreign_keys",
        "partition_by",
        "sort_by",
        "semantic_rules",
        "table_metadata",
    ):
        if key not in data:
            raise DataContractError(f"missing key: {key}")
    name = data["contract_name"]
    version = data["version"]
    if not isinstance(name, str) or not name:
        raise DataContractError("contract_name invalid")
    if not isinstance(version, int) or version < 0:
        raise DataContractError("version invalid")
    fields_raw = data["fields"]
    if not isinstance(fields_raw, list) or not fields_raw:
        raise DataContractError("fields must be non-empty list")
    fields = tuple(_parse_field(f, depth=0) for f in fields_raw)
    field_names = [f.name for f in fields]
    if len(field_names) != len(set(field_names)):
        raise DataContractError("duplicate field names")
    pk = data["primary_key"]
    if not isinstance(pk, list) or not pk or not all(isinstance(x, str) for x in pk):
        raise DataContractError("primary_key invalid")
    if any(x not in field_names for x in pk):
        raise DataContractError("primary_key references missing field")
    fks: list[ForeignKeySpec] = []
    for item in data["foreign_keys"]:
        if not isinstance(item, dict):
            raise DataContractError("foreign_key must be object")
        ff = item.get("fields")
        rf = item.get("ref_fields")
        rc = item.get("ref_contract")
        rv = item.get("ref_version")
        nullable = bool(item.get("nullable", False))
        if not isinstance(ff, list) or not isinstance(rf, list):
            raise DataContractError("foreign_key fields invalid")
        if len(ff) != len(rf):
            raise DataContractError("foreign_key arity mismatch")
        if not isinstance(rc, str) or not isinstance(rv, int):
            raise DataContractError("foreign_key ref invalid")
        if any(x not in field_names for x in ff):
            raise DataContractError("foreign_key local field missing")
        fks.append(
            ForeignKeySpec(
                fields=tuple(ff),
                ref_contract=rc,
                ref_fields=tuple(rf),
                ref_version=rv,
                nullable=nullable,
            )
        )
    meta = data["table_metadata"]
    if not isinstance(meta, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
    ):
        raise DataContractError("table_metadata must be string→string")
    rules = data["semantic_rules"]
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise DataContractError("semantic_rules invalid")
    part = data["partition_by"]
    sort = data["sort_by"]
    if not isinstance(part, list) or not isinstance(sort, list):
        raise DataContractError("partition_by/sort_by invalid")
    return ContractSpec(
        contract_name=name,
        version=version,
        description=str(data["description"]),
        fields=fields,
        primary_key=tuple(pk),
        foreign_keys=tuple(fks),
        partition_by=tuple(str(x) for x in part),
        sort_by=tuple(str(x) for x in sort),
        semantic_rules=tuple(dict(r) for r in rules),
        table_metadata=dict(meta),
        source_path=source_path,
    )


def load_contract_spec(path: Path, *, contain_root: Path | None = None) -> ContractSpec:
    target = Path(path)
    if target.is_symlink():
        raise DataContractError("contract spec must not be a symlink")
    if not target.is_file():
        raise DataContractError(f"contract spec missing: {target}")
    if contain_root is not None:
        try:
            target.resolve().relative_to(contain_root.resolve())
        except ValueError as exc:
            raise DataContractError("spec path escapes containment") from exc
    size = target.stat().st_size
    if size > MAX_SPEC_BYTES:
        raise DataContractError("contract spec exceeds size limit")
    raw = target.read_bytes()
    if len(raw) > MAX_SPEC_BYTES:
        raise DataContractError("contract spec exceeds size limit")
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DataContractError(f"invalid JSON: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise DataContractError("contract root must be object")
    return parse_contract_dict(data, source_path=str(target))
