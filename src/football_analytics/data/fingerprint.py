"""Contract / schema fingerprinting."""

from __future__ import annotations

from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.types import (
    FINGERPRINT_EXCLUDE_KEYS,
    META_FINGERPRINT,
    ContractSpec,
    FieldSpec,
)


def _field_to_dict(field: FieldSpec) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": field.name,
        "type": field.type_name,
        "nullable": field.nullable,
    }
    if field.value_type is not None:
        d["value_type"] = field.value_type
    if field.list_size is not None:
        d["list_size"] = field.list_size
    if field.byte_width is not None:
        d["byte_width"] = field.byte_width
    if field.unit is not None:
        d["unit"] = field.unit
    if field.tz is not None:
        d["tz"] = field.tz
    if field.fields:
        d["fields"] = [_field_to_dict(c) for c in field.fields]
    return d


def normalized_contract_dict(spec: ContractSpec) -> dict[str, Any]:
    """Semantic normalization for fingerprint (excludes description/source_path)."""
    return {
        "contract_name": spec.contract_name,
        "version": spec.version,
        "fields": [_field_to_dict(f) for f in spec.fields],
        "primary_key": list(spec.primary_key),
        "foreign_keys": [
            {
                "fields": list(fk.fields),
                "ref_contract": fk.ref_contract,
                "ref_fields": list(fk.ref_fields),
                "ref_version": fk.ref_version,
                "nullable": fk.nullable,
            }
            for fk in spec.foreign_keys
        ],
        "partition_by": list(spec.partition_by),
        "sort_by": list(spec.sort_by),
        "semantic_rules": [
            {k: v for k, v in rule.items() if k not in FINGERPRINT_EXCLUDE_KEYS}
            for rule in spec.semantic_rules
        ],
        "table_metadata": dict(sorted(spec.table_metadata.items())),
    }


def contract_fingerprint(spec: ContractSpec) -> str:
    return hash_canonical_json(normalized_contract_dict(spec))


def arrow_schema_fingerprint(schema: Any) -> str:
    """Fingerprint structural Arrow schema (names/types/nullability/order)."""
    fields = []
    for f in schema:
        fields.append(
            {
                "name": f.name,
                "type": str(f.type),
                "nullable": f.nullable,
            }
        )
    return hash_canonical_json({"fields": fields})


def verify_schema_fingerprint(schema: Any, expected: str) -> bool:
    meta = schema.metadata or {}
    key = META_FINGERPRINT.encode("utf-8")
    actual = meta.get(key, b"").decode("utf-8")
    return actual == expected and len(expected) == 64
