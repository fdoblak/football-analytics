"""Strict taxonomy loader and model-class → entity/role mapping (Stage 5A)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.perception.types import (
    EntityType,
    PerceptionContractError,
    RoleLabel,
    RoleSource,
)

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

ENTITY_TYPES = frozenset(e.value for e in EntityType)
ROLE_LABELS = frozenset(r.value for r in RoleLabel)
ROLE_SOURCES = frozenset(s.value for s in RoleSource)
UNMAPPED_POLICIES = frozenset({"reject", "unknown"})


class TaxonomyError(PerceptionContractError):
    """Taxonomy config failure."""


@dataclass(frozen=True)
class ClassMappingResult:
    entity_type: EntityType
    role_label: RoleLabel
    role_source: RoleSource
    mapped: bool
    rejected: bool
    rule_note: str | None
    class_name_normalized: str


REQUIRED_TOP = frozenset(
    {
        "taxonomy_version",
        "config_version",
        "entity_types",
        "role_labels",
        "role_sources",
        "model_class_mapping",
        "unmapped_policy",
        "ball_role_policy",
        "auto_player_from_person",
        "notes",
    }
)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaxonomyError(f"{label} must be a mapping")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaxonomyError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TaxonomyError(f"{label} must be a bool")
    return value


def _require_str_list(
    value: Any, *, label: str, allowed: frozenset[str] | None = None
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TaxonomyError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise TaxonomyError(f"{label} entries must be non-empty strings")
        if allowed is not None and item not in allowed:
            raise TaxonomyError(f"{label} unsupported value: {item}")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_taxonomy(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise TaxonomyError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise TaxonomyError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise TaxonomyError(f"config_version must be {CONFIG_VERSION}")

    entity_types = _require_str_list(
        raw["entity_types"], label="entity_types", allowed=ENTITY_TYPES
    )
    if set(entity_types) != ENTITY_TYPES:
        raise TaxonomyError("entity_types must match canonical allowlist exactly")
    role_labels = _require_str_list(raw["role_labels"], label="role_labels", allowed=ROLE_LABELS)
    if set(role_labels) != ROLE_LABELS:
        raise TaxonomyError("role_labels must match canonical allowlist exactly")
    role_sources = _require_str_list(
        raw["role_sources"], label="role_sources", allowed=ROLE_SOURCES
    )
    if set(role_sources) != ROLE_SOURCES:
        raise TaxonomyError("role_sources must match canonical allowlist exactly")

    unmapped = _require_str(raw["unmapped_policy"], label="unmapped_policy")
    if unmapped not in UNMAPPED_POLICIES:
        raise TaxonomyError("unmapped_policy must be reject|unknown")
    ball_role = _require_str(raw["ball_role_policy"], label="ball_role_policy")
    if ball_role != "forbid_non_unknown":
        raise TaxonomyError("ball_role_policy must be forbid_non_unknown")
    auto_player = _require_bool(raw["auto_player_from_person"], label="auto_player_from_person")
    if auto_player:
        raise TaxonomyError("auto_player_from_person must be false")

    mapping = raw["model_class_mapping"]
    if not isinstance(mapping, list) or not mapping:
        raise TaxonomyError("model_class_mapping must be a non-empty list")
    seen_names: set[str] = set()
    cleaned_rules: list[dict[str, Any]] = []
    for i, rule in enumerate(mapping):
        block = dict(_require_mapping(rule, label=f"model_class_mapping[{i}]"))
        names = _require_str_list(
            block["class_names"], label=f"model_class_mapping[{i}].class_names"
        )
        names_norm = [n.strip().lower() for n in names]
        for n in names_norm:
            if n in seen_names:
                raise TaxonomyError(f"duplicate class_name mapping: {n}")
            seen_names.add(n)
        class_ids = block.get("class_ids")
        if class_ids is not None and (
            not isinstance(class_ids, list)
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in class_ids)
        ):
            raise TaxonomyError(f"model_class_mapping[{i}].class_ids must be int list or null")
        entity = _require_str(block["entity_type"], label=f"model_class_mapping[{i}].entity_type")
        if entity not in ENTITY_TYPES:
            raise TaxonomyError(f"model_class_mapping[{i}].entity_type invalid")
        role = _require_str(block["role_label"], label=f"model_class_mapping[{i}].role_label")
        if role not in ROLE_LABELS:
            raise TaxonomyError(f"model_class_mapping[{i}].role_label invalid")
        source = _require_str(block["role_source"], label=f"model_class_mapping[{i}].role_source")
        if source not in ROLE_SOURCES:
            raise TaxonomyError(f"model_class_mapping[{i}].role_source invalid")
        if entity == "ball" and role != "unknown":
            raise TaxonomyError("ball mapping must use role_label unknown")
        # Hard rule: person-family never auto-player
        if any(n in {"person", "human", "pedestrian"} for n in names_norm) and role == "player":
            raise TaxonomyError("person-family classes must not map to player")
        note = block.get("note")
        if note is not None and not isinstance(note, str):
            raise TaxonomyError(f"model_class_mapping[{i}].note must be string or null")
        cleaned_rules.append(
            {
                "class_names": names_norm,
                "class_ids": None if class_ids is None else list(class_ids),
                "entity_type": entity,
                "role_label": role,
                "role_source": source,
                "note": note,
            }
        )

    notes = raw["notes"]
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise TaxonomyError("notes must be a list of strings")

    return {
        "taxonomy_version": _require_str(raw["taxonomy_version"], label="taxonomy_version"),
        "config_version": CONFIG_VERSION,
        "entity_types": entity_types,
        "role_labels": role_labels,
        "role_sources": role_sources,
        "model_class_mapping": cleaned_rules,
        "unmapped_policy": unmapped,
        "ball_role_policy": ball_role,
        "auto_player_from_person": False,
        "notes": list(notes),
    }


def default_taxonomy_path(*, project_root: Path | None = None) -> Path:
    if project_root is None:
        from football_analytics.data.registry import default_project_root

        project_root = default_project_root()
    return project_root / "configs" / "perception" / "detection_taxonomy.yaml"


def load_detection_taxonomy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    cfg_path = path or default_taxonomy_path(project_root=project_root)
    if cfg_path.is_symlink():
        raise TaxonomyError(f"symlink rejected: {cfg_path}")
    if not cfg_path.is_file():
        raise TaxonomyError(f"taxonomy missing: {cfg_path}")
    size = cfg_path.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise TaxonomyError(f"taxonomy size out of bounds: {size}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TaxonomyError("taxonomy root must be a mapping")
    validated = _validate_taxonomy(raw)
    return _deep_freeze(validated)


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_unfreeze(v) for v in value]
    return value


def taxonomy_fingerprint(taxonomy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(taxonomy))


def map_model_class(
    class_id: int | None,
    class_name: str,
    *,
    taxonomy: Mapping[str, Any] | None = None,
) -> ClassMappingResult:
    """Map detector class → canonical entity/role.

    Generic person never becomes player. Unmapped follows taxonomy policy.
    """
    tax = taxonomy if taxonomy is not None else load_detection_taxonomy()
    if not isinstance(class_name, str) or not class_name.strip():
        raise TaxonomyError("class_name must be non-empty")
    name = class_name.strip().lower()
    if class_id is not None and (not isinstance(class_id, int) or isinstance(class_id, bool)):
        raise TaxonomyError("class_id must be int or None")

    for rule in tax["model_class_mapping"]:
        names = set(rule["class_names"])
        ids = rule["class_ids"]
        name_hit = name in names
        id_hit = ids is not None and class_id is not None and class_id in ids
        if not name_hit and not id_hit:
            continue
        # Prefer name match; class_id alone only if listed.
        if not name_hit and id_hit and name not in names:
            # id-only hit still applies
            pass
        entity = EntityType(rule["entity_type"])
        role = RoleLabel(rule["role_label"])
        source = RoleSource(rule["role_source"])
        if entity == EntityType.BALL and role != RoleLabel.UNKNOWN:
            raise TaxonomyError("ball role must remain unknown")
        if name in {"person", "human", "pedestrian"} and role == RoleLabel.PLAYER:
            raise TaxonomyError("person must not auto-map to player")
        return ClassMappingResult(
            entity_type=entity,
            role_label=role,
            role_source=source,
            mapped=True,
            rejected=False,
            rule_note=rule.get("note"),
            class_name_normalized=name,
        )

    policy = str(tax["unmapped_policy"])
    if policy == "reject":
        return ClassMappingResult(
            entity_type=EntityType.UNKNOWN,
            role_label=RoleLabel.UNKNOWN,
            role_source=RoleSource.UNKNOWN,
            mapped=False,
            rejected=True,
            rule_note="MODEL_CLASS_UNMAPPED",
            class_name_normalized=name,
        )
    return ClassMappingResult(
        entity_type=EntityType.UNKNOWN,
        role_label=RoleLabel.UNKNOWN,
        role_source=RoleSource.UNKNOWN,
        mapped=False,
        rejected=False,
        rule_note="unmapped→unknown",
        class_name_normalized=name,
    )


__all__ = [
    "CONFIG_VERSION",
    "TaxonomyError",
    "ClassMappingResult",
    "default_taxonomy_path",
    "load_detection_taxonomy",
    "taxonomy_fingerprint",
    "map_model_class",
]
