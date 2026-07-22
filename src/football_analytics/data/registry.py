"""Schema registry loader and graph checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from football_analytics.data import DataContractError
from football_analytics.data.specs import load_contract_spec
from football_analytics.data.types import ContractSpec

ALLOWED_STATUS = frozenset({"active", "deprecated", "experimental"})


@dataclass(frozen=True)
class MigrationEdge:
    frm: int
    to: int
    migration_id: str


@dataclass(frozen=True)
class ContractEntry:
    name: str
    current_version: int
    supported_versions: tuple[int, ...]
    status: str
    versions: dict[int, str]  # version -> rel spec path
    edges: tuple[MigrationEdge, ...]


@dataclass(frozen=True)
class SchemaRegistry:
    registry_version: int
    contracts: dict[str, ContractEntry]
    root: Path

    def list_contracts(self) -> list[str]:
        return sorted(self.contracts)

    def get_entry(self, name: str) -> ContractEntry:
        if name not in self.contracts:
            raise DataContractError(f"unknown contract: {name}")
        return self.contracts[name]

    def resolve_spec_path(self, name: str, version: int | None = None) -> Path:
        entry = self.get_entry(name)
        ver = entry.current_version if version is None else version
        if ver not in entry.versions:
            raise DataContractError(f"unsupported version {ver} for {name}")
        if ver not in entry.supported_versions:
            raise DataContractError(f"version {ver} not in supported_versions for {name}")
        rel = entry.versions[ver]
        path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise DataContractError("spec path escapes project root") from exc
        return path

    def load_contract(self, name: str, version: int | None = None) -> ContractSpec:
        path = self.resolve_spec_path(name, version)
        spec = load_contract_spec(path, contain_root=self.root)
        if spec.contract_name != name:
            raise DataContractError(
                f"contract_name mismatch: registry={name} spec={spec.contract_name}"
            )
        expected = self.get_entry(name).current_version if version is None else version
        if spec.version != expected:
            raise DataContractError(
                f"version mismatch for {name}: expected {expected} got {spec.version}"
            )
        return spec


def _validate_edges(name: str, supported: set[int], edges: list[MigrationEdge]) -> None:
    graph: dict[int, list[int]] = {v: [] for v in supported}
    for e in edges:
        if e.frm not in supported or e.to not in supported:
            raise DataContractError(f"{name}: dangling migration edge {e.frm}->{e.to}")
        if e.frm == e.to:
            raise DataContractError(f"{name}: self-edge forbidden")
        graph[e.frm].append(e.to)
    # detect cycles via DFS
    visiting: set[int] = set()
    visited: set[int] = set()

    def dfs(n: int) -> None:
        if n in visiting:
            raise DataContractError(f"{name}: migration cycle detected")
        if n in visited:
            return
        visiting.add(n)
        for nxt in graph.get(n, []):
            dfs(nxt)
        visiting.remove(n)
        visited.add(n)

    for node in list(graph):
        dfs(node)
    # ambiguous: more than one outgoing? allow chain only — multiple outs ok if unique path
    # reject duplicate edges
    seen = set()
    for e in edges:
        key = (e.frm, e.to)
        if key in seen:
            raise DataContractError(f"{name}: duplicate migration edge")
        seen.add(key)


def load_schema_registry(path: Path, *, project_root: Path | None = None) -> SchemaRegistry:
    reg_path = Path(path)
    if not reg_path.is_file() or reg_path.is_symlink():
        raise DataContractError("schema registry must be a regular file")
    root = project_root or reg_path.resolve().parents[2]
    try:
        data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DataContractError(f"registry YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise DataContractError("registry root must be mapping")
    if data.get("registry_version") != 1:
        raise DataContractError("registry_version must be 1")
    contracts_raw = data.get("contracts")
    if not isinstance(contracts_raw, dict) or not contracts_raw:
        raise DataContractError("contracts mapping required")
    contracts: dict[str, ContractEntry] = {}
    for name, raw in contracts_raw.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise DataContractError("invalid contract entry")
        if name in contracts:
            raise DataContractError(f"duplicate contract: {name}")
        status = raw.get("status")
        if status not in ALLOWED_STATUS:
            raise DataContractError(f"{name}: invalid status")
        current = raw.get("current_version")
        supported = raw.get("supported_versions")
        versions = raw.get("versions")
        edges_raw = raw.get("migration_edges")
        if not isinstance(current, int) or not isinstance(supported, list):
            raise DataContractError(f"{name}: version fields invalid")
        if not isinstance(versions, dict) or not isinstance(edges_raw, list):
            raise DataContractError(f"{name}: versions/edges invalid")
        supported_t = tuple(int(v) for v in supported)
        if current not in supported_t:
            raise DataContractError(f"{name}: current_version not supported")
        ver_map: dict[int, str] = {}
        for vk, vv in versions.items():
            ver = int(vk)
            if ver in ver_map:
                raise DataContractError(f"{name}: duplicate version {ver}")
            if not isinstance(vv, dict) or not isinstance(vv.get("spec_path"), str):
                raise DataContractError(f"{name}: version {ver} missing spec_path")
            ver_map[ver] = vv["spec_path"]
        if set(supported_t) != set(ver_map):
            raise DataContractError(f"{name}: supported_versions must match versions keys")
        edges = []
        for e in edges_raw:
            if not isinstance(e, dict):
                raise DataContractError(f"{name}: bad edge")
            edges.append(
                MigrationEdge(
                    frm=int(e["from"]), to=int(e["to"]), migration_id=str(e["migration_id"])
                )
            )
        _validate_edges(name, set(supported_t), edges)
        contracts[name] = ContractEntry(
            name=name,
            current_version=current,
            supported_versions=supported_t,
            status=status,
            versions=ver_map,
            edges=tuple(edges),
        )
    return SchemaRegistry(registry_version=1, contracts=contracts, root=root.resolve())


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "data" / "schema_registry.yaml"


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]
