"""Structure catalog loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import yaml

from .schemas import ConfigError, StructureRecord


def load_structure_catalog(path: str | Path) -> list[StructureRecord]:
    """Load a YAML/JSON catalog into structure records."""

    catalog_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):
        entries = payload.get("structures", [])
    else:
        raise ConfigError("Structure catalog must be a list or mapping.")
    records = [StructureRecord.from_mapping(entry) for entry in entries]
    _ensure_unique_ids(records)
    return records


def validate_catalog_paths(
    records: Iterable[StructureRecord],
    catalog_path: str | Path,
) -> list[str]:
    """Return human-readable catalog path errors."""

    catalog_root = Path(catalog_path).expanduser().resolve().parent
    errors: list[str] = []
    for record in records:
        resolved = record.resolved_path(catalog_root)
        if not resolved.exists():
            errors.append(
                f"{record.structure_id}: missing structure file {resolved}"
            )
        elif record.file_format.lower() not in {"cif", "mcif", "poscar"}:
            errors.append(
                f"{record.structure_id}: unsupported format "
                f"{record.file_format!r}"
            )
    return errors


def catalog_as_lookup(
    records: Iterable[StructureRecord],
) -> dict[str, StructureRecord]:
    """Return records keyed by structure id."""

    return {record.structure_id: record for record in records}


def _ensure_unique_ids(records: list[StructureRecord]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.structure_id in seen:
            duplicates.add(record.structure_id)
        seen.add(record.structure_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ConfigError(f"Duplicate structure_id values: {joined}")


def discover_structures(
    root: str | Path,
    *,
    family: str = "",
    phase_class: str = "",
) -> list[StructureRecord]:
    """Create catalog records for CIF/POSCAR files under a directory."""

    root_path = Path(root).expanduser().resolve()
    patterns = ("*.cif", "*.mcif", "POSCAR", "*.vasp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root_path.rglob(pattern))
    records: list[StructureRecord] = []
    for path in sorted(set(files)):
        rel_path = path.relative_to(root_path)
        records.append(
            StructureRecord.from_mapping(
                {
                    "name": path.stem,
                    "path": str(rel_path),
                    "file_format": _infer_format(path),
                    "family": family,
                    "phase_class": phase_class,
                }
            )
        )
    return records


def _infer_format(path: Path) -> str:
    if path.name.upper() == "POSCAR" or path.suffix.lower() == ".vasp":
        return "poscar"
    return path.suffix.lower().lstrip(".") or "cif"
