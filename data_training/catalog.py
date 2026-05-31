"""Structure catalog records for training data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config_io import load_config, require_mapping, tuple_of_strings


@dataclass(frozen=True, slots=True)
class StructureCatalogRecord:
    """One structure source with searchable training metadata."""

    structure_id: str
    name: str
    source_path: str
    file_format: str = "cif"
    source_kind: str = "local_file"
    chemistry_family: str = ""
    dimensionality: str = ""
    texture_tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    expected_motifs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "StructureCatalogRecord":
        data = require_mapping(payload, context="structure record")
        source = require_mapping(data.get("source", {}), context="source")
        source_path = str(source.get("path") or data.get("source_path") or "")
        if not source_path:
            raise ValueError("structure record is missing source.path.")
        structure_id = str(data.get("structure_id") or Path(source_path).stem)
        return cls(
            structure_id=structure_id,
            name=str(data.get("name") or structure_id),
            source_path=source_path,
            file_format=str(
                source.get("format") or data.get("file_format") or "cif"
            ),
            source_kind=str(source.get("kind") or "local_file"),
            chemistry_family=str(data.get("chemistry_family") or ""),
            dimensionality=str(data.get("dimensionality") or ""),
            texture_tags=tuple_of_strings(data.get("texture_tags")),
            references=tuple_of_strings(data.get("references")),
            expected_motifs=tuple_of_strings(data.get("expected_motifs")),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the record into a stable manifest-friendly mapping."""

        return {
            "structure_id": self.structure_id,
            "name": self.name,
            "source": {
                "path": self.source_path,
                "format": self.file_format,
                "kind": self.source_kind,
            },
            "chemistry_family": self.chemistry_family,
            "dimensionality": self.dimensionality,
            "texture_tags": list(self.texture_tags),
            "references": list(self.references),
            "expected_motifs": list(self.expected_motifs),
            "metadata": dict(self.metadata),
        }

    def resolved_path(self, root: str | Path | None = None) -> Path:
        """Resolve the structure path against an optional catalog root."""

        path = Path(self.source_path)
        if path.is_absolute() or root is None:
            return path
        return Path(root) / path


@dataclass(frozen=True, slots=True)
class StructureCatalog:
    """A collection of structure records and path context."""

    catalog_id: str
    records: tuple[StructureCatalogRecord, ...]
    default_structure_root: str = "."
    schema_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "StructureCatalog":
        data = require_mapping(payload, context="structure catalog")
        records = tuple(
            StructureCatalogRecord.from_mapping(item)
            for item in data.get("records", [])
        )
        if not records:
            raise ValueError(
                "structure catalog must contain at least one record."
            )
        return cls(
            catalog_id=str(data.get("catalog_id") or "structure_catalog"),
            records=records,
            default_structure_root=str(
                data.get("default_structure_root") or "."
            ),
            schema_version=str(data.get("schema_version") or "1"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "StructureCatalog":
        return cls.from_mapping(load_config(path))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "default_structure_root": self.default_structure_root,
            "metadata": dict(self.metadata),
            "records": [record.to_mapping() for record in self.records],
        }

    def by_id(self) -> dict[str, StructureCatalogRecord]:
        return {record.structure_id: record for record in self.records}

    def select(
        self, structure_ids: Iterable[str] | str
    ) -> tuple[StructureCatalogRecord, ...]:
        """Return records by id, or all records when ``structure_ids`` is all."""

        if structure_ids == "all":
            return self.records
        wanted = set(tuple_of_strings(structure_ids))
        records = tuple(
            record for record in self.records if record.structure_id in wanted
        )
        missing = sorted(
            wanted.difference(record.structure_id for record in records)
        )
        if missing:
            raise KeyError(f"Unknown structure id(s): {', '.join(missing)}")
        return records

    def with_texture_tag(self, tag: str) -> tuple[StructureCatalogRecord, ...]:
        return tuple(
            record for record in self.records if tag in record.texture_tags
        )


def read_structure_catalog(path: str | Path) -> StructureCatalog:
    """Parse a structure catalog from JSON/YAML."""

    return StructureCatalog.from_file(path)
