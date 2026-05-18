"""High-level data import helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ewald.data.models import DataFileRef, DataGroupRef
from ewald.io.metadata import (
    FolderMetadataReport,
    detector_files_in_folder,
    infer_folder_metadata,
)


def build_data_group_from_paths(
    paths: Iterable[str | Path],
    *,
    group_name: str | None = None,
    group_path: str | Path | None = None,
    import_kind: str = "file",
    metadata_type: str = "filename",
    delimiter: str = "_",
    metadata_yml: str | Path | None = None,
) -> tuple[DataGroupRef, FolderMetadataReport]:
    """Build a project data group from image paths and filename metadata."""

    path_list = [Path(path) for path in paths]
    report = infer_folder_metadata(path_list, delimiter=delimiter)
    sidecar_metadata = _load_sidecar_metadata(metadata_yml)
    if group_name is None:
        if group_path is not None:
            group_name = Path(group_path).name
        elif len(path_list) == 1:
            group_name = path_list[0].stem
        else:
            group_name = "Experimental Data"
    group = DataGroupRef(
        name=group_name,
        path=Path(group_path) if group_path is not None else None,
        import_kind=import_kind,
        metadata_type=metadata_type,
        delimiter=delimiter,
        parse_report=report.as_dict(),
    )
    for parsed in report.parsed_files:
        metadata = parsed.as_metadata()
        metadata.update(_metadata_for_path(sidecar_metadata, parsed.path))
        if "header" in metadata_type:
            metadata.update(_load_tiff_header_metadata(parsed.path))
        metadata["_metadata_type"] = metadata_type
        if metadata_yml is not None:
            metadata["_metadata_sidecar"] = str(Path(metadata_yml))
        group.data_files.append(
            DataFileRef(
                path=parsed.path,
                data_id=parsed.path.stem,
                kind="detector-image",
                metadata=metadata,
            )
        )
    return group, report


def build_data_group_from_folder(
    folder: str | Path,
    *,
    delimiter: str = "_",
    metadata_type: str = "filename",
    metadata_yml: str | Path | None = None,
) -> tuple[DataGroupRef, FolderMetadataReport]:
    """Build a project data group from all supported image files in a folder."""

    folder_path = Path(folder)
    return build_data_group_from_paths(
        detector_files_in_folder(folder_path),
        group_name=folder_path.name,
        group_path=folder_path,
        import_kind="folder",
        metadata_type=metadata_type,
        delimiter=delimiter,
        metadata_yml=metadata_yml,
    )


def _load_sidecar_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Metadata sidecar YAML must contain a mapping.")
    return payload


def _metadata_for_path(
    payload: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    if not payload:
        return {}
    defaults = payload.get("defaults", {})
    file_entries = payload.get("files", payload.get("data", {}))
    if (
        "defaults" not in payload
        and "files" not in payload
        and "data" not in payload
    ):
        defaults = payload
        file_entries = {}
    metadata = dict(defaults or {})
    if isinstance(file_entries, list):
        file_entries = {
            str(item.get("path") or item.get("file") or item.get("name")): item
            for item in file_entries
            if isinstance(item, dict)
        }
    if isinstance(file_entries, dict):
        for key in (str(path), path.name, path.stem):
            if key in file_entries and isinstance(file_entries[key], dict):
                metadata.update(file_entries[key])
    return metadata


def _load_tiff_header_metadata(path: Path) -> dict[str, Any]:
    try:
        import tifffile
    except ImportError:
        return {"_header_metadata_error": "tifffile is not installed"}

    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = {
                tag.name: _safe_tiff_value(tag.value)
                for tag in page.tags.values()
                if tag.name
            }
    except Exception as exc:  # pragma: no cover - depends on TIFF variants.
        return {"_header_metadata_error": str(exc)}
    return {"_tiff_header": tags}


def _safe_tiff_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_tiff_value(item) for item in value]
    return str(value)
