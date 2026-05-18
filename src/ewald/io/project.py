"""Read and write EWALD ``.ewld`` project files."""

from __future__ import annotations

import json
import posixpath
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from ewald.data.models import (
    PROJECT_EXTENSION,
    CorrectionAssetRef,
    DataFileRef,
    ProjectState,
)

MANIFEST_NAME = "manifest.json"
ARCHIVE_DATA_DIR = "data_files"
ARCHIVE_MASK_DIR = "assets/masks"
ARCHIVE_CALIBRANT_DIR = "assets/calibrants"
ARCHIVE_GENERATED_CIF_DIR = "structures/generated_cifs"


def normalize_project_path(path: str | Path) -> Path:
    project_path = Path(path)
    if project_path.suffix != PROJECT_EXTENSION:
        project_path = project_path.with_suffix(PROJECT_EXTENSION)
    return project_path


def save_project(project: ProjectState, path: str | Path) -> Path:
    project_path = normalize_project_path(path)
    project_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(project_path, "w", compression=ZIP_DEFLATED) as archive:
        payload = _archive_project_payload(project, archive)
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(payload, indent=2, sort_keys=True),
        )
    return project_path


def load_project(path: str | Path) -> ProjectState:
    project_path = normalize_project_path(path)
    with ZipFile(project_path, "r") as archive:
        with archive.open(MANIFEST_NAME) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        extraction_root = _extraction_root(project_path)
        _extract_archived_payload_files(archive, payload, extraction_root)
    project = ProjectState.from_dict(payload)
    _attach_extracted_paths(project, extraction_root)
    return project


def _archive_project_payload(
    project: ProjectState,
    archive: ZipFile,
) -> dict[str, Any]:
    payload = project.as_dict()
    for group, group_payload in zip(
        project.data_groups,
        payload["data_groups"],
        strict=False,
    ):
        if group.import_kind != "file":
            continue
        for data_file, data_file_payload in zip(
            group.data_files,
            group_payload["data_files"],
            strict=False,
        ):
            archive_path = _write_ref_file(
                archive,
                data_file,
                ARCHIVE_DATA_DIR,
                data_file.data_id or data_file.path.stem,
            )
            if archive_path:
                data_file_payload["archive_path"] = archive_path

    for data_file, data_file_payload in zip(
        project.data_files,
        payload["data_files"],
        strict=False,
    ):
        archive_path = _write_ref_file(
            archive,
            data_file,
            ARCHIVE_DATA_DIR,
            data_file.data_id or data_file.path.stem,
        )
        if archive_path:
            data_file_payload["archive_path"] = archive_path

    for asset, asset_payload in zip(
        project.masks,
        payload["masks"],
        strict=False,
    ):
        archive_path = _write_asset_file(archive, asset, ARCHIVE_MASK_DIR)
        if archive_path:
            asset_payload["archive_path"] = archive_path

    for asset, asset_payload in zip(
        project.calibrants,
        payload["calibrants"],
        strict=False,
    ):
        archive_path = _write_asset_file(archive, asset, ARCHIVE_CALIBRANT_DIR)
        if archive_path:
            asset_payload["archive_path"] = archive_path
    _archive_generated_cif_records(payload, archive)
    return payload


def _write_ref_file(
    archive: ZipFile,
    data_file: DataFileRef,
    archive_dir: str,
    archive_id: str,
) -> str | None:
    source = data_file.usable_path
    if not source.is_file():
        return None
    archive_path = _archive_member_path(archive_dir, archive_id, source.name)
    archive.write(source, archive_path)
    return archive_path


def _write_asset_file(
    archive: ZipFile,
    asset: CorrectionAssetRef,
    archive_dir: str,
) -> str | None:
    source = asset.usable_path
    if source is None or not source.is_file():
        return None
    archive_path = _archive_member_path(
        archive_dir,
        asset.asset_id or asset.name,
        source.name,
    )
    archive.write(source, archive_path)
    return archive_path


def _archive_generated_cif_records(
    payload: dict[str, Any],
    archive: ZipFile,
) -> None:
    archive_paths_by_id: dict[str, str] = {}
    reference_generated = payload.get("reference_cifs", {}).get(
        "generated",
        {},
    )
    if isinstance(reference_generated, dict):
        for cif_id, record in reference_generated.items():
            if isinstance(record, dict):
                _archive_generated_cif_record(
                    archive,
                    record,
                    str(record.get("cif_id") or cif_id),
                    archive_paths_by_id,
                )

    structures = payload.get("structures", {})
    if isinstance(structures, dict):
        for structure_id, record in structures.items():
            if not isinstance(record, dict):
                continue
            if not _is_generated_cif_record(record):
                continue
            _archive_generated_cif_record(
                archive,
                record,
                str(
                    record.get("cif_id")
                    or record.get("structure_id")
                    or structure_id
                ),
                archive_paths_by_id,
            )

    for record in _iter_analysis_generated_cifs(payload):
        _archive_generated_cif_record(
            archive,
            record,
            str(record.get("cif_id") or record.get("id") or "generated_cif"),
            archive_paths_by_id,
        )


def _archive_generated_cif_record(
    archive: ZipFile,
    record: dict[str, Any],
    record_id: str,
    archive_paths_by_id: dict[str, str],
) -> str | None:
    if not record_id:
        record_id = "generated_cif"
    archive_path = archive_paths_by_id.get(record_id)
    if archive_path:
        record["archive_path"] = archive_path
        return archive_path

    source_path = _record_existing_path(record)
    filename = (
        source_path.name
        if source_path is not None
        else f"{_safe_archive_segment(record_id)}.cif"
    )
    archive_path = _archive_member_path(
        ARCHIVE_GENERATED_CIF_DIR,
        record_id,
        filename,
    )
    if source_path is not None:
        archive.write(source_path, archive_path)
    else:
        cif_text = str(record.get("cif_text") or "")
        if not cif_text.strip():
            return None
        archive.writestr(archive_path, cif_text)
    record["archive_path"] = archive_path
    archive_paths_by_id[record_id] = archive_path
    return archive_path


def _record_existing_path(record: dict[str, Any]) -> Path | None:
    for key in ("local_path", "path", "structure_path"):
        value = record.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_file():
            return path
    return None


def _is_generated_cif_record(record: dict[str, Any]) -> bool:
    source = str(record.get("source", "")).lower()
    return (
        bool(record.get("cif_text"))
        or bool(record.get("cif_id"))
        or "generated_cif" in source
        or "structure_analysis" in source
    )


def _iter_analysis_generated_cifs(payload: dict[str, Any]):
    structure_analysis = payload.get("analysis_results", {}).get(
        "structure_analysis",
        {},
    )
    if not isinstance(structure_analysis, dict):
        return
    for analysis in structure_analysis.values():
        if not isinstance(analysis, dict):
            continue
        wyckoff = analysis.get("wyckoff", {})
        if not isinstance(wyckoff, dict):
            continue
        for record in wyckoff.get("generated_cifs", []):
            if isinstance(record, dict):
                yield record


def _archive_member_path(
    archive_dir: str,
    item_id: str,
    filename: str,
) -> str:
    return posixpath.join(
        archive_dir,
        _safe_archive_segment(item_id),
        _safe_archive_segment(filename),
    )


def _safe_archive_segment(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._")
    return safe or "file"


def _extraction_root(project_path: Path) -> Path:
    return project_path.with_suffix(".ewld_assets")


def _extract_archived_payload_files(
    archive: ZipFile,
    payload: dict[str, Any],
    extraction_root: Path,
) -> None:
    for archive_path in _iter_archive_paths(payload):
        if archive_path in archive.namelist():
            _extract_member(archive, archive_path, extraction_root)


def _iter_archive_paths(payload: dict[str, Any]):
    for group in payload.get("data_groups", []):
        for data_file in group.get("data_files", []):
            archive_path = data_file.get("archive_path")
            if archive_path:
                yield archive_path
    for data_file in payload.get("data_files", []):
        archive_path = data_file.get("archive_path")
        if archive_path:
            yield archive_path
    for key in ("masks", "calibrants"):
        for asset in payload.get(key, []):
            archive_path = asset.get("archive_path")
            if archive_path:
                yield archive_path
    reference_generated = payload.get("reference_cifs", {}).get(
        "generated",
        {},
    )
    if isinstance(reference_generated, dict):
        for record in reference_generated.values():
            if isinstance(record, dict) and record.get("archive_path"):
                yield record["archive_path"]
    structures = payload.get("structures", {})
    if isinstance(structures, dict):
        for record in structures.values():
            if isinstance(record, dict) and record.get("archive_path"):
                yield record["archive_path"]
    for record in _iter_analysis_generated_cifs(payload):
        if record.get("archive_path"):
            yield record["archive_path"]


def _extract_member(
    archive: ZipFile,
    archive_path: str,
    extraction_root: Path,
) -> Path:
    target = _target_for_archive_path(extraction_root, archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(archive_path) as source:
        target.write_bytes(source.read())
    return target


def _target_for_archive_path(
    extraction_root: Path,
    archive_path: str,
) -> Path:
    parts = Path(archive_path).parts
    if Path(archive_path).is_absolute() or ".." in parts:
        raise ValueError(f"Unsafe archive path: {archive_path}")
    target = extraction_root.joinpath(*parts)
    resolved_root = extraction_root.resolve()
    resolved_target_parent = target.parent.resolve()
    if resolved_root not in (
        resolved_target_parent,
        *resolved_target_parent.parents,
    ):
        raise ValueError(f"Unsafe archive path: {archive_path}")
    return target


def _attach_extracted_paths(
    project: ProjectState,
    extraction_root: Path,
) -> None:
    for group in project.data_groups:
        for data_file in group.data_files:
            _attach_data_file_path(data_file, extraction_root)
    for data_file in project.data_files:
        _attach_data_file_path(data_file, extraction_root)
    for asset in [*project.masks, *project.calibrants]:
        if asset.archive_path:
            asset.local_path = _target_for_archive_path(
                extraction_root,
                asset.archive_path,
            )
    _attach_generated_cif_paths(project, extraction_root)


def _attach_data_file_path(
    data_file: DataFileRef,
    extraction_root: Path,
) -> None:
    if data_file.archive_path:
        data_file.local_path = _target_for_archive_path(
            extraction_root,
            data_file.archive_path,
        )


def _attach_generated_cif_paths(
    project: ProjectState,
    extraction_root: Path,
) -> None:
    for record in _iter_project_generated_cif_records(project):
        archive_path = record.get("archive_path")
        if not archive_path:
            continue
        local_path = _target_for_archive_path(extraction_root, archive_path)
        record["local_path"] = str(local_path)
        path_value = record.get("path")
        path_exists = bool(path_value) and Path(str(path_value)).is_file()
        if not path_exists:
            record["path"] = str(local_path)
        structure_path_value = record.get("structure_path")
        structure_path_exists = (
            bool(structure_path_value)
            and Path(str(structure_path_value)).is_file()
        )
        if not structure_path_exists:
            record["structure_path"] = str(local_path)


def _iter_project_generated_cif_records(project: ProjectState):
    reference_generated = project.reference_cifs.get("generated", {})
    if isinstance(reference_generated, dict):
        for record in reference_generated.values():
            if isinstance(record, dict):
                yield record
    for record in project.structures.values():
        if isinstance(record, dict) and _is_generated_cif_record(record):
            yield record
    structure_analysis = project.analysis_results.get("structure_analysis", {})
    if not isinstance(structure_analysis, dict):
        return
    for analysis in structure_analysis.values():
        if not isinstance(analysis, dict):
            continue
        wyckoff = analysis.get("wyckoff", {})
        if not isinstance(wyckoff, dict):
            continue
        for record in wyckoff.get("generated_cifs", []):
            if isinstance(record, dict):
                yield record
