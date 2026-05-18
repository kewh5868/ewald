"""Metadata inference for experimental detector filenames and
folders."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ewald.io.images import SUPPORTED_IMAGE_EXTENSIONS

_POSITION_PATTERN = re.compile(
    r"^(?P<axis>[xyz])(?P<value>[-+]?\d+(?:[.p]\d+)?)$"
)
_SAMPLE_PATTERN = re.compile(r"^(sam|sample)(?P<number>\d+)$")
_INTEGER_PATTERN = re.compile(r"^\d+$")


@dataclass(frozen=True, slots=True)
class MetadataField:
    """One inferred metadata field from a filename token."""

    key: str
    value: Any
    raw_token: str
    unit: str | None = None
    confidence: float = 0.8

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "raw_token": self.raw_token,
            "unit": self.unit,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class FilenameMetadata:
    """Metadata inference result for one detector filename."""

    path: Path
    delimiter: str
    tokens: list[str]
    fields: list[MetadataField] = field(default_factory=list)
    unresolved_tokens: list[str] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def get(self, key: str, default: Any = None) -> Any:
        for item in self.fields:
            if item.key == key:
                return item.value
        return default

    def as_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            field.key: field.value for field in self.fields
        }
        metadata["_metadata_delimiter"] = self.delimiter
        metadata["_metadata_token_count"] = self.token_count
        metadata["_metadata_tokens"] = self.tokens
        metadata["_metadata_fields"] = [
            field.as_dict() for field in self.fields
        ]
        if self.unresolved_tokens:
            metadata["_unresolved_tokens"] = self.unresolved_tokens
        return metadata


@dataclass(slots=True)
class FolderMetadataReport:
    """Summary of metadata consistency across a group of files."""

    delimiter: str
    file_count: int
    token_counts: dict[str, int]
    consistent_token_count: bool
    recurrent_exposure_time_s: float | None
    files_requiring_metadata_input: list[str]
    parsed_files: list[FilenameMetadata]

    def as_dict(self) -> dict[str, Any]:
        return {
            "delimiter": self.delimiter,
            "file_count": self.file_count,
            "token_counts": self.token_counts,
            "consistent_token_count": self.consistent_token_count,
            "recurrent_exposure_time_s": self.recurrent_exposure_time_s,
            "files_requiring_metadata_input": self.files_requiring_metadata_input,
        }


def infer_filename_metadata(
    path: str | Path,
    *,
    delimiter: str = "_",
) -> FilenameMetadata:
    """Infer detector metadata from a delimited filename."""

    file_path = Path(path)
    tokens = [token for token in file_path.stem.split(delimiter) if token]
    used_indexes: set[int] = set()
    fields: list[MetadataField] = []

    detector_index = _find_detector_index(tokens)
    sample_index = _find_sample_index(tokens)

    if sample_index is not None:
        token = tokens[sample_index]
        sample_match = _SAMPLE_PATTERN.match(token.lower())
        if sample_match:
            used_indexes.add(sample_index)
            number = int(sample_match.group("number"))
            fields.append(
                MetadataField(
                    key="sample_label",
                    value=token,
                    raw_token=token,
                    confidence=0.95,
                )
            )
            fields.append(
                MetadataField(
                    key="sample_number",
                    value=number,
                    raw_token=token,
                    confidence=0.95,
                )
            )
        composition_index = sample_index + 1
        if composition_index < len(tokens):
            composition = tokens[composition_index]
            used_indexes.add(composition_index)
            fields.append(
                MetadataField(
                    key="sample_composition",
                    value=composition,
                    raw_token=composition,
                    confidence=0.65,
                )
            )

    if detector_index is not None:
        token = tokens[detector_index]
        used_indexes.add(detector_index)
        fields.append(
            MetadataField(
                key="detector_type",
                value=token.lower(),
                raw_token=token,
                confidence=0.95,
            )
        )
        _append_frame_and_run_fields(
            tokens, detector_index, fields, used_indexes
        )

    duration_candidates: list[float] = []
    duration_indexes: list[int] = []
    for index, token in enumerate(tokens):
        lower = token.lower()
        if index in used_indexes:
            continue
        if _is_detector_token(lower):
            continue
        duration = _parse_duration_seconds(token)
        if duration is not None:
            duration_candidates.append(duration)
            duration_indexes.append(index)

    for index, token in enumerate(tokens):
        if index in used_indexes:
            continue
        field = _infer_token_field(token)
        if field is not None:
            used_indexes.add(index)
            fields.append(field)

    if duration_candidates:
        for index in duration_indexes:
            used_indexes.add(index)
        timestamp = max(duration_candidates)
        timestamp_token = tokens[
            duration_indexes[duration_candidates.index(timestamp)]
        ]
        fields.append(
            MetadataField(
                key="frame_timestamp_s",
                value=timestamp,
                raw_token=timestamp_token,
                unit="s",
                confidence=0.7,
            )
        )
        exposure = min(duration_candidates)
        exposure_token = tokens[
            duration_indexes[duration_candidates.index(exposure)]
        ]
        fields.append(
            MetadataField(
                key="exposure_time_s",
                value=exposure,
                raw_token=exposure_token,
                unit="s",
                confidence=0.7,
            )
        )
        fields.append(
            MetadataField(
                key="duration_candidates_s",
                value=duration_candidates,
                raw_token=",".join(
                    tokens[index] for index in duration_indexes
                ),
                unit="s",
                confidence=0.6,
            )
        )

    unresolved = [
        token
        for index, token in enumerate(tokens)
        if index not in used_indexes and not _is_detector_token(token.lower())
    ]
    return FilenameMetadata(
        path=file_path,
        delimiter=delimiter,
        tokens=tokens,
        fields=fields,
        unresolved_tokens=unresolved,
    )


def infer_folder_metadata(
    paths: Iterable[str | Path],
    *,
    delimiter: str = "_",
) -> FolderMetadataReport:
    """Infer metadata for a folder or explicit set of detector files."""

    detector_paths = [
        Path(item)
        for item in paths
        if Path(item).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    parsed_files = [
        infer_filename_metadata(path, delimiter=delimiter)
        for path in sorted(detector_paths)
    ]
    token_counts = Counter(str(item.token_count) for item in parsed_files)
    consistent_token_count = len(token_counts) <= 1
    exposure = _recurrent_exposure_time(parsed_files)

    if exposure is not None:
        _apply_recurrent_exposure(parsed_files, exposure)

    files_requiring_input = [
        str(item.path)
        for item in parsed_files
        if item.unresolved_tokens or not consistent_token_count
    ]
    return FolderMetadataReport(
        delimiter=delimiter,
        file_count=len(parsed_files),
        token_counts=dict(token_counts),
        consistent_token_count=consistent_token_count,
        recurrent_exposure_time_s=exposure,
        files_requiring_metadata_input=files_requiring_input,
        parsed_files=parsed_files,
    )


def detector_files_in_folder(folder: str | Path) -> list[Path]:
    """Return supported detector image files from one folder."""

    folder_path = Path(folder)
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _infer_token_field(token: str) -> MetadataField | None:
    lower = token.lower()

    if lower in {"filt", "filtered"}:
        return MetadataField(
            "filtration_status", "filtered", token, confidence=0.9
        )
    if lower in {"unfilt", "unfiltered"}:
        return MetadataField(
            "filtration_status", "unfiltered", token, confidence=0.9
        )
    if lower in {"si", "ito", "glass"}:
        return MetadataField("substrate", token, token, confidence=0.9)
    if lower.endswith("scfh"):
        value = _parse_number(token[:-4])
        if value is not None:
            return MetadataField(
                "flow_rate_scfh", value, token, unit="scfh", confidence=0.9
            )
    if lower.endswith("ul"):
        value = _parse_number(token[:-2])
        if value is not None:
            return MetadataField(
                "solution_volume_uL", value, token, unit="uL", confidence=0.9
            )
    if lower.endswith("m"):
        value = _parse_number(token[:-1])
        if value is not None and ("p" in lower or "." in lower):
            return MetadataField(
                "concentration_molar", value, token, unit="M", confidence=0.85
            )
    position_match = _POSITION_PATTERN.match(lower)
    if position_match:
        value = _parse_number(position_match.group("value"))
        if value is not None:
            return MetadataField(
                f"{position_match.group('axis')}_position",
                value,
                token,
                confidence=0.85,
            )
    if _looks_like_incidence_angle(lower):
        value = _parse_number(_strip_alpha_prefix(token))
        if value is not None:
            return MetadataField(
                "incidence_angle_deg",
                value,
                token,
                unit="deg",
                confidence=0.85,
            )
    return None


def _append_frame_and_run_fields(
    tokens: list[str],
    detector_index: int,
    fields: list[MetadataField],
    used_indexes: set[int],
) -> None:
    frame_index = detector_index - 1
    if frame_index >= 0 and _INTEGER_PATTERN.match(tokens[frame_index]):
        used_indexes.add(frame_index)
        fields.append(
            MetadataField(
                "frame_number",
                int(tokens[frame_index]),
                tokens[frame_index],
                confidence=0.8,
            )
        )
    numeric_before_frame = [
        (index, int(token))
        for index, token in enumerate(tokens[:frame_index])
        if _INTEGER_PATTERN.match(token)
    ]
    if numeric_before_frame:
        run_index, run_value = max(
            numeric_before_frame, key=lambda item: item[1]
        )
        used_indexes.add(run_index)
        fields.append(
            MetadataField(
                "run_id",
                run_value,
                tokens[run_index],
                confidence=0.75,
            )
        )


def _apply_recurrent_exposure(
    parsed_files: list[FilenameMetadata],
    exposure: float,
) -> None:
    for result in parsed_files:
        filtered = [
            field
            for field in result.fields
            if field.key not in {"exposure_time_s", "duration_candidates_s"}
        ]
        candidates = [
            field
            for field in result.fields
            if field.key == "duration_candidates_s"
        ]
        raw_token = str(exposure)
        if candidates:
            raw_values = candidates[0].raw_token.split(",")
            for token in raw_values:
                if _parse_duration_seconds(token) == exposure:
                    raw_token = token
                    break
            filtered.extend(candidates)
        filtered.append(
            MetadataField(
                "exposure_time_s",
                exposure,
                raw_token,
                unit="s",
                confidence=0.9,
            )
        )
        result.fields = filtered


def _recurrent_exposure_time(
    parsed_files: list[FilenameMetadata],
) -> float | None:
    if not parsed_files:
        return None
    counts: Counter[float] = Counter()
    for result in parsed_files:
        candidates = result.get("duration_candidates_s", [])
        for value in candidates:
            counts[round(float(value), 6)] += 1
    recurrent = [
        value
        for value, count in counts.items()
        if count > 1 or len(parsed_files) == 1
    ]
    if not recurrent:
        return None
    return min(recurrent)


def _find_detector_index(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if _is_detector_token(token.lower()):
            return index
    return None


def _find_sample_index(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if _SAMPLE_PATTERN.match(token.lower()):
            return index
    return None


def _is_detector_token(token: str) -> bool:
    return token in {"maxs", "saxs", "waxs"}


def _looks_like_incidence_angle(token: str) -> bool:
    return (
        token.startswith("th")
        or token.startswith("theta")
        or token.startswith("incident")
        or token.startswith("incidence")
        or token.startswith("ai")
    )


def _strip_alpha_prefix(token: str) -> str:
    return re.sub(r"^[A-Za-z]+", "", token)


def _parse_duration_seconds(token: str) -> float | None:
    lower = token.lower()
    if not lower.endswith("s") or _is_detector_token(lower):
        return None
    return _parse_number(token[:-1])


def _parse_number(text: str) -> float | None:
    normalized = re.sub(r"(?<=\d)p(?=\d)", ".", text.lower())
    match = re.search(r"[-+]?\d+(?:\.\d+)?", normalized)
    if match is None:
        return None
    return float(match.group(0))
