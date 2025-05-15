from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union


def _validate_file_extension(path: Path, allowed: List[str]) -> Path:
    """
    Ensure the file has one of the allowed extensions.
    """
    if path.suffix.lower() not in allowed:
        raise ValueError(f"Invalid file extension: {path.suffix}. Expected one of: {allowed}")
    return path


@dataclass
class MetadataAttribute:
    """
    Represents a user-defined metadata field for a single image.
    name: the metadata key
    value: the metadata value (string or numeric)
    is_value: True if treated as numeric value, False if treated as string
    """
    name: str
    value: Union[str, float, int]
    is_value: bool  # True for numeric, False for string


@dataclass
class SingleImage:
    """
    Data container for a single GIWAXS image and its associated settings.
    - file_path: path to the image (e.g. .tiff, .tif)
    - mask_file: optional mask file (e.g. .edf, .json)
    - poni_file: optional PONI geometry file (.poni)
    - incidence_angle: incidence angle in degrees
    - polarization: polarization correction factor (0.0–1.0)
    - solid_angle_on: whether solid-angle correction is enabled
    - metadata_attributes: user-defined metadata fields
    """
    file_path: Path
    mask_file: Optional[Path] = None
    poni_file: Optional[Path] = None
    incidence_angle: float = 0.0
    polarization: float = 0.0
    solid_angle_on: bool = False
    metadata_attributes: List[MetadataAttribute] = field(default_factory=list)

    def __post_init__(self):
        # Validate file extensions
        self.file_path = _validate_file_extension(self.file_path, ['.tiff', '.tif'])
        if self.mask_file:
            self.mask_file = _validate_file_extension(self.mask_file, ['.edf', '.json'])
        if self.poni_file:
            self.poni_file = _validate_file_extension(self.poni_file, ['.poni'])

    def add_metadata(self, name: str, value: Union[str, float, int], is_value: bool):
        """
        Add a new metadata attribute to this image.
        """
        self.metadata_attributes.append(MetadataAttribute(name, value, is_value))

    def remove_metadata(self, name: str):
        """
        Remove all metadata attributes with the given name.
        """
        self.metadata_attributes = [m for m in self.metadata_attributes if m.name != name]

    def get_metadata(self, name: str) -> List[MetadataAttribute]:
        """
        Retrieve metadata attributes matching the given name.
        """
        return [m for m in self.metadata_attributes if m.name == name]
