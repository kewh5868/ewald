from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List
import xarray as xr

# import helper and MetadataAttribute from your single_image module
from dataclass.single_image import MetadataAttribute, _validate_file_extension

@dataclass
class SeriesImage:
    """
    Data container for a stack of GIWAXS images (e.g., over time or temperature).

    Attributes:
      data: xarray.Dataset with one dimension (e.g., 'time' or 'temperature') holding 2D TIFF frames.
      dim_name: the name of the series dimension in `data`.
      mask_file: optional shared mask file (.edf, .json).
      poni_file: optional shared PONI geometry file (.poni).
      incidence_angle: incidence angle in degrees.
      polarization: polarization correction factor (0.0–1.0).
      solid_angle_on: whether solid-angle correction is enabled.
      frame_metadata: mapping from each coordinate along `dim_name` to its list of MetadataAttribute.
    """
    data: xr.Dataset
    dim_name: str
    mask_file: Optional[Path] = None
    poni_file: Optional[Path] = None
    incidence_angle: float = 0.0
    polarization: float = 0.0
    solid_angle_on: bool = False
    frame_metadata: Dict[Any, List[MetadataAttribute]] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure series dimension exists
        if self.dim_name not in self.data.dims:
            raise ValueError(f"Dimension '{self.dim_name}' not found in data (found dims: {self.data.dims})")

        # Validate shared files
        if self.mask_file:
            _validate_file_extension(self.mask_file, ['.edf', '.json'])
        if self.poni_file:
            _validate_file_extension(self.poni_file, ['.poni'])

    def get_frame_metadata(self, coord: Any) -> List[MetadataAttribute]:
        """Retrieve metadata for a specific frame coordinate."""
        return self.frame_metadata.get(coord, [])

    def add_frame_metadata(self, coord: Any, metadata: MetadataAttribute) -> None:
        """Append a metadata attribute to a specific frame."""
        self.frame_metadata.setdefault(coord, []).append(metadata)
