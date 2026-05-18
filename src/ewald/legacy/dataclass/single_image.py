from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union
import importlib.util
import warnings

import PyHyperScattering as phs
# from pyhyper import PFFIGeneralIntegrator

# (optional) mute PyHyperScattering-wide UserWarnings
warnings.filterwarnings("ignore",
    ".*Unable to load optional dependency.*",
    category=UserWarning,
    module="PyHyperScattering"
)

# Path to your local file
file_path = Path("/Users/keithwhite/repos/PyHyperScattering/src/"
                 "PyHyperScattering/PFFIGeneralIntegrator.py")

spec = importlib.util.spec_from_file_location("local_pffig", str(file_path))
local_pffig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_pffig)

# Grab the class
PFFIGeneralIntegrator = local_pffig.PFFIGeneralIntegrator

# # Test it
# print(PFFIGeneralIntegrator)

def _validate_file_extension(path: Union[str, Path], allowed: List[str]) -> Path:
    """
    Ensure the file has one of the allowed extensions.
    Accepts either a string or a Path.
    Returns a pathlib.Path on success.
    """
    path = Path(path)  # if it was a str, turn it into a Path
    suffix = path.suffix.lower()
    # normalize allowed extensions to lowercase as well
    allowed_lc = [ext.lower() for ext in allowed]
    if suffix not in allowed_lc:
        raise ValueError(f"Invalid file extension: {suffix!r}. Expected one of: {allowed_lc}")
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
    Attributes:
    - data_name: name of the data object
    - file_path: path to the image (e.g. .tiff, .tif)
    - mask_file: optional mask file (e.g. .edf, .json)
    - poni_file: optional PONI geometry file (.poni)
    - incident_angle: incidence angle in degrees
    - tilt_angle: tilt angle in degrees, converted to radians later on.
    - sample_orientation: sample orientation (1-8) for detector rotation
    - split_pixels: whether to split (rebin) pixels (True/False)
    - output_space: output space for integration ('recip', 'polar', or 'both')
    - polarization: polarization correction factor (0.0–1.0)
    - solid_angle: whether solid-angle correction is enabled
    - metadata_attributes: user-defined metadata fields
    """
    data_name: str
    file_path: Path
    mask_file: Optional[Path] = None
    poni_file: Optional[Path] = None
    incident_angle: float = 0.3
    tilt_angle: float = 0.0
    sample_orientation: int = 4  # 1-8, default to 4
    split_pixels: bool = True
    output_space: str = 'recip'  # 'recip', 'polar', or 'both'
    polarization: float = 0.95  ## For synchrotron data, set to 0.95 typically.
    solid_angle: bool = True
    metadata_attributes: List[MetadataAttribute] = field(default_factory=list)
    type: str = field(init=False, default="single")

    def __post_init__(self):
        # Validate file extensions
        self.file_path = _validate_file_extension(self.file_path, ['.tiff', '.tif'])
        if self.mask_file:
            self.mask_file = _validate_file_extension(self.mask_file, ['.edf', '.json'])
        if self.poni_file:
            self.poni_file = _validate_file_extension(self.poni_file, ['.poni'])

        # Initialize PyHyperScattering loader
        metadata_list = [m.name for m in self.metadata_attributes]
        # Ensure required metadata fields are included
        for required in ['data_name','incident_angle', 'tilt_angle', 'sample_orientation']:
            if required not in metadata_list:
                metadata_list.append(required)

        # *** temporary override for testing
        metadata_list = ['sample', 'material', 'filter', 
                 'concentration', 'flowrate', 'substrate', 
                 'solution_volume', 'runNumber', 'global_time', 
                 'xpos', 'incident_angle', 'exposure_time', 
                 'scan_id', 'scan_number', 'detector']
        
        self.loader = phs.load.CMSGIWAXSLoader(md_naming_scheme=metadata_list)

        # Determine mask method based on extension
        maskmethod = self.mask_file.suffix.lower().lstrip('.') if self.mask_file else 'edf'

        print (f"Mask method: {maskmethod}")
        print (f"Mask file: {self.mask_file}")
        print (f"Poni file: {self.poni_file}")
        print (f"Sample orientation: {self.sample_orientation}")
        print (f"Incident angle: {self.incident_angle}")
        print (f"Tilt angle: {self.tilt_angle}")
        print (f"Split pixels: {self.split_pixels}")
        print (f"Output space: {self.output_space}")
        print (f"Polarization: {self.polarization}")
        print (f"Solid angle: {self.solid_angle}")
        print (f"Metadata attributes: {self.metadata_attributes}")

        maskfile = Path(self.mask_file)
        ponifile = Path(self.poni_file)
        # maskfile = Path(self.mask_file) if self.mask_file is not None else None
        # ponifile = Path(self.poni_file) if self.poni_file is not None else None

        ## Adding a popup dialog for integration and dataset generation time.
        # Initialize PyHyperScattering Wrapper for PyFAI FiberIntegrator object
        self.integrator = PFFIGeneralIntegrator(
            geomethod='ponifile',
            ponifile=ponifile,
            maskmethod=maskmethod,
            maskpath=maskfile,
            sample_orientation=int(self.sample_orientation),
            tilt_angle=self.tilt_angle,
            split_pixels=self.split_pixels,
            incident_angle=self.incident_angle,
            output_space=self.output_space
            # Note: polarization and solid_angle_on are not used in the current implementation
            # polarization=self.polarization,
            # solid_angle_on=self.solid_angle_on
        )

        self.fileset = [self.file_path]

        self.util = phs.util.IntegrationUtils.CMSGIWAXS(self.fileset, 
                                            self.loader, 
                                            self.integrator)
        
        raw_DS, recip_DS = self.util.single_images_to_dataset()

        self.recip_DS = recip_DS
        self.raw_DS = raw_DS

        print(recip_DS)

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

def UtilManager():
    """
    This is a placeholder for the UtilManager class.
    It should be implemented to manage utility functions and data.
    """
    pass

