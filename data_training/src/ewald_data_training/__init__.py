"""Training-data utilities for EWALD structure recognition.

This package is intentionally isolated from the GUI/backend runtime. The
modules here can be used locally, on a SLURM node, or inside a future
training container without importing Qt.
"""

from .schemas import (
    ArtifactProfile,
    DatasetSample,
    DetectorGeometry,
    SimulationCondition,
    StructureRecord,
)

__all__ = [
    "ArtifactProfile",
    "DatasetSample",
    "DetectorGeometry",
    "SimulationCondition",
    "StructureRecord",
]
