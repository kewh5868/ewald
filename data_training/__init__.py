"""Configuration-first scaffolding for EWALD training data
generation."""

from .artifacts import ArtifactOperation, ArtifactProfile
from .catalog import StructureCatalog, StructureCatalogRecord
from .conditions import SimulationCondition, SimulationSweepSpec, SweepAxis
from .manifest import DatasetImageRecord, DatasetManifest, PeakLabel
from .runtime import ClusterRuntimeConfig

__all__ = [
    "ArtifactOperation",
    "ArtifactProfile",
    "ClusterRuntimeConfig",
    "DatasetImageRecord",
    "DatasetManifest",
    "PeakLabel",
    "SimulationCondition",
    "SimulationSweepSpec",
    "StructureCatalog",
    "StructureCatalogRecord",
    "SweepAxis",
]
