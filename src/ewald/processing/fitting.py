"""Contracts for 2D detector peak fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ewald.processing.peak_detection import PeakCandidate


@dataclass(slots=True)
class FitResult:
    """Result of fitting one peak or a related group of peaks."""

    model_name: str
    center_qx: float
    center_qz: float
    amplitude: float
    width_qx: float | None = None
    width_qz: float | None = None
    statistics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "center_qx": self.center_qx,
            "center_qz": self.center_qz,
            "amplitude": self.amplitude,
            "width_qx": self.width_qx,
            "width_qz": self.width_qz,
            "statistics": self.statistics,
            "metadata": self.metadata,
        }


def fit_peak_set(
    peaks: Iterable[PeakCandidate],
    *,
    model_name: str = "placeholder-centroid",
) -> list[FitResult]:
    """Convert peak candidates into fit-result records.

    The later implementation should replace this with lmfit-backed 2D
    Gaussian, Voigt, or instrument-aware models. Keeping this function small
    makes the surrounding workflow testable before the fitter is complete.
    """

    return [
        FitResult(
            model_name=model_name,
            center_qx=peak.x,
            center_qz=peak.y,
            amplitude=peak.intensity,
            metadata={"source_label": peak.label},
        )
        for peak in peaks
    ]
