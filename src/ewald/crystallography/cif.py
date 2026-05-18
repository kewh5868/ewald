"""Reference CIF metadata and comparison helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ReferenceCIF:
    """A generated or imported CIF tied to experimental evidence."""

    path: Path
    source: str = "generated"
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


def compare_cif_atom_coordinates(
    generated_path: str | Path,
    reference_path: str | Path,
) -> dict[str, Any]:
    """Compare two CIFs with formula, lattice, and atom-coordinate
    metrics."""

    generated = _load_structure_for_comparison(generated_path)
    reference = _load_structure_for_comparison(reference_path)
    coordinate_match = _coordinate_match_summary(generated, reference)
    return {
        "generated": str(generated_path),
        "reference": str(reference_path),
        "generated_summary": _structure_summary(generated),
        "reference_summary": _structure_summary(reference),
        "composition_delta": _composition_delta(generated, reference),
        "lattice_delta": _lattice_delta(generated, reference),
        "coordinate_match": coordinate_match,
    }


def _load_structure_for_comparison(path: str | Path) -> Any:
    try:
        from pymatgen.core import Structure
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pymatgen is required for CIF atom-coordinate comparison."
        ) from exc
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Issues encountered while parsing CIF: .*fractional "
                r"coordinates rounded to ideal values.*"
            ),
            category=UserWarning,
        )
        return Structure.from_file(str(path))


def _structure_summary(structure: Any) -> dict[str, Any]:
    lattice = structure.lattice
    return {
        "formula": str(structure.composition.formula),
        "reduced_formula": str(structure.composition.reduced_formula),
        "site_count": len(structure),
        "lattice": {
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
        },
    }


def _composition_delta(generated: Any, reference: Any) -> dict[str, float]:
    generated_comp = generated.composition.as_dict()
    reference_comp = reference.composition.as_dict()
    elements = sorted(set(generated_comp) | set(reference_comp))
    return {
        element: float(generated_comp.get(element, 0.0))
        - float(reference_comp.get(element, 0.0))
        for element in elements
        if abs(
            float(generated_comp.get(element, 0.0))
            - float(reference_comp.get(element, 0.0))
        )
        > 1.0e-9
    }


def _lattice_delta(generated: Any, reference: Any) -> dict[str, float]:
    generated_lattice = generated.lattice
    reference_lattice = reference.lattice
    return {
        "a": float(generated_lattice.a - reference_lattice.a),
        "b": float(generated_lattice.b - reference_lattice.b),
        "c": float(generated_lattice.c - reference_lattice.c),
        "alpha": float(generated_lattice.alpha - reference_lattice.alpha),
        "beta": float(generated_lattice.beta - reference_lattice.beta),
        "gamma": float(generated_lattice.gamma - reference_lattice.gamma),
        "abc_abs_sum": float(
            abs(generated_lattice.a - reference_lattice.a)
            + abs(generated_lattice.b - reference_lattice.b)
            + abs(generated_lattice.c - reference_lattice.c)
        ),
    }


def _coordinate_match_summary(
    generated: Any,
    reference: Any,
) -> dict[str, Any]:
    generated_species = [site.specie.symbol for site in generated]
    reference_species = [site.specie.symbol for site in reference]
    generated_coords = np.asarray(
        [site.frac_coords for site in generated],
        dtype=float,
    )
    reference_coords = np.asarray(
        [site.frac_coords for site in reference],
        dtype=float,
    )
    reference_lattice = np.asarray(reference.lattice.matrix, dtype=float)
    elements = sorted(set(generated_species) | set(reference_species))
    by_element: dict[str, dict[str, Any]] = {}
    all_fractional_sq: list[float] = []
    all_cartesian_sq: list[float] = []
    for element in elements:
        generated_indices = [
            index
            for index, species in enumerate(generated_species)
            if species == element
        ]
        reference_indices = [
            index
            for index, species in enumerate(reference_species)
            if species == element
        ]
        deltas = _assigned_fractional_deltas(
            generated_coords[generated_indices],
            reference_coords[reference_indices],
        )
        fractional_distances = np.linalg.norm(deltas, axis=1)
        cartesian_distances = np.linalg.norm(
            deltas @ reference_lattice,
            axis=1,
        )
        all_fractional_sq.extend((fractional_distances**2).tolist())
        all_cartesian_sq.extend((cartesian_distances**2).tolist())
        by_element[element] = {
            "generated_count": len(generated_indices),
            "reference_count": len(reference_indices),
            "matched_count": int(len(deltas)),
            "unmatched_count": abs(
                len(generated_indices) - len(reference_indices)
            ),
            "fractional_rms": _rms_or_none(fractional_distances),
            "fractional_max": _max_or_none(fractional_distances),
            "cartesian_rms_angstrom": _rms_or_none(cartesian_distances),
            "cartesian_max_angstrom": _max_or_none(cartesian_distances),
        }
    return {
        "matched_count": int(len(all_fractional_sq)),
        "unmatched_count": int(
            sum(item["unmatched_count"] for item in by_element.values())
        ),
        "fractional_rms": _rms_from_squares_or_none(all_fractional_sq),
        "cartesian_rms_angstrom": _rms_from_squares_or_none(all_cartesian_sq),
        "by_element": by_element,
    }


def _assigned_fractional_deltas(
    generated_coords: np.ndarray,
    reference_coords: np.ndarray,
) -> np.ndarray:
    if generated_coords.size == 0 or reference_coords.size == 0:
        return np.empty((0, 3), dtype=float)
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "scipy is required for CIF atom-coordinate assignment."
        ) from exc
    delta = (
        generated_coords[:, np.newaxis, :] - reference_coords[np.newaxis, :, :]
    )
    delta -= np.round(delta)
    distances = np.linalg.norm(delta, axis=2)
    generated_order, reference_order = linear_sum_assignment(distances)
    return delta[generated_order, reference_order]


def _rms_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values**2)))


def _rms_from_squares_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float))))


def _max_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))
