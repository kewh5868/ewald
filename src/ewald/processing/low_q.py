"""Low-q GIWAXS feature helpers for image-correction review."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, radians, sin, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from ewald.crystallography.cif import suppress_pymatgen_cif_warnings

HC_KEV_ANGSTROM = 12.398419843320026
AVOGADRO_PER_MOL = 6.02214076e23
CLASSICAL_ELECTRON_RADIUS_ANGSTROM = 2.8179403262e-5
ANGSTROM_CUBED_PER_CM_CUBED = 1.0e24
COMPOSITION_ALIASES = {
    "MA": {"C": 1.0, "H": 6.0, "N": 1.0},
    "FA": {"C": 1.0, "H": 5.0, "N": 2.0},
}


@dataclass(slots=True)
class LowQFeature:
    """A q-space guide feature for beamline geometry checks."""

    kind: str
    label: str
    qxy: float | None = None
    qz: float | None = None
    display: str = "point"
    x_px: float | None = None
    y_px: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "qxy": self.qxy,
            "qz": self.qz,
            "display": self.display,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RefractiveIndexEstimate:
    """Film chemistry estimate for the X-ray refractive-index
    decrement."""

    input_formula: str
    normalized_formula: str
    density_g_cm3: float
    xray_energy_kev: float
    wavelength_angstrom: float
    molar_mass_g_mol: float
    electrons_per_formula_unit: float
    electron_density_per_a3: float
    delta: float
    critical_angle_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_formula": self.input_formula,
            "normalized_formula": self.normalized_formula,
            "density_g_cm3": self.density_g_cm3,
            "xray_energy_kev": self.xray_energy_kev,
            "wavelength_angstrom": self.wavelength_angstrom,
            "molar_mass_g_mol": self.molar_mass_g_mol,
            "electrons_per_formula_unit": self.electrons_per_formula_unit,
            "electron_density_per_a3": self.electron_density_per_a3,
            "delta": self.delta,
            "critical_angle_deg": self.critical_angle_deg,
        }


@dataclass(slots=True)
class StructureOpticsEstimate:
    """Film-optics estimate derived from a reference crystal
    structure."""

    structure_path: str
    file_format: str
    formula: str
    normalized_formula: str
    composition: dict[str, float]
    atom_count: int
    unit_cell_volume_a3: float
    density_g_cm3: float
    xray_energy_kev: float
    wavelength_angstrom: float
    molar_mass_g_mol: float
    electrons_per_formula_unit: float
    electron_density_per_a3: float
    delta: float
    critical_angle_deg: float
    refractive_index_real: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_path": self.structure_path,
            "file_format": self.file_format,
            "formula": self.formula,
            "normalized_formula": self.normalized_formula,
            "composition": self.composition,
            "atom_count": self.atom_count,
            "unit_cell_volume_a3": self.unit_cell_volume_a3,
            "density_g_cm3": self.density_g_cm3,
            "xray_energy_kev": self.xray_energy_kev,
            "wavelength_angstrom": self.wavelength_angstrom,
            "molar_mass_g_mol": self.molar_mass_g_mol,
            "electrons_per_formula_unit": self.electrons_per_formula_unit,
            "electron_density_per_a3": self.electron_density_per_a3,
            "delta": self.delta,
            "critical_angle_deg": self.critical_angle_deg,
            "refractive_index_real": self.refractive_index_real,
        }


def wavelength_angstrom_from_energy_kev(energy_kev: float) -> float:
    """Return X-ray wavelength in Angstroms from energy in keV."""

    if energy_kev <= 0:
        raise ValueError("X-ray energy must be positive.")
    return HC_KEV_ANGSTROM / energy_kev


def estimate_refractive_index_delta(
    formula: str,
    density_g_cm3: float,
    xray_energy_kev: float,
) -> RefractiveIndexEstimate:
    """Estimate X-ray refractive delta from composition and mass
    density.

    The estimate uses the non-resonant electron-density approximation,
    ``delta = r_e * lambda^2 * rho_e / (2*pi)``. It is intended as a
    GIWAXS geometry guide for critical-angle and Yoneda markers, not as a
    substitute for tabulated anomalous scattering factors near absorption
    edges.
    """

    if not formula.strip():
        raise ValueError("Enter a stoichiometric formula.")
    if density_g_cm3 <= 0:
        raise ValueError("Sample density must be positive.")
    wavelength = wavelength_angstrom_from_energy_kev(xray_energy_kev)
    composition = _composition_from_formula(formula)
    molar_mass = float(composition.weight)
    electrons = float(
        sum(element.Z * amount for element, amount in composition.items())
    )
    electron_density = (
        density_g_cm3
        * AVOGADRO_PER_MOL
        * electrons
        / molar_mass
        / ANGSTROM_CUBED_PER_CM_CUBED
    )
    delta = (
        CLASSICAL_ELECTRON_RADIUS_ANGSTROM
        * wavelength**2
        * electron_density
        / (2.0 * pi)
    )
    return RefractiveIndexEstimate(
        input_formula=formula,
        normalized_formula=composition.reduced_formula,
        density_g_cm3=float(density_g_cm3),
        xray_energy_kev=float(xray_energy_kev),
        wavelength_angstrom=wavelength,
        molar_mass_g_mol=molar_mass,
        electrons_per_formula_unit=electrons,
        electron_density_per_a3=electron_density,
        delta=float(delta),
        critical_angle_deg=critical_angle_deg_from_delta(float(delta)),
    )


def estimate_refractive_index_from_structure(
    structure_file: str | Path,
    xray_energy_kev: float,
) -> StructureOpticsEstimate:
    """Estimate film-optics constants from a CIF/POSCAR/VASP structure.

    The crystal density comes from pymatgen's parsed structure. The optical
    estimate then uses the same non-resonant electron-density approximation as
    :func:`estimate_refractive_index_delta`, keeping the film-optics workflow
    consistent with manually entered stoichiometry and density.
    """

    path = Path(structure_file)
    file_format = _structure_file_format(path)
    if not path.exists():
        raise ValueError(f"Structure file does not exist: {path}")
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ValueError(
            "Reference-structure optics estimates require pymatgen."
        ) from exc

    try:
        with suppress_pymatgen_cif_warnings():
            structure = Structure.from_file(str(path))
    except Exception as exc:
        raise ValueError(
            f"Could not parse {file_format} structure file {path.name!r}: "
            f"{exc}"
        ) from exc

    if len(structure) == 0:
        raise ValueError(f"Structure file {path.name!r} contains no atoms.")
    density = float(structure.density)
    if density <= 0:
        raise ValueError(
            f"Could not estimate a positive density from {path.name!r}."
        )
    composition = structure.composition.element_composition
    composition_by_symbol = {
        str(element): float(amount)
        for element, amount in sorted(
            composition.items(),
            key=lambda item: item[0].symbol,
        )
    }
    normalized_formula = composition.reduced_formula
    estimate = estimate_refractive_index_delta(
        normalized_formula,
        density,
        xray_energy_kev,
    )
    return StructureOpticsEstimate(
        structure_path=str(path),
        file_format=file_format,
        formula=composition.formula,
        normalized_formula=estimate.normalized_formula,
        composition=composition_by_symbol,
        atom_count=len(structure),
        unit_cell_volume_a3=float(structure.volume),
        density_g_cm3=density,
        xray_energy_kev=estimate.xray_energy_kev,
        wavelength_angstrom=estimate.wavelength_angstrom,
        molar_mass_g_mol=estimate.molar_mass_g_mol,
        electrons_per_formula_unit=estimate.electrons_per_formula_unit,
        electron_density_per_a3=estimate.electron_density_per_a3,
        delta=estimate.delta,
        critical_angle_deg=estimate.critical_angle_deg,
        refractive_index_real=1.0 - estimate.delta,
    )


def critical_angle_deg_from_delta(delta: float) -> float:
    """Return the small-angle X-ray critical angle from refractive
    delta."""

    if delta < 0:
        raise ValueError("Refractive-index delta must be non-negative.")
    return np.degrees(sqrt(2.0 * delta))


def qz_for_exit_angle(
    incident_angle_deg: float,
    exit_angle_deg: float,
    wavelength_angstrom: float,
) -> float:
    """Approximate GI q_{z} for one exit angle in the sample frame."""

    k = 2.0 * pi / wavelength_angstrom
    return k * (
        sin(radians(incident_angle_deg)) + sin(radians(exit_angle_deg))
    )


def critical_q_a_inv(
    critical_angle_deg: float,
    wavelength_angstrom: float,
) -> float:
    """Return critical q for specular reflection geometry."""

    return 4.0 * pi / wavelength_angstrom * sin(radians(critical_angle_deg))


def build_low_q_features(
    *,
    incident_angle_deg: float,
    critical_angle_deg: float | None,
    xray_energy_kev: float,
    reflected_beam_x_px: float | None = None,
    reflected_beam_y_px: float | None = None,
) -> list[LowQFeature]:
    """Build q-space guide features used during GI image correction."""

    wavelength = wavelength_angstrom_from_energy_kev(xray_energy_kev)
    base_metadata = {
        "geometry": "grazing-incidence",
        "incident_angle_deg": incident_angle_deg,
        "xray_energy_kev": xray_energy_kev,
        "wavelength_angstrom": wavelength,
    }
    horizon_qz = qz_for_exit_angle(
        incident_angle_deg,
        0.0,
        wavelength,
    )
    reflected_qz = qz_for_exit_angle(
        incident_angle_deg,
        incident_angle_deg,
        wavelength,
    )
    features = [
        LowQFeature(
            kind="direct_beam",
            label="Primary direct beam (q = 0)",
            qxy=0.0,
            qz=0.0,
            display="point",
            metadata={
                **base_metadata,
                "role": "unscattered incident beam reference",
            },
        ),
        LowQFeature(
            kind="sample_horizon",
            label="Sample horizon / alpha_f = 0",
            qz=horizon_qz,
            display="horizontal_line",
            metadata={
                **base_metadata,
                "exit_angle_deg": 0.0,
                "formula": "q_{z} = k(sin(alpha_i) + sin(alpha_f))",
                "role": "sample plane horizon",
            },
        ),
        LowQFeature(
            kind="reflected_beam",
            label="Specular reflected beam / alpha_f = alpha_i",
            qxy=0.0,
            qz=reflected_qz,
            display="point",
            x_px=reflected_beam_x_px,
            y_px=reflected_beam_y_px,
            metadata={
                **base_metadata,
                "exit_angle_deg": incident_angle_deg,
                "role": "specular reflection position",
            },
        ),
        LowQFeature(
            kind="effective_beam_center",
            label="Low-q cut center between direct beam and horizon",
            qxy=0.0,
            qz=0.5 * horizon_qz,
            display="point",
            metadata={
                **base_metadata,
                "model": "midpoint between primary beam and sample horizon",
                "role": "low-q exclusion center estimate",
            },
        ),
    ]
    if critical_angle_deg is not None and critical_angle_deg > 0:
        features.extend(
            [
                LowQFeature(
                    kind="critical_q",
                    label="Critical edge q_c",
                    qz=critical_q_a_inv(critical_angle_deg, wavelength),
                    display="horizontal_line",
                    metadata={
                        **base_metadata,
                        "critical_angle_deg": critical_angle_deg,
                        "formula": "q_c = 4*pi*sin(alpha_c)/lambda",
                        "role": "film critical-angle edge",
                    },
                ),
                LowQFeature(
                    kind="yoneda_band",
                    label="Yoneda band / alpha_f = alpha_c",
                    qz=qz_for_exit_angle(
                        incident_angle_deg,
                        critical_angle_deg,
                        wavelength,
                    ),
                    display="horizontal_line",
                    metadata={
                        **base_metadata,
                        "exit_angle_deg": critical_angle_deg,
                        "critical_angle_deg": critical_angle_deg,
                        "role": "Yoneda-enhanced diffuse scattering line",
                    },
                ),
            ]
        )
    return features


def estimate_bright_spot_centroid(
    image: Any,
    *,
    percentile: float = 99.8,
    rotation_deg: int = 0,
    mirrored_y: bool = False,
) -> tuple[float, float] | None:
    """Estimate a bright low-q beam spot centroid from a detector
    image."""

    array = np.asarray(image, dtype=float)
    if array.ndim > 2:
        array = array[0]
    array = _apply_orientation(array, rotation_deg, mirrored_y)
    finite = np.isfinite(array)
    if not finite.any():
        return None
    cutoff = np.nanpercentile(array[finite], percentile)
    mask = finite & (array >= cutoff)
    if not mask.any():
        return None
    yy, xx = np.nonzero(mask)
    weights = array[yy, xx] - np.nanmin(array[finite])
    if not np.isfinite(weights).any() or float(np.sum(weights)) <= 0:
        return float(np.mean(xx)), float(np.mean(yy))
    return (
        float(np.average(xx, weights=weights)),
        float(np.average(yy, weights=weights)),
    )


def _apply_orientation(
    image: np.ndarray,
    rotation_deg: int,
    mirrored_y: bool,
) -> np.ndarray:
    turns = (rotation_deg // 90) % 4
    oriented = np.rot90(image, k=-turns) if turns else image
    if mirrored_y:
        oriented = np.fliplr(oriented)
    return oriented


def _structure_file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".cif", ".mcif"}:
        return "CIF"
    if suffix == ".vasp":
        return "VASP"
    if suffix in {".poscar", ".contcar"}:
        return "POSCAR"
    if name.startswith("poscar") or name.startswith("contcar"):
        return "POSCAR"
    raise ValueError(
        "Unsupported structure file type. Choose a CIF (.cif/.mcif), "
        "POSCAR/CONTCAR, or VASP (.vasp) structure file."
    )


def _composition_from_formula(formula: str):
    from pymatgen.core import Composition

    try:
        return Composition(formula)
    except Exception:
        return Composition(_normalize_alias_formula(formula))


def _normalize_alias_formula(formula: str) -> str:
    """Normalize simple beamline/sample labels into an elemental
    formula."""

    from pymatgen.core import Element

    value = formula.strip().replace(" ", "").replace("_", "").replace("-", "")
    if not value:
        raise ValueError("Enter a stoichiometric formula.")
    totals: dict[str, float] = {}
    token_pattern = (
        r"(?P<prefix>\d+(?:\.\d+)?)?"
        r"(?P<symbol>MA|FA|[A-Z][a-z]?)"
        r"(?P<suffix>\d*(?:\.\d+)?)"
    )
    import re

    position = 0
    for match in re.finditer(token_pattern, value):
        if match.start() != position:
            raise ValueError(
                f"Could not parse formula near {value[position:]!r}."
            )
        symbol = match.group("symbol")
        count_text = match.group("suffix") or match.group("prefix") or "1"
        count = float(count_text)
        if symbol in COMPOSITION_ALIASES:
            for element, amount in COMPOSITION_ALIASES[symbol].items():
                totals[element] = totals.get(element, 0.0) + amount * count
        else:
            Element(symbol)
            totals[symbol] = totals.get(symbol, 0.0) + count
        position = match.end()
    if position != len(value) or not totals:
        raise ValueError(f"Could not parse formula {formula!r}.")
    return "".join(
        f"{element}{_formula_count_text(count)}"
        for element, count in sorted(totals.items())
    )


def _formula_count_text(count: float) -> str:
    if float(count).is_integer():
        return str(int(count))
    return f"{count:g}"
