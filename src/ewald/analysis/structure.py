"""Structure-analysis geometry, fitting, and candidate helpers."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from itertools import combinations_with_replacement, product
from math import comb, gcd
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

from ewald.crystallography.lattice import Lattice
from ewald.crystallography.overlay import (
    CRYSTAL_SYSTEMS,
    CrystalOverlayParameters,
    apply_crystal_system_constraints,
    normalize_quaternion,
    quaternion_from_axis_angle,
    rotate_vectors_by_quaternion,
)
from ewald.data.models import (
    PEAK_HKL_METADATA_KEY,
    PEAK_PHASE_METADATA_KEY,
    PEAK_POINT_KIND_GAP_ESTIMATED,
)

DEFAULT_PHASE_TAG = "Phase 1 / main phase"
PHASE_SECONDARY = "Phase 2 / secondary phase"
PHASE_UNASSIGNED = "unassigned"
PHASE_GAP_ESTIMATED = "gap-estimated"
PHASE_FORBIDDEN = "forbidden reflection"
PHASE_REJECTED = "rejected/excluded from fit"
DEFAULT_PHASE_TAGS = [
    DEFAULT_PHASE_TAG,
    PHASE_SECONDARY,
    PHASE_UNASSIGNED,
    PHASE_GAP_ESTIMATED,
    PHASE_FORBIDDEN,
    PHASE_REJECTED,
]
FAMILY_KIND_SIMILAR_QXY = "similar q_xy"
FAMILY_KIND_SIMILAR_QZ = "similar q_z"
FAMILY_KIND_QXY_MULTIPLES = "q_xy multiples"
FAMILY_KIND_QZ_MULTIPLES = "q_z multiples"
DEFAULT_FAMILY_DETECTION_KINDS = (
    FAMILY_KIND_SIMILAR_QXY,
    FAMILY_KIND_SIMILAR_QZ,
    FAMILY_KIND_QXY_MULTIPLES,
    FAMILY_KIND_QZ_MULTIPLES,
)
SOURCE_ROI_FIT = "ROI fit"
SOURCE_USER_EDIT = "user edit"
SOURCE_GAP_ESTIMATE = "gap estimate"
SOURCE_IMPORTED_POINT = "imported point"
SOURCE_MARKED_POINT = "marked point"
REFERENCE_MOLECULES = {
    "DMF": {
        "formula": "C3H7NO",
        "name": "N,N-dimethylformamide",
        "source": "built-in",
        "atoms": (
            ("O", 0.00, 1.18, 0.00),
            ("C", 0.00, 0.00, 0.00),
            ("H", 0.92, -0.55, 0.00),
            ("N", -1.20, -0.72, 0.00),
            ("C", -2.42, 0.08, 0.00),
            ("H", -2.33, 0.72, 0.89),
            ("H", -2.51, 0.72, -0.89),
            ("H", -3.30, -0.57, 0.00),
            ("C", -1.32, -2.17, 0.00),
            ("H", -0.35, -2.68, 0.00),
            ("H", -1.85, -2.47, 0.89),
            ("H", -1.85, -2.47, -0.89),
        ),
    },
    "DMSO": {
        "formula": "C2H6OS",
        "name": "dimethyl sulfoxide",
        "source": "built-in",
        "atoms": (
            ("S", 0.00, 0.00, 0.00),
            ("O", 0.00, 1.48, 0.00),
            ("C", 1.42, -0.72, 0.00),
            ("H", 1.26, -1.81, 0.00),
            ("H", 2.03, -0.42, 0.86),
            ("H", 2.03, -0.42, -0.86),
            ("C", -1.42, -0.72, 0.00),
            ("H", -1.26, -1.81, 0.00),
            ("H", -2.03, -0.42, 0.86),
            ("H", -2.03, -0.42, -0.86),
        ),
    },
    "MA": {
        "formula": "CH6N",
        "name": "methylammonium",
        "source": "built-in",
        "atoms": (
            ("C", -0.72, 0.00, 0.00),
            ("N", 0.75, 0.00, 0.00),
            ("H", -1.09, 1.02, 0.00),
            ("H", -1.09, -0.51, 0.88),
            ("H", -1.09, -0.51, -0.88),
            ("H", 1.10, 0.94, 0.00),
            ("H", 1.10, -0.47, 0.81),
            ("H", 1.10, -0.47, -0.81),
        ),
    },
    "FA": {
        "formula": "CH5N2",
        "name": "formamidinium",
        "source": "built-in",
        "atoms": (
            ("C", 0.00, 0.00, 0.00),
            ("H", 0.00, 1.08, 0.00),
            ("N", -1.25, -0.40, 0.00),
            ("H", -1.55, -1.35, 0.00),
            ("H", -2.00, 0.25, 0.00),
            ("N", 1.25, -0.40, 0.00),
            ("H", 1.55, -1.35, 0.00),
            ("H", 2.00, 0.25, 0.00),
        ),
    },
    "BA": {
        "formula": "C4H12N",
        "name": "butylammonium",
        "source": "built-in",
        "atoms": (
            ("N", -2.92, 0.00, 0.00),
            ("H", -3.25, 0.94, 0.00),
            ("H", -3.25, -0.47, 0.81),
            ("H", -3.25, -0.47, -0.81),
            ("C", -1.48, 0.00, 0.00),
            ("H", -1.20, 0.54, 0.89),
            ("H", -1.20, 0.54, -0.89),
            ("C", 0.00, 0.00, 0.00),
            ("H", 0.25, -0.58, 0.89),
            ("H", 0.25, -0.58, -0.89),
            ("C", 1.48, 0.00, 0.00),
            ("H", 1.74, 0.58, 0.89),
            ("H", 1.74, 0.58, -0.89),
            ("C", 2.92, 0.00, 0.00),
            ("H", 3.32, -1.02, 0.00),
            ("H", 3.32, 0.51, 0.88),
            ("H", 3.32, 0.51, -0.88),
        ),
    },
    "NMP": {
        "formula": "C5H9NO",
        "name": "N-methyl-2-pyrrolidone",
        "source": "built-in",
        "atoms": (
            ("N", -0.95, 0.10, 0.00),
            ("C", 0.25, 0.92, 0.00),
            ("O", 0.24, 2.14, 0.00),
            ("C", 1.47, 0.00, 0.00),
            ("H", 2.42, 0.55, 0.00),
            ("H", 1.48, -0.63, 0.90),
            ("C", 0.93, -1.34, 0.00),
            ("H", 1.45, -1.91, 0.80),
            ("H", 1.45, -1.91, -0.80),
            ("C", -0.58, -1.28, 0.00),
            ("H", -1.03, -1.78, 0.86),
            ("H", -1.03, -1.78, -0.86),
            ("C", -2.22, 0.58, 0.00),
            ("H", -2.08, 1.68, 0.00),
            ("H", -2.84, 0.25, 0.86),
            ("H", -2.84, 0.25, -0.86),
        ),
    },
}
MOLECULE_POSE_RESTRAINTS: dict[str, dict[str, Any]] = {
    "MA": {
        "pose_role": "hydrogen_bond_donor",
        "donor_elements": ("N",),
        "acceptor_elements": (),
        "rigid_body": True,
    },
    "FA": {
        "pose_role": "hydrogen_bond_donor",
        "donor_elements": ("N",),
        "acceptor_elements": (),
        "planar_heavy_atom_restraint": True,
        "rigid_body": True,
    },
    "BA": {
        "pose_role": "hydrogen_bond_donor",
        "donor_elements": ("N",),
        "acceptor_elements": (),
        "rigid_body": True,
    },
    "DMF": {
        "pose_role": "hydrogen_bond_acceptor",
        "donor_elements": (),
        "acceptor_elements": ("O",),
        "planar_heavy_atom_restraint": True,
        "rigid_body": True,
    },
    "DMSO": {
        "pose_role": "hydrogen_bond_acceptor",
        "donor_elements": (),
        "acceptor_elements": ("O",),
        "rigid_body": True,
    },
    "NMP": {
        "pose_role": "hydrogen_bond_acceptor",
        "donor_elements": (),
        "acceptor_elements": ("O",),
        "planar_heavy_atom_restraint": True,
        "rigid_body": True,
    },
}
CRYSTAL_SYSTEM_SPACE_GROUP_RANGES = {
    "Triclinic": range(1, 3),
    "Monoclinic": range(3, 16),
    "Orthorhombic": range(16, 75),
    "Tetragonal": range(75, 143),
    "Trigonal": range(143, 168),
    "Hexagonal": range(168, 195),
    "Cubic": range(195, 231),
}
SIMPLE_CRYSTAL_SYSTEM_ORDER = (
    "Cubic",
    "Tetragonal",
    "Hexagonal",
    "Trigonal",
    "Orthorhombic",
    "Monoclinic",
    "Triclinic",
)
CRYSTAL_SYSTEM_COMPLEXITY = {
    system: rank for rank, system in enumerate(SIMPLE_CRYSTAL_SYSTEM_ORDER)
}
WYCKOFF_CRYSTAL_SYSTEMS = tuple(CRYSTAL_SYSTEM_SPACE_GROUP_RANGES)
WYCKOFF_COMBINATION_DISPLAY_LIMIT = 1000
PROJECTION_RADIAL = "radial"
PROJECTION_CARTESIAN_XZ = "cartesian_xz"
PROJECTION_FIBER_QXY_QZ = "fiber_qxy_qz"
DEFAULT_SPACE_GROUP_BY_SYSTEM = {
    "Triclinic": 2,
    "Monoclinic": 14,
    "Orthorhombic": 58,
    "Tetragonal": 123,
    "Trigonal": 166,
    "Hexagonal": 194,
    "Cubic": 221,
}
MA2_DMF2_PB3I8_TEMPLATE_ROWS: tuple[
    tuple[str, str, float, float, float], ...
] = (
    ("Pb1", "Pb", 0.70752, 0.60227, 1.0000),
    ("Pb2", "Pb", 0.5000, 0.5000, 0.5000),
    ("I1", "I", 0.83085, 0.56609, 0.5000),
    ("I2", "I", 0.74422, 0.73737, 1.0000),
    ("I3", "I", 0.57686, 0.63381, 0.5000),
    ("I4", "I", 0.37774, 0.54045, 0.0000),
    ("N1", "N", 0.4061, 0.7773, 0.0000),
    ("H1A", "H", 0.3560, 0.7894, 0.0000),
    ("H1B", "H", 0.4298, 0.7923, 0.1608),
    ("C1", "C", 0.4102, 0.7081, 0.0000),
    ("H1C", "H", 0.4648, 0.6951, 0.0000),
    ("H1D", "H", 0.3843, 0.6919, 0.1753),
    ("O1", "O", 0.4989, 0.8098, 0.5000),
    ("N2", "N", 0.5963, 0.8777, 0.5000),
    ("C2", "C", 0.5688, 0.8206, 0.5000),
    ("H2", "H", 0.6045, 0.7876, 0.5000),
    ("C3", "C", 0.6794, 0.8895, 0.5000),
    ("H3A", "H", 0.6899, 0.9334, 0.5000),
    ("H3B", "H", 0.7034, 0.8711, 0.3261),
    ("C4", "C", 0.5450, 0.9302, 0.5000),
    ("H4A", "H", 0.5747, 0.9683, 0.5000),
    ("H4B", "H", 0.5115, 0.9287, 0.3259),
)
PNNM_SYMMETRY_OPERATIONS: tuple[
    tuple[tuple[int, int, int], tuple[float, float, float]], ...
] = (
    ((1, 1, 1), (0.0, 0.0, 0.0)),
    ((-1, -1, 1), (0.0, 0.0, 0.0)),
    ((-1, 1, -1), (0.5, 0.5, 0.5)),
    ((1, -1, -1), (0.5, 0.5, 0.5)),
    ((-1, -1, -1), (0.0, 0.0, 0.0)),
    ((1, 1, -1), (0.0, 0.0, 0.0)),
    ((1, -1, 1), (-0.5, -0.5, -0.5)),
    ((-1, 1, 1), (-0.5, -0.5, -0.5)),
)


@dataclass(slots=True)
class StructurePeak:
    """One peak position used by Structure Analysis."""

    peak_id: str
    label: str
    qxy: float
    qz: float
    source: str = SOURCE_MARKED_POINT
    phase_tag: str = DEFAULT_PHASE_TAG
    hkl_label: str = ""
    include: bool = True
    fit_quality: float | None = None
    status: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def q_magnitude(self) -> float:
        return float(np.hypot(self.qxy, self.qz))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructurePeak":
        return cls(
            peak_id=str(payload.get("peak_id") or payload.get("id") or ""),
            label=str(payload.get("label") or payload.get("peak_id") or ""),
            qxy=float(payload.get("qxy", payload.get("qx", 0.0))),
            qz=float(payload.get("qz", 0.0)),
            source=str(payload.get("source", SOURCE_MARKED_POINT)),
            phase_tag=str(payload.get("phase_tag", DEFAULT_PHASE_TAG)),
            hkl_label=str(payload.get("hkl_label", "")),
            include=bool(payload.get("include", True)),
            fit_quality=_optional_float(payload.get("fit_quality")),
            status=str(payload.get("status", "")),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class CandidateSearchConfig:
    """Configuration shared by lattice refinement and candidate
    guessing."""

    crystal_systems: tuple[str, ...] = SIMPLE_CRYSTAL_SYSTEM_ORDER
    hkl_max: int = 4
    q_tolerance: float = 0.06
    relative_tolerance: float = 0.035
    lattice_min: float = 2.5
    lattice_max: float = 35.0
    grid_points: int = 16
    max_candidates: int = 12
    phase_tag: str = DEFAULT_PHASE_TAG
    orientation_quaternion: tuple[float, float, float, float] | None = None
    enable_projected_axis_search: bool = True


@dataclass(slots=True)
class LatticeCandidate:
    """A scored lattice/crystal-system candidate."""

    candidate_id: str
    crystal_system: str
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    score: float = float("inf")
    rms_error: float = float("inf")
    matched_count: int = 0
    outlier_count: int = 0
    method: str = "grid"
    assignments: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    orientation_quaternion: tuple[float, float, float, float] | None = None
    projection_mode: str = PROJECTION_RADIAL

    def constrained(self) -> "LatticeCandidate":
        values = self.as_parameters().as_dict()
        apply_crystal_system_constraints(values)
        orientation = _optional_orientation_quaternion(
            self.orientation_quaternion
        )
        return LatticeCandidate(
            candidate_id=self.candidate_id,
            crystal_system=str(values["crystal_system"]),
            a=float(values["a"]),
            b=float(values["b"]),
            c=float(values["c"]),
            alpha=float(values["alpha"]),
            beta=float(values["beta"]),
            gamma=float(values["gamma"]),
            score=self.score,
            rms_error=self.rms_error,
            matched_count=self.matched_count,
            outlier_count=self.outlier_count,
            method=self.method,
            assignments=list(self.assignments),
            notes=self.notes,
            orientation_quaternion=orientation,
            projection_mode=self.projection_mode,
        )

    def as_parameters(self) -> CrystalOverlayParameters:
        return CrystalOverlayParameters(
            crystal_system=self.crystal_system,
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            h_max=4,
            k_max=4,
            l_max=4,
            orientation_quaternion=_orientation_quaternion_or_identity(
                self.orientation_quaternion
            ),
            positive_qz_only=True,
        ).constrained()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameters"] = self.as_parameters().as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LatticeCandidate":
        params = payload.get("parameters", payload)
        orientation_payload = (
            payload.get("orientation_quaternion")
            if "orientation_quaternion" in payload
            else params.get("orientation_quaternion")
        )
        return cls(
            candidate_id=str(payload.get("candidate_id", "candidate")),
            crystal_system=str(params.get("crystal_system", "Cubic")),
            a=float(params.get("a", payload.get("a", 6.0))),
            b=float(params.get("b", payload.get("b", params.get("a", 6.0)))),
            c=float(params.get("c", payload.get("c", params.get("a", 6.0)))),
            alpha=float(params.get("alpha", payload.get("alpha", 90.0))),
            beta=float(params.get("beta", payload.get("beta", 90.0))),
            gamma=float(params.get("gamma", payload.get("gamma", 90.0))),
            score=float(payload.get("score", float("inf"))),
            rms_error=float(payload.get("rms_error", float("inf"))),
            matched_count=int(payload.get("matched_count", 0)),
            outlier_count=int(payload.get("outlier_count", 0)),
            method=str(payload.get("method", "grid")),
            assignments=list(payload.get("assignments", [])),
            notes=str(payload.get("notes", "")),
            orientation_quaternion=_optional_orientation_quaternion(
                orientation_payload
            ),
            projection_mode=str(
                payload.get("projection_mode", PROJECTION_RADIAL)
            ),
        )


@dataclass(frozen=True, slots=True)
class WyckoffSiteOption:
    """One registered Wyckoff site possibility for a space group."""

    space_group_number: int
    space_group_symbol: str
    crystal_system: str
    letter: str
    multiplicity: int
    parameter_count: int = 0

    @property
    def site_label(self) -> str:
        return f"{self.multiplicity}{self.letter}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "space_group_number": self.space_group_number,
            "space_group_symbol": self.space_group_symbol,
            "crystal_system": self.crystal_system,
            "letter": self.letter,
            "multiplicity": self.multiplicity,
            "parameter_count": self.parameter_count,
            "site_label": self.site_label,
        }


@dataclass(frozen=True, slots=True)
class WyckoffSpaceGroupOption:
    """Registered Wyckoff possibilities for one space group."""

    number: int
    symbol: str
    full_symbol: str
    crystal_system: str
    sites: tuple[WyckoffSiteOption, ...]

    def as_dict(self, *, include_sites: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "number": self.number,
            "symbol": self.symbol,
            "full_symbol": self.full_symbol,
            "crystal_system": self.crystal_system,
            "site_count": len(self.sites),
        }
        if include_sites:
            payload["sites"] = [site.as_dict() for site in self.sites]
        return payload


def build_structure_peaks(
    peak_records: Iterable[dict[str, Any]],
    fit_records: dict[str, Any] | None = None,
    existing: Iterable[dict[str, Any]] | None = None,
) -> list[StructurePeak]:
    """Merge Peak Identification records into Structure Analysis peaks.

    Structure Analysis owns user edits. Existing edited peak centers are
    preserved, while unedited rows are refreshed from ROI fits when
    available.
    """

    fit_records = fit_records or {}
    existing_by_id = {
        StructurePeak.from_dict(item).peak_id: StructurePeak.from_dict(item)
        for item in existing or []
        if isinstance(item, dict)
    }
    peaks: list[StructurePeak] = []
    for record in peak_records:
        peak_id = _peak_id(record)
        previous = existing_by_id.get(peak_id)
        store = fit_records.get(peak_id, {})
        if not isinstance(store, dict):
            store = {}
        imported = _structure_peak_from_record(record, store)
        if previous is not None:
            imported.phase_tag = previous.phase_tag or imported.phase_tag
            imported.hkl_label = previous.hkl_label or imported.hkl_label
            imported.include = previous.include
            imported.notes = previous.notes
            imported.metadata.update(previous.metadata)
            if previous.metadata.get("user_edited_center"):
                imported.qxy = previous.qxy
                imported.qz = previous.qz
                imported.source = SOURCE_USER_EDIT
        peaks.append(imported)
    return peaks


def guess_lattice_candidates(
    peaks: Iterable[StructurePeak | dict[str, Any]],
    config: CandidateSearchConfig | None = None,
) -> list[LatticeCandidate]:
    """Generate and rank plausible lattice candidates."""

    config = config or CandidateSearchConfig()
    active = _active_peaks(peaks, config.phase_tag)
    if not active:
        return []
    candidates: list[LatticeCandidate] = []
    if config.enable_projected_axis_search:
        candidates.extend(_projected_axis_candidates(active, config))
    for system in _ordered_crystal_systems(config.crystal_systems):
        if system not in CRYSTAL_SYSTEMS:
            continue
        candidates.extend(_grid_candidates(active, system, config))
        if system == "Cubic":
            candidates.extend(_cubic_indexing_candidates(active, config))
    ranked = _rank_unique_candidates(candidates, active, config)
    refined = [
        refine_lattice_candidate(active, candidate, config)
        for candidate in ranked[: config.max_candidates]
    ]
    return sorted(
        refined,
        key=lambda item: _candidate_preference_key(item, config),
    )[: config.max_candidates]


def refine_lattice_candidate(
    peaks: Iterable[StructurePeak | dict[str, Any]],
    initial: LatticeCandidate | CrystalOverlayParameters | dict[str, Any],
    config: CandidateSearchConfig | None = None,
) -> LatticeCandidate:
    """Refine lattice constants against active observed peak
    positions."""

    config = config or CandidateSearchConfig()
    active = _active_peaks(peaks, config.phase_tag)
    candidate = _candidate_from_initial(initial)
    if (
        candidate.orientation_quaternion is None
        and config.orientation_quaternion is not None
    ):
        candidate.orientation_quaternion = _optional_orientation_quaternion(
            config.orientation_quaternion
        )
    if not active:
        return _score_candidate(candidate, [], config, method="least_squares")

    variables, bounds, unpack = _parameterization(candidate, config)
    initial_variables = np.asarray(variables, dtype=float)

    def residuals(values: np.ndarray) -> np.ndarray:
        trial = unpack(values)
        assignments = _assign_peaks_to_candidate(active, trial, config)
        residual_values = []
        for assignment in assignments:
            sigma = max(float(assignment["tolerance"]), 1.0e-9)
            residual = float(assignment["delta_q"]) / sigma
            residual_values.append(residual)
        if candidate.projection_mode == PROJECTION_FIBER_QXY_QZ:
            for value, initial_value in zip(values, initial_variables):
                sigma = max(abs(float(initial_value)) * 0.08, 0.2)
                residual_values.append(
                    (float(value) - float(initial_value)) / sigma
                )
        if not residual_values:
            return np.array([1.0e6])
        return np.asarray(residual_values, dtype=float)

    try:
        result = least_squares(
            residuals,
            np.asarray(variables, dtype=float),
            bounds=bounds,
            loss="soft_l1",
            max_nfev=250,
        )
        candidate = unpack(result.x)
        method = "least_squares"
    except Exception as exc:
        method = "least_squares-fallback"
        candidate.notes = f"Refinement fallback: {exc}"
    return _score_candidate(candidate, active, config, method=method)


def suggest_non_main_phase_peaks(
    peaks: Iterable[StructurePeak | dict[str, Any]],
    candidate: LatticeCandidate | dict[str, Any],
    config: CandidateSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """Return peaks that do not agree with the current main-phase
    lattice."""

    config = config or CandidateSearchConfig()
    active = _active_peaks(peaks, config.phase_tag)
    candidate_obj = (
        LatticeCandidate.from_dict(candidate)
        if isinstance(candidate, dict)
        else candidate
    )
    assignments = _assign_peaks_to_candidate(active, candidate_obj, config)
    suggestions = []
    for assignment in assignments:
        if float(assignment["abs_delta_q"]) <= float(assignment["tolerance"]):
            continue
        suggestions.append(
            {
                "peak_id": assignment["peak_id"],
                "label": assignment["label"],
                "phase_suggestion": PHASE_UNASSIGNED,
                "reason": "outside main-phase q tolerance",
                "delta_q": assignment["delta_q"],
                "nearest_hkl": assignment["hkl"],
            }
        )
    return suggestions


def group_peak_families(
    peaks: Iterable[StructurePeak | dict[str, Any]],
    *,
    tolerance: float = 0.04,
    ratio_tolerance: float = 0.06,
    phase_tag: str = DEFAULT_PHASE_TAG,
    enabled_kinds: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Suggest candidate plane families by similar coordinates and
    ratios."""

    active = _active_peaks(peaks, phase_tag)
    kinds = (
        set(DEFAULT_FAMILY_DETECTION_KINDS)
        if enabled_kinds is None
        else {str(kind) for kind in enabled_kinds}
    )
    families: list[dict[str, Any]] = []
    if FAMILY_KIND_SIMILAR_QXY in kinds:
        families.extend(
            _coordinate_families(
                active,
                key="qxy",
                label=FAMILY_KIND_SIMILAR_QXY,
                tolerance=tolerance,
            )
        )
    if FAMILY_KIND_SIMILAR_QZ in kinds:
        families.extend(
            _coordinate_families(
                active,
                key="qz",
                label=FAMILY_KIND_SIMILAR_QZ,
                tolerance=tolerance,
            )
        )
    if FAMILY_KIND_QXY_MULTIPLES in kinds:
        families.extend(
            _multiple_families(
                active,
                key="qxy",
                label=FAMILY_KIND_QXY_MULTIPLES,
                tolerance=ratio_tolerance,
            )
        )
    if FAMILY_KIND_QZ_MULTIPLES in kinds:
        families.extend(
            _multiple_families(
                active,
                key="qz",
                label=FAMILY_KIND_QZ_MULTIPLES,
                tolerance=ratio_tolerance,
            )
        )
    for index, family in enumerate(families, start=1):
        family["family_id"] = f"family_{index:03d}"
        family["phase_tag"] = phase_tag
    return families


def crystal_system_for_space_group(space_group_number: int) -> str:
    """Return the conventional crystal system for an IT space-group
    number."""

    number = int(space_group_number)
    for crystal_system, numbers in CRYSTAL_SYSTEM_SPACE_GROUP_RANGES.items():
        if number in numbers:
            return crystal_system
    raise ValueError(f"Space-group number must be in 1..230, got {number!r}.")


@lru_cache(maxsize=230)
def wyckoff_space_group_option(
    space_group_number: int,
) -> WyckoffSpaceGroupOption:
    """Return all registered Wyckoff site options for one space
    group."""

    number = int(space_group_number)
    crystal_system = crystal_system_for_space_group(number)
    multiplicities, parameter_counts = _load_wyckoff_position_data()
    sites_for_group = multiplicities.get(str(number))
    if not sites_for_group:
        raise RuntimeError(
            "Pymatgen Wyckoff-position data is unavailable or incomplete."
        )
    params_for_group = parameter_counts.get(str(number), {})
    symbol, full_symbol = _space_group_symbols(number)
    sites = tuple(
        sorted(
            (
                WyckoffSiteOption(
                    space_group_number=number,
                    space_group_symbol=symbol,
                    crystal_system=crystal_system,
                    letter=str(letter),
                    multiplicity=int(multiplicity),
                    parameter_count=int(params_for_group.get(letter, 0)),
                )
                for letter, multiplicity in sites_for_group.items()
            ),
            key=lambda item: (item.multiplicity, item.letter),
        )
    )
    return WyckoffSpaceGroupOption(
        number=number,
        symbol=symbol,
        full_symbol=full_symbol,
        crystal_system=crystal_system,
        sites=sites,
    )


def wyckoff_space_group_options(
    crystal_system: str | None = None,
) -> list[WyckoffSpaceGroupOption]:
    """Return registered space-group/Wyckoff possibilities."""

    if crystal_system:
        system = _normalize_crystal_system_name(crystal_system)
        numbers: Iterable[int] = CRYSTAL_SYSTEM_SPACE_GROUP_RANGES[system]
    else:
        numbers = range(1, 231)
    return [wyckoff_space_group_option(number) for number in numbers]


def wyckoff_site_options(
    space_group_number: int,
) -> tuple[WyckoffSiteOption, ...]:
    """Return registered Wyckoff sites for one space group."""

    return wyckoff_space_group_option(space_group_number).sites


def wyckoff_combination_count(
    space_group_number: int,
    *,
    site_count: int,
    ordered: bool = False,
) -> int:
    """Return the number of Wyckoff-site combinations."""

    site_total = len(wyckoff_site_options(space_group_number))
    count = max(1, int(site_count))
    if ordered:
        return site_total**count
    return comb(site_total + count - 1, count)


def wyckoff_site_combinations(
    space_group_number: int,
    *,
    site_count: int = 1,
    max_combinations: int | None = None,
    max_multiplicity_total: int | None = None,
    ordered: bool = False,
) -> list[dict[str, Any]]:
    """Generate registered Wyckoff-site combination possibilities.

    By default, combinations are unordered multisets of Wyckoff site classes.
    Set ``ordered=True`` when assigning an ordered atom/molecule basis, where
    ``Pb@1a, I@3c`` and ``Pb@3c, I@1a`` are distinct possibilities.
    """

    option = wyckoff_space_group_option(space_group_number)
    count = max(1, int(site_count))
    combinations: list[dict[str, Any]] = []
    if ordered:
        site_groups = product(option.sites, repeat=count)
    else:
        site_groups = combinations_with_replacement(option.sites, count)
    for site_group in site_groups:
        total_multiplicity = sum(site.multiplicity for site in site_group)
        if (
            max_multiplicity_total is not None
            and total_multiplicity > max_multiplicity_total
        ):
            continue
        labels = [site.site_label for site in site_group]
        combination = {
            "combination_id": (
                f"sg{option.number}_{count}sites_"
                + "_".join(labels).replace(" ", "")
            ),
            "space_group_number": option.number,
            "space_group_symbol": option.symbol,
            "crystal_system": option.crystal_system,
            "site_count": count,
            "ordered": ordered,
            "site_labels": labels,
            "letters": [site.letter for site in site_group],
            "multiplicities": [site.multiplicity for site in site_group],
            "total_multiplicity": total_multiplicity,
            "free_parameter_count": sum(
                site.parameter_count for site in site_group
            ),
            "sites": [site.as_dict() for site in site_group],
        }
        combinations.append(combination)
        if (
            max_combinations is not None
            and len(combinations) >= max_combinations
        ):
            break
    return combinations


def registered_wyckoff_possibilities(
    crystal_system: str | None = None,
    *,
    include_sites: bool = True,
) -> list[dict[str, Any]]:
    """Return the Wyckoff registry as serializable possibility
    records."""

    return [
        option.as_dict(include_sites=include_sites)
        for option in wyckoff_space_group_options(crystal_system)
    ]


def wyckoff_registry_summary() -> dict[str, Any]:
    """Summarize registered Wyckoff coverage across crystal systems."""

    systems = []
    for crystal_system, numbers in CRYSTAL_SYSTEM_SPACE_GROUP_RANGES.items():
        options = wyckoff_space_group_options(crystal_system)
        systems.append(
            {
                "crystal_system": crystal_system,
                "space_group_numbers": list(numbers),
                "space_group_count": len(options),
                "wyckoff_site_count": sum(
                    len(option.sites) for option in options
                ),
            }
        )
    return {
        "crystal_systems": systems,
        "space_group_count": sum(
            item["space_group_count"] for item in systems
        ),
        "wyckoff_site_count": sum(
            item["wyckoff_site_count"] for item in systems
        ),
    }


def generate_ranked_cif_records(
    candidate: LatticeCandidate | dict[str, Any],
    *,
    atoms: Iterable[str],
    atom_specs: Iterable[dict[str, Any]] | None = None,
    molecules: Iterable[dict[str, Any]],
    space_group_number: int | None = None,
    wyckoff_combinations: Iterable[dict[str, Any]] | None = None,
    stoichiometry: str = "",
    density_g_cm3: float | None = None,
    occupancy_constraints: str = "",
    observed_score: float | None = None,
    limit: int = 5,
    allow_explicit_templates: bool = True,
    assume_unit_cell_symmetry: bool = False,
) -> list[dict[str, Any]]:
    """Create lightweight ranked CIF records for downstream
    simulation."""

    candidate_obj = (
        LatticeCandidate.from_dict(candidate)
        if isinstance(candidate, dict)
        else candidate
    ).constrained()
    atom_list = [atom.strip() for atom in atoms if atom and atom.strip()]
    normalized_atom_specs = _normalize_atom_specs(atom_specs, atom_list)
    if normalized_atom_specs:
        atom_list = []
        for spec in normalized_atom_specs:
            element = str(spec.get("element") or "").strip()
            if element and element not in atom_list:
                atom_list.append(element)
    molecule_list = _normalized_molecule_list(molecules, stoichiometry)
    composition = _composition_from_inputs(
        atom_list,
        normalized_atom_specs,
        molecule_list,
        stoichiometry,
    )
    basis_labels = _basis_labels_from_atom_specs(
        normalized_atom_specs,
        atom_list,
    ) + [
        str(item.get("label") or item.get("name") or "molecule")
        for item in molecule_list
    ]
    site_count = max(1, len(basis_labels))
    sg_number = (
        int(space_group_number)
        if space_group_number is not None
        else _default_space_group_for_system(candidate_obj.crystal_system)
    )
    space_group = wyckoff_space_group_option(sg_number)
    uses_explicit_template = allow_explicit_templates and bool(
        _explicit_template_rows(composition, space_group, stoichiometry)
    )
    combination_list = [dict(item) for item in wyckoff_combinations or []]
    if not combination_list:
        combination_list = wyckoff_site_combinations(
            sg_number,
            site_count=site_count,
            max_combinations=max(limit * 20, limit),
            ordered=True,
        )
    combination_list = sorted(
        combination_list,
        key=lambda item: (
            abs(int(item.get("site_count", site_count)) - site_count),
            int(item.get("total_multiplicity", 0)),
            int(item.get("free_parameter_count", 0)),
            tuple(item.get("site_labels", [])),
        ),
    )
    base_score = (
        candidate_obj.score
        if observed_score is None
        else float(observed_score)
    )
    composition_penalty = 0.0 if stoichiometry.strip() else 0.25
    molecule_penalty = max(0, len(molecule_list) - 1) * 0.03
    records = []
    for rank in range(1, max(1, limit) + 1):
        combination = (
            combination_list[(rank - 1) % len(combination_list)]
            if combination_list
            else {}
        )
        assignments = _basis_wyckoff_assignments(basis_labels, combination)
        score = base_score + composition_penalty + molecule_penalty
        score += _wyckoff_combination_penalty(combination, site_count)
        score += 0.04 * (rank - 1)
        cif_id = f"{candidate_obj.candidate_id}_cif_{rank:02d}"
        records.append(
            {
                "cif_id": cif_id,
                "candidate_id": candidate_obj.candidate_id,
                "rank": rank,
                "score": float(score),
                "composition": stoichiometry.strip(),
                "density_g_cm3": density_g_cm3,
                "occupancy_constraints": occupancy_constraints.strip(),
                "atoms": atom_list,
                "atom_specs": normalized_atom_specs,
                "molecules": molecule_list,
                "composition_elements": dict(composition),
                "space_group": space_group.as_dict(include_sites=False),
                "wyckoff_combination": combination,
                "wyckoff_assignments": assignments,
                "unit_cell_symmetry_assumed": bool(assume_unit_cell_symmetry),
                "coordinate_model": (
                    "explicit_full_cell_ma_dmf_pb3i8"
                    if uses_explicit_template
                    else (
                        "symmetry_constrained_fractional_grid"
                        if assume_unit_cell_symmetry
                        else "deterministic_fractional_grid"
                    )
                ),
                "status": (
                    "full-cell molecular draft"
                    if uses_explicit_template
                    else (
                        "symmetry-constrained draft"
                        if assume_unit_cell_symmetry
                        else "ranked draft"
                    )
                ),
                "cif_text": _draft_cif_text(
                    candidate_obj,
                    cif_id,
                    atom_list,
                    molecule_list,
                    stoichiometry,
                    space_group=space_group,
                    wyckoff_combination=combination,
                    wyckoff_assignments=assignments,
                    composition=composition,
                    atom_specs=normalized_atom_specs,
                    allow_explicit_template=allow_explicit_templates,
                    assume_unit_cell_symmetry=assume_unit_cell_symmetry,
                ),
            }
        )
    return sorted(records, key=lambda item: item["score"])


def format_hkl(value: Any) -> str:
    """Return crystallographic ``(h k l)`` formatting."""

    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped.startswith("(") and stripped.endswith(")"):
            return stripped
        parts = stripped.replace(",", " ").split()
    else:
        parts = list(value)
    if len(parts) != 3:
        return str(value)
    try:
        h, k, ell = (int(float(part)) for part in parts)
    except (TypeError, ValueError):
        return str(value)
    return f"({h} {k} {ell})"


def _structure_peak_from_record(
    record: dict[str, Any],
    fit_store: dict[str, Any],
) -> StructurePeak:
    peak_id = _peak_id(record)
    fit_2d = fit_store.get("fit_2d") if isinstance(fit_store, dict) else None
    source = _structure_source(record, fit_2d)
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    qxy = _peak_qxy(record)
    qz = _peak_qz(record)
    fit_quality = None
    if isinstance(fit_2d, dict):
        qxy = float(fit_2d.get("center_qxy", qxy))
        qz = float(fit_2d.get("center_qz", qz))
        fit_quality = _optional_float(
            fit_2d.get("statistics", {}).get("r_squared")
        )
    phase_tag = str(
        record.get(PEAK_PHASE_METADATA_KEY)
        or metadata.get(PEAK_PHASE_METADATA_KEY)
        or DEFAULT_PHASE_TAG
    )
    if _record_is_forbidden_reflection(record, metadata):
        status = PHASE_FORBIDDEN
        phase_tag = PHASE_FORBIDDEN
        include = False
    elif _record_is_gap_estimated(record, metadata):
        status = PHASE_GAP_ESTIMATED
        if phase_tag == DEFAULT_PHASE_TAG:
            phase_tag = PHASE_GAP_ESTIMATED
        include = bool(
            record.get("include", record.get("include_in_fit", True))
        )
    else:
        status = str(record.get("status", ""))
        include = bool(
            record.get("include", record.get("include_in_fit", True))
        )
    hkl_label = _record_hkl_label(record, metadata)
    return StructurePeak(
        peak_id=peak_id,
        label=str(record.get("label", peak_id)),
        qxy=qxy,
        qz=qz,
        source=source,
        phase_tag=phase_tag,
        hkl_label=hkl_label,
        include=include,
        fit_quality=fit_quality,
        status=status,
        notes=str(record.get("notes", "")),
        metadata={"peak_record": dict(record), "fit_record": dict(fit_store)},
    )


def _structure_source(record: dict[str, Any], fit_2d: Any) -> str:
    if isinstance(fit_2d, dict):
        return SOURCE_ROI_FIT
    source = str(record.get("source", "")).lower()
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if "gap" in source or _record_is_gap_estimated(record, metadata):
        return SOURCE_GAP_ESTIMATE
    if "integration" in source or "import" in source:
        return SOURCE_IMPORTED_POINT
    if "manual" in source or "user" in source:
        return SOURCE_USER_EDIT
    return SOURCE_MARKED_POINT


def _record_is_gap_estimated(
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    return bool(
        record.get("gap_estimated")
        or record.get("point_kind") == PEAK_POINT_KIND_GAP_ESTIMATED
        or metadata.get("gap_estimate")
        or str(record.get("source", "")).lower().startswith("gap")
    )


def _record_is_forbidden_reflection(
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    status = str(record.get("reflection_status", "")).lower()
    return bool(
        record.get("forbidden_reflection")
        or record.get("excluded_from_indexing")
        or metadata.get("forbidden_reflection")
        or metadata.get("excluded_from_indexing")
        or status == "forbidden"
    )


def _record_hkl_label(
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    if record.get("hkl_label"):
        return str(record.get("hkl_label"))
    raw = record.get("hkl")
    if raw is None:
        raw = record.get(PEAK_HKL_METADATA_KEY)
    if raw is None:
        raw = metadata.get(PEAK_HKL_METADATA_KEY)
    if isinstance(raw, dict):
        label = str(raw.get("label") or "").strip()
        if label:
            return label
        raw = [raw.get("h"), raw.get("k"), raw.get("l")]
    return format_hkl(raw or "")


def _active_peaks(
    peaks: Iterable[StructurePeak | dict[str, Any]],
    phase_tag: str,
) -> list[StructurePeak]:
    active: list[StructurePeak] = []
    for item in peaks:
        peak = (
            StructurePeak.from_dict(item) if isinstance(item, dict) else item
        )
        if not peak.include:
            continue
        if peak.phase_tag in {PHASE_REJECTED, PHASE_FORBIDDEN}:
            continue
        if phase_tag and peak.phase_tag != phase_tag:
            continue
        if peak.q_magnitude <= 1.0e-9 or not np.isfinite(peak.q_magnitude):
            continue
        active.append(peak)
    return active


def _grid_candidates(
    peaks: list[StructurePeak],
    crystal_system: str,
    config: CandidateSearchConfig,
) -> list[LatticeCandidate]:
    values = np.linspace(
        config.lattice_min,
        config.lattice_max,
        max(3, config.grid_points),
    )
    coarse_values = np.linspace(
        config.lattice_min,
        config.lattice_max,
        max(3, min(config.grid_points, 10)),
    )
    grid: Iterable[tuple[float, ...]]
    if crystal_system == "Cubic":
        grid = ((a,) for a in values)
    elif crystal_system in {"Tetragonal", "Hexagonal", "Trigonal"}:
        grid = product(values, coarse_values)
    elif crystal_system == "Orthorhombic":
        grid = product(coarse_values, repeat=3)
    elif crystal_system == "Monoclinic":
        grid = product(coarse_values, coarse_values, coarse_values, (90.0,))
    else:
        grid = product(coarse_values, coarse_values, coarse_values)

    candidates = []
    for index, variables in enumerate(grid):
        candidate = _candidate_for_variables(
            crystal_system,
            tuple(float(value) for value in variables),
            f"{crystal_system.lower()}_grid_{index:04d}",
            orientation_quaternion=config.orientation_quaternion,
        )
        candidates.append(_score_candidate(candidate, peaks, config))
    return sorted(
        candidates,
        key=lambda item: _candidate_preference_key(item, config),
    )[:4]


def _cubic_indexing_candidates(
    peaks: list[StructurePeak],
    config: CandidateSearchConfig,
) -> list[LatticeCandidate]:
    hkl_rows = _hkl_families(config.hkl_max)
    estimates = []
    for peak in peaks:
        q_obs = peak.q_magnitude
        for hkl in hkl_rows:
            norm = float(np.linalg.norm(hkl))
            if norm <= 0.0:
                continue
            a = 2.0 * np.pi * norm / q_obs
            if config.lattice_min <= a <= config.lattice_max:
                estimates.append(a)
    if not estimates:
        return []
    bins: dict[int, list[float]] = {}
    for estimate in estimates:
        bins.setdefault(int(round(estimate / 0.25)), []).append(estimate)
    candidates = []
    for index, values in enumerate(
        sorted(
            bins.values(),
            key=lambda values: (-len(values), float(np.median(values))),
        )[:8]
    ):
        a = float(np.median(values))
        candidate = LatticeCandidate(
            candidate_id=f"cubic_indexed_{index + 1:02d}",
            crystal_system="Cubic",
            a=a,
            b=a,
            c=a,
            method="cubic-indexing",
            orientation_quaternion=config.orientation_quaternion,
        )
        candidates.append(_score_candidate(candidate, peaks, config))
    return candidates


def _projected_axis_candidates(
    peaks: list[StructurePeak],
    config: CandidateSearchConfig,
) -> list[LatticeCandidate]:
    """Infer GIWAXS-oriented cells from qxy/qz peak-coordinate
    families."""

    systems = set(_ordered_crystal_systems(config.crystal_systems))
    if not systems.intersection({"Orthorhombic", "Monoclinic", "Triclinic"}):
        return []
    qxy_values = [
        abs(float(peak.qxy))
        for peak in peaks
        if np.isfinite(peak.qxy) and abs(float(peak.qxy)) > config.q_tolerance
    ]
    qz_values = [
        abs(float(peak.qz))
        for peak in peaks
        if np.isfinite(peak.qz) and abs(float(peak.qz)) > config.q_tolerance
    ]
    if len(qxy_values) < 2 or len(qz_values) < 2:
        return []

    long_inplane_steps = _coordinate_step_candidates(
        qxy_values,
        config,
        max_steps=4,
    )
    vertical_steps = _coordinate_step_candidates(
        qz_values,
        config,
        max_steps=4,
    )
    if not long_inplane_steps or not vertical_steps:
        return []

    orientation = quaternion_from_axis_angle((1.0, 0.0, 0.0), 90.0)
    candidates: list[LatticeCandidate] = []
    index = 0
    for qxy_step, _, _ in long_inplane_steps:
        compact_steps = _compact_inplane_step_candidates(
            qxy_values,
            qxy_step,
            config,
            max_steps=5,
        )
        for qz_step, _, _ in vertical_steps:
            for compact_step, _, _ in compact_steps or [(qxy_step, 0, 0.0)]:
                lengths = (
                    2.0 * np.pi / qxy_step,
                    2.0 * np.pi / qz_step,
                    2.0 * np.pi / compact_step,
                )
                if not all(
                    config.lattice_min <= value <= config.lattice_max
                    for value in lengths
                ):
                    continue
                if lengths[2] > max(lengths[0], lengths[1]) * 0.85:
                    continue
                for system in (
                    "Orthorhombic",
                    "Monoclinic",
                    "Triclinic",
                ):
                    if system not in systems:
                        continue
                    index += 1
                    candidate = LatticeCandidate(
                        candidate_id=f"projected_axis_{index:03d}",
                        crystal_system=system,
                        a=float(lengths[0]),
                        b=float(lengths[1]),
                        c=float(lengths[2]),
                        alpha=90.0,
                        beta=90.0,
                        gamma=90.0,
                        method="projected-axis-indexing",
                        notes=(
                            "Inferred from qxy/qz coordinate families; "
                            "a and c are in-plane, b is out-of-plane."
                        ),
                        orientation_quaternion=orientation,
                        projection_mode=PROJECTION_FIBER_QXY_QZ,
                    )
                    scored = _score_candidate(candidate, peaks, config)
                    total = max(scored.matched_count + scored.outlier_count, 1)
                    if (
                        scored.matched_count >= 3
                        and scored.matched_count / total >= 0.3
                    ):
                        candidates.append(scored)
    return sorted(
        candidates,
        key=lambda item: _candidate_preference_key(item, config),
    )[: max(4, config.max_candidates)]


def _coordinate_step_candidates(
    values: Iterable[float],
    config: CandidateSearchConfig,
    *,
    max_steps: int,
) -> list[tuple[float, int, float]]:
    values_array = np.asarray(
        [abs(float(value)) for value in values if abs(float(value)) > 1.0e-9],
        dtype=float,
    )
    if values_array.size == 0:
        return []
    q_min = 2.0 * np.pi / config.lattice_max
    q_max = 2.0 * np.pi / config.lattice_min
    estimates: list[float] = []
    for value in values_array:
        for order in range(1, max(1, config.hkl_max) + 1):
            step = float(value) / float(order)
            if q_min <= step <= q_max:
                estimates.append(step)
    clusters = _cluster_coordinate_steps(
        estimates,
        tolerance=max(config.q_tolerance * 0.4, 0.01),
    )
    scored = []
    for step in clusters:
        residuals = _coordinate_step_residuals(
            values_array,
            step,
            hkl_max=config.hkl_max,
        )
        tolerance = max(config.q_tolerance, step * config.relative_tolerance)
        support = int(np.count_nonzero(residuals <= tolerance))
        finite = residuals[np.isfinite(residuals)]
        rms = (
            float(np.sqrt(np.nanmean(finite**2)))
            if finite.size
            else float("inf")
        )
        scored.append((float(step), support, rms))
    return sorted(
        scored,
        key=lambda item: (-item[1], item[2], item[0]),
    )[:max_steps]


def _compact_inplane_step_candidates(
    qxy_values: Iterable[float],
    primary_step: float,
    config: CandidateSearchConfig,
    *,
    max_steps: int,
) -> list[tuple[float, int, float]]:
    values_array = np.asarray(
        [abs(float(value)) for value in qxy_values if abs(float(value)) > 0.0],
        dtype=float,
    )
    if values_array.size == 0 or primary_step <= 0.0:
        return []
    q_min = 2.0 * np.pi / config.lattice_max
    q_max = 2.0 * np.pi / config.lattice_min
    compact_min = max(primary_step * 2.25, q_min)
    estimates = []
    for value in values_array:
        for h_index in range(0, max(1, config.hkl_max) + 1):
            residual_sq = value * value - (h_index * primary_step) ** 2
            if residual_sq <= 1.0e-12:
                continue
            residual = float(np.sqrt(residual_sq))
            for l_index in range(1, max(1, config.hkl_max) + 1):
                step = residual / float(l_index)
                if compact_min <= step <= q_max:
                    estimates.append(step)
    estimates.extend(
        value for value in values_array if compact_min <= value <= q_max
    )
    clusters = _cluster_coordinate_steps(
        estimates,
        tolerance=max(config.q_tolerance * 0.4, 0.01),
    )
    scored = []
    for step in clusters:
        residuals = _fiber_inplane_step_residuals(
            values_array,
            primary_step,
            step,
            config.hkl_max,
        )
        tolerance = max(config.q_tolerance, step * config.relative_tolerance)
        support = int(np.count_nonzero(residuals <= tolerance))
        rms = float(np.sqrt(np.nanmean(residuals**2)))
        scored.append((float(step), support, rms))
    return sorted(
        scored,
        key=lambda item: (-item[1], item[2], item[0]),
    )[:max_steps]


def _fiber_inplane_step_residuals(
    values: np.ndarray,
    primary_step: float,
    compact_step: float,
    hkl_max: int,
) -> np.ndarray:
    limit = max(1, int(hkl_max))
    predicted = np.asarray(
        [
            np.hypot(h_index * primary_step, l_index * compact_step)
            for h_index in range(0, limit + 1)
            for l_index in range(0, limit + 1)
            if h_index or l_index
        ],
        dtype=float,
    )
    if predicted.size == 0:
        return np.full(values.shape, np.inf, dtype=float)
    return np.asarray(
        [float(np.nanmin(np.abs(predicted - value))) for value in values],
        dtype=float,
    )


def _direct_coordinate_steps(
    values: Iterable[float],
    *,
    minimum: float,
    maximum: float,
    tolerance: float,
    max_steps: int,
) -> list[tuple[float, int, float]]:
    direct = [
        abs(float(value))
        for value in values
        if minimum <= abs(float(value)) <= maximum
    ]
    clusters = _cluster_coordinate_steps(direct, tolerance=tolerance)
    scored = []
    for step in clusters:
        residuals = [
            abs(abs(float(value)) - step)
            for value in values
            if abs(abs(float(value)) - step) <= tolerance
        ]
        support = len(residuals)
        rms = (
            float(np.sqrt(np.nanmean(np.asarray(residuals, dtype=float) ** 2)))
            if residuals
            else float("inf")
        )
        scored.append((float(step), support, rms))
    return sorted(
        scored,
        key=lambda item: (-item[1], item[2], item[0]),
    )[:max_steps]


def _cluster_coordinate_steps(
    values: Iterable[float],
    *,
    tolerance: float,
) -> list[float]:
    sorted_values = sorted(
        float(value) for value in values if np.isfinite(value)
    )
    if not sorted_values:
        return []
    clusters: list[list[float]] = []
    for value in sorted_values:
        if (
            not clusters
            or abs(value - float(np.median(clusters[-1]))) > tolerance
        ):
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.median(cluster)) for cluster in clusters]


def _coordinate_step_residuals(
    values: np.ndarray,
    step: float,
    *,
    hkl_max: int | None = None,
) -> np.ndarray:
    if step <= 0.0:
        return np.full(values.shape, np.inf, dtype=float)
    orders = np.maximum(1.0, np.round(values / step))
    residuals = np.abs(values - orders * step)
    if hkl_max is not None:
        limit = max(1, int(hkl_max))
        invalid = orders > float(limit)
        residuals = np.asarray(residuals, dtype=float)
        residuals[invalid] = np.inf
    return residuals


def _rank_unique_candidates(
    candidates: list[LatticeCandidate],
    peaks: list[StructurePeak],
    config: CandidateSearchConfig,
) -> list[LatticeCandidate]:
    ranked = sorted(
        candidates,
        key=lambda item: _candidate_preference_key(item, config),
    )
    unique: list[LatticeCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in ranked:
        constrained = candidate.constrained()
        key = (
            constrained.crystal_system,
            constrained.projection_mode,
            round(constrained.a, 1),
            round(constrained.b, 1),
            round(constrained.c, 1),
            round(constrained.alpha, 1),
            round(constrained.beta, 1),
            round(constrained.gamma, 1),
            tuple(
                round(value, 3)
                for value in _orientation_quaternion_or_identity(
                    constrained.orientation_quaternion
                )
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(_score_candidate(constrained, peaks, config))
        if len(unique) >= config.max_candidates:
            break
    for index, candidate in enumerate(unique, start=1):
        if not candidate.candidate_id.startswith("candidate_"):
            candidate.candidate_id = f"candidate_{index:03d}"
    return unique


def _ordered_crystal_systems(systems: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for system in systems:
        if system in CRYSTAL_SYSTEMS and system not in seen:
            ordered.append(system)
            seen.add(system)
    return tuple(sorted(ordered, key=_crystal_system_complexity_rank))


def _crystal_system_complexity_rank(system: str) -> int:
    return CRYSTAL_SYSTEM_COMPLEXITY.get(
        system,
        len(SIMPLE_CRYSTAL_SYSTEM_ORDER),
    )


def _candidate_lattice_scale(candidate: LatticeCandidate) -> float:
    constrained = candidate.constrained()
    return float(np.mean([constrained.a, constrained.b, constrained.c]))


def _candidate_preference_key(
    candidate: LatticeCandidate,
    config: CandidateSearchConfig,
) -> tuple[float, ...]:
    score = float(candidate.score)
    if not np.isfinite(score):
        return (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
    score_bin = max(float(config.q_tolerance) * 0.5, 1.0e-9)
    return (
        _projection_preference_rank(candidate),
        float(candidate.outlier_count),
        -float(candidate.matched_count),
        float(np.floor(score / score_bin)),
        _candidate_assignment_complexity(candidate, config),
        float(_crystal_system_complexity_rank(candidate.crystal_system)),
        _candidate_lattice_scale(candidate),
        score,
    )


def _candidate_assignment_complexity(
    candidate: LatticeCandidate,
    config: CandidateSearchConfig,
) -> float:
    return _assignment_hkl_complexity(candidate.assignments, config)


def _projection_preference_rank(candidate: LatticeCandidate) -> float:
    total = max(candidate.matched_count + candidate.outlier_count, 1)
    matched_fraction = candidate.matched_count / total
    if (
        candidate.projection_mode != PROJECTION_RADIAL
        and candidate.matched_count >= 3
        and matched_fraction >= 0.3
    ):
        return 0.0
    return 1.0


def _optional_orientation_quaternion(
    value: Any,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    return tuple(float(component) for component in normalize_quaternion(value))


def _orientation_quaternion_or_identity(
    value: Any,
) -> tuple[float, float, float, float]:
    orientation = _optional_orientation_quaternion(value)
    if orientation is None:
        return 0.0, 0.0, 0.0, 1.0
    return orientation


def _score_candidate(
    candidate: LatticeCandidate,
    peaks: list[StructurePeak],
    config: CandidateSearchConfig,
    *,
    method: str | None = None,
) -> LatticeCandidate:
    if (
        candidate.orientation_quaternion is None
        and config.orientation_quaternion is not None
    ):
        candidate.orientation_quaternion = _optional_orientation_quaternion(
            config.orientation_quaternion
        )
    candidate = candidate.constrained()
    assignments = _assign_peaks_to_candidate(peaks, candidate, config)
    if not assignments:
        candidate.score = float("inf")
        candidate.rms_error = float("inf")
        candidate.matched_count = 0
        candidate.outlier_count = 0
        candidate.assignments = []
        return candidate
    deltas = np.asarray(
        [float(item["delta_q"]) for item in assignments], dtype=float
    )
    tolerances = np.asarray(
        [float(item["tolerance"]) for item in assignments], dtype=float
    )
    hkl_complexity = _assignment_hkl_complexity(assignments, config)
    within = np.abs(deltas) <= tolerances
    rms = float(np.sqrt(np.nanmean(deltas**2)))
    outlier_count = int(np.count_nonzero(~within))
    score = rms + outlier_count * config.q_tolerance * 1.5
    if rms > 1.0e-9 or outlier_count:
        score += hkl_complexity * config.q_tolerance * 0.35
    candidate.score = float(score)
    candidate.rms_error = rms
    candidate.matched_count = int(np.count_nonzero(within))
    candidate.outlier_count = outlier_count
    candidate.assignments = assignments
    if method is not None:
        candidate.method = method
    return candidate


def _assign_peaks_to_candidate(
    peaks: list[StructurePeak],
    candidate: LatticeCandidate,
    config: CandidateSearchConfig,
) -> list[dict[str, Any]]:
    if (
        candidate.orientation_quaternion is not None
        or candidate.projection_mode != PROJECTION_RADIAL
    ):
        return _assign_projected_peaks_to_candidate(peaks, candidate, config)
    hkl_rows = _hkl_families(config.hkl_max)
    predicted = _predicted_q_values(candidate, hkl_rows)
    if predicted.size == 0:
        return []
    assignments = []
    for peak in peaks:
        q_obs = peak.q_magnitude
        labeled_hkl = _parse_hkl_label(peak.hkl_label)
        if labeled_hkl is None:
            index = int(np.argmin(np.abs(predicted - q_obs)))
            hkl = hkl_rows[index]
            hkl_label = format_hkl(hkl)
            q_predicted = float(predicted[index])
        else:
            hkl = np.asarray(labeled_hkl, dtype=int)
            hkl_label = format_hkl(hkl)
            q_predicted = float(
                _predicted_q_values(
                    candidate,
                    np.asarray([labeled_hkl], dtype=int),
                )[0]
            )
        delta = float(q_obs - q_predicted)
        tolerance = max(
            config.q_tolerance,
            abs(q_obs) * config.relative_tolerance,
        )
        if peak.phase_tag == PHASE_GAP_ESTIMATED:
            tolerance *= 1.5
        assignments.append(
            {
                "peak_id": peak.peak_id,
                "label": peak.label,
                "q_observed": float(q_obs),
                "q_predicted": q_predicted,
                "delta_q": delta,
                "abs_delta_q": abs(delta),
                "tolerance": float(tolerance),
                "hkl": hkl_label,
                **_hkl_assignment_metrics(hkl),
            }
        )
    return assignments


def _assign_projected_peaks_to_candidate(
    peaks: list[StructurePeak],
    candidate: LatticeCandidate,
    config: CandidateSearchConfig,
) -> list[dict[str, Any]]:
    hkl_rows = _signed_hkl_grid(config.hkl_max)
    predicted = _predicted_projected_q_points(candidate, hkl_rows)
    if predicted.size == 0:
        return []
    assignments = []
    for peak in peaks:
        qxy_observed = (
            abs(float(peak.qxy))
            if candidate.projection_mode == PROJECTION_FIBER_QXY_QZ
            else float(peak.qxy)
        )
        labeled_hkl = _parse_hkl_label(peak.hkl_label)
        if labeled_hkl is None:
            deltas = predicted - np.array(
                [qxy_observed, peak.qz],
                dtype=float,
            )
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            index = int(np.argmin(distances))
            hkl = hkl_rows[index]
            hkl_label = format_hkl(hkl)
            qxy_predicted = float(predicted[index, 0])
            qz_predicted = float(predicted[index, 1])
            distance = float(distances[index])
        else:
            hkl = np.asarray(labeled_hkl, dtype=int)
            hkl_label = format_hkl(hkl)
            qxy_predicted, qz_predicted = (
                float(value)
                for value in _predicted_projected_q_points(
                    candidate,
                    np.asarray([labeled_hkl], dtype=int),
                )[0]
            )
            distance = float(
                np.hypot(
                    qxy_observed - qxy_predicted,
                    peak.qz - qz_predicted,
                )
            )
        q_predicted = float(np.hypot(qxy_predicted, qz_predicted))
        tolerance = max(
            config.q_tolerance,
            peak.q_magnitude * config.relative_tolerance,
        )
        if peak.phase_tag == PHASE_GAP_ESTIMATED:
            tolerance *= 1.5
        assignments.append(
            {
                "peak_id": peak.peak_id,
                "label": peak.label,
                "q_observed": float(peak.q_magnitude),
                "q_predicted": q_predicted,
                "qxy_observed": qxy_observed,
                "qz_observed": float(peak.qz),
                "qxy_predicted": qxy_predicted,
                "qz_predicted": qz_predicted,
                "delta_q": distance,
                "abs_delta_q": distance,
                "projected_delta_q": distance,
                "tolerance": float(tolerance),
                "hkl": hkl_label,
                **_hkl_assignment_metrics(hkl),
            }
        )
    return assignments


def _hkl_assignment_metrics(hkl: Any) -> dict[str, Any]:
    values = np.asarray(hkl, dtype=int).reshape(3)
    abs_values = np.abs(values)
    return {
        "hkl_indices": [int(value) for value in values],
        "hkl_abs_sum": int(np.sum(abs_values)),
        "hkl_max_abs": int(np.max(abs_values)),
        "hkl_norm": float(np.linalg.norm(values)),
    }


def _assignment_hkl_complexity(
    assignments: list[dict[str, Any]],
    config: CandidateSearchConfig,
) -> float:
    if not assignments:
        return 0.0
    limit = max(float(config.hkl_max), 1.0)
    norms = [
        float(item.get("hkl_norm", 0.0))
        for item in assignments
        if np.isfinite(float(item.get("hkl_norm", 0.0)))
    ]
    if not norms:
        return 0.0
    return float(np.mean(norms) / limit)


def _predicted_q_values(
    candidate: LatticeCandidate,
    hkl_rows: np.ndarray,
) -> np.ndarray:
    vectors = _reciprocal_q_vectors(candidate, hkl_rows)
    return np.linalg.norm(vectors, axis=1)


def _predicted_projected_q_points(
    candidate: LatticeCandidate,
    hkl_rows: np.ndarray,
) -> np.ndarray:
    vectors = _reciprocal_q_vectors(candidate, hkl_rows)
    orientation = _orientation_quaternion_or_identity(
        candidate.orientation_quaternion
    )
    rotated = rotate_vectors_by_quaternion(vectors, orientation)
    if candidate.projection_mode == PROJECTION_FIBER_QXY_QZ:
        return np.column_stack(
            (np.hypot(rotated[:, 0], rotated[:, 1]), rotated[:, 2])
        )
    return np.column_stack((rotated[:, 0], rotated[:, 2]))


def _reciprocal_q_vectors(
    candidate: LatticeCandidate,
    hkl_rows: np.ndarray,
) -> np.ndarray:
    params = candidate.as_parameters()
    lattice = Lattice(
        a=params.a,
        b=params.b,
        c=params.c,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
    )
    reciprocal = lattice.reciprocal()
    return np.asarray(
        [reciprocal.q_vector(row) for row in hkl_rows],
        dtype=float,
    )


def _parameterization(
    candidate: LatticeCandidate,
    config: CandidateSearchConfig,
):
    system = candidate.crystal_system
    lower = config.lattice_min
    upper = config.lattice_max
    if system == "Cubic":
        variables = [candidate.a]
        bounds = ([lower], [upper])

        def unpack(values: Iterable[float]) -> LatticeCandidate:
            a = float(list(values)[0])
            return LatticeCandidate(
                candidate.candidate_id,
                system,
                a,
                a,
                a,
                method=candidate.method,
                orientation_quaternion=candidate.orientation_quaternion,
                projection_mode=candidate.projection_mode,
            )

    elif system in {"Tetragonal", "Hexagonal", "Trigonal"}:
        variables = [candidate.a, candidate.c]
        bounds = ([lower, lower], [upper, upper])

        def unpack(values: Iterable[float]) -> LatticeCandidate:
            a, c = (float(value) for value in values)
            gamma = 120.0 if system in {"Hexagonal", "Trigonal"} else 90.0
            return LatticeCandidate(
                candidate.candidate_id,
                system,
                a,
                a,
                c,
                gamma=gamma,
                method=candidate.method,
                orientation_quaternion=candidate.orientation_quaternion,
                projection_mode=candidate.projection_mode,
            )

    else:
        variables = [candidate.a, candidate.b, candidate.c]
        bounds = ([lower, lower, lower], [upper, upper, upper])

        def unpack(values: Iterable[float]) -> LatticeCandidate:
            a, b, c = (float(value) for value in values)
            return LatticeCandidate(
                candidate.candidate_id,
                system,
                a,
                b,
                c,
                alpha=candidate.alpha,
                beta=candidate.beta,
                gamma=candidate.gamma,
                method=candidate.method,
                orientation_quaternion=candidate.orientation_quaternion,
                projection_mode=candidate.projection_mode,
            )

    return variables, bounds, unpack


def _candidate_for_variables(
    crystal_system: str,
    variables: tuple[float, ...],
    candidate_id: str,
    *,
    orientation_quaternion: tuple[float, float, float, float] | None = None,
) -> LatticeCandidate:
    if crystal_system == "Cubic":
        a = variables[0]
        return LatticeCandidate(
            candidate_id,
            crystal_system,
            a,
            a,
            a,
            orientation_quaternion=orientation_quaternion,
            projection_mode=(
                PROJECTION_CARTESIAN_XZ
                if orientation_quaternion is not None
                else PROJECTION_RADIAL
            ),
        )
    if crystal_system in {"Tetragonal", "Hexagonal", "Trigonal"}:
        a, c = variables[:2]
        gamma = 120.0 if crystal_system in {"Hexagonal", "Trigonal"} else 90.0
        return LatticeCandidate(
            candidate_id,
            crystal_system,
            a,
            a,
            c,
            gamma=gamma,
            orientation_quaternion=orientation_quaternion,
            projection_mode=(
                PROJECTION_CARTESIAN_XZ
                if orientation_quaternion is not None
                else PROJECTION_RADIAL
            ),
        )
    a, b, c = variables[:3]
    return LatticeCandidate(
        candidate_id,
        crystal_system,
        a,
        b,
        c,
        orientation_quaternion=orientation_quaternion,
        projection_mode=(
            PROJECTION_CARTESIAN_XZ
            if orientation_quaternion is not None
            else PROJECTION_RADIAL
        ),
    )


def _candidate_from_initial(
    initial: LatticeCandidate | CrystalOverlayParameters | dict[str, Any],
) -> LatticeCandidate:
    if isinstance(initial, LatticeCandidate):
        return initial
    if isinstance(initial, CrystalOverlayParameters):
        params = initial.constrained()
        return LatticeCandidate(
            candidate_id="best_guess_refined",
            crystal_system=params.crystal_system,
            a=params.a,
            b=params.b,
            c=params.c,
            alpha=params.alpha,
            beta=params.beta,
            gamma=params.gamma,
            method="best-guess",
            orientation_quaternion=tuple(params.orientation_quaternion),
            projection_mode=PROJECTION_CARTESIAN_XZ,
        )
    return LatticeCandidate.from_dict(initial)


def _hkl_families(max_index: int) -> np.ndarray:
    rows = []
    for h, k, ell in product(range(max_index + 1), repeat=3):
        if h == k == ell == 0:
            continue
        divisor = gcd(gcd(abs(h), abs(k)), abs(ell))
        if divisor > 1:
            reduced = (h // divisor, k // divisor, ell // divisor)
        else:
            reduced = (h, k, ell)
        rows.append(tuple(sorted(reduced, reverse=True)))
    unique = sorted(
        set(rows), key=lambda row: (sum(value * value for value in row), row)
    )
    return np.asarray(unique, dtype=int)


def _signed_hkl_grid(max_index: int) -> np.ndarray:
    limit = max(0, int(max_index))
    rows = [
        (h, k, ell)
        for h, k, ell in product(
            range(-limit, limit + 1),
            repeat=3,
        )
        if not (h == k == ell == 0)
    ]
    return np.asarray(rows, dtype=int)


def _parse_hkl_label(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        stripped = stripped.strip("()")
        parts = stripped.replace(",", " ").split()
    else:
        parts = list(value)
    if len(parts) != 3:
        return None
    try:
        return tuple(int(float(part)) for part in parts)
    except (TypeError, ValueError):
        return None


def _coordinate_families(
    peaks: list[StructurePeak],
    *,
    key: str,
    label: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    clusters = _coordinate_value_clusters(
        [
            (abs(float(getattr(peak, key))), peak)
            for peak in peaks
            if abs(float(getattr(peak, key))) > 1.0e-9
        ],
        tolerance=tolerance,
    )
    families = []
    for group in clusters:
        if len(group) < 2:
            continue
        values = [abs(float(getattr(peak, key))) for peak in group]
        reference = float(np.median(values))
        spread = max(values) - min(values)
        tightness = max(0.0, 1.0 - spread / max(float(tolerance), 1.0e-9))
        count_bonus = min(0.15, 0.05 * max(0, len(group) - 2))
        confidence = min(1.0, 0.55 + 0.4 * tightness + count_bonus)
        reason = (
            f"{len(group)} peaks share similar |{key}| values near "
            f"{reference:.4g}; spread {spread:.4g} within tolerance "
            f"{tolerance:.4g}."
        )
        families.append(
            {
                "kind": label,
                "reference": reference,
                "member_count": len(group),
                "span": float(spread),
                "relative_span": float(spread / max(abs(reference), 1.0e-9)),
                "peak_ids": [peak.peak_id for peak in group],
                "labels": [peak.label for peak in group],
                "confidence": confidence,
                "reason": reason,
                "notes": "candidate family; similar coordinate within cluster",
            }
        )
    return families


def _coordinate_value_clusters(
    values: Iterable[tuple[float, StructurePeak]],
    *,
    tolerance: float,
) -> list[list[StructurePeak]]:
    sorted_values = sorted(
        (
            (float(value), peak)
            for value, peak in values
            if np.isfinite(float(value))
        ),
        key=lambda item: item[0],
    )
    clusters: list[list[tuple[float, StructurePeak]]] = []
    for value, peak in sorted_values:
        if not clusters:
            clusters.append([(value, peak)])
            continue
        cluster_values = [item[0] for item in clusters[-1]]
        candidate_min = min(min(cluster_values), value)
        candidate_max = max(max(cluster_values), value)
        if candidate_max - candidate_min > tolerance:
            clusters.append([(value, peak)])
        else:
            clusters[-1].append((value, peak))
    return [[peak for _, peak in cluster] for cluster in clusters]


def _multiple_families(
    peaks: list[StructurePeak],
    *,
    key: str,
    label: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    positive = [
        peak for peak in peaks if abs(float(getattr(peak, key))) > 1.0e-9
    ]
    families = []
    for base in positive:
        base_value = abs(float(getattr(base, key)))
        group = [base]
        ratios = ["1"]
        ratio_errors = [0.0]
        for peak in positive:
            if peak.peak_id == base.peak_id:
                continue
            ratio = abs(float(getattr(peak, key))) / base_value
            nearest = round(ratio)
            if nearest in {2, 3, 4} and abs(ratio - nearest) <= tolerance:
                group.append(peak)
                ratios.append(str(nearest))
                ratio_errors.append(abs(ratio - nearest))
        if len(group) >= 2:
            mean_error = float(np.mean(ratio_errors))
            tightness = max(
                0.0,
                1.0 - mean_error / max(float(tolerance), 1.0e-9),
            )
            count_bonus = min(0.15, 0.05 * max(0, len(group) - 2))
            confidence = min(1.0, 0.55 + 0.4 * tightness + count_bonus)
            reason = (
                f"{len(group)} peaks fit simple |{key}| multiples from "
                f"{base.label}; mean ratio error {mean_error:.4g} "
                f"within tolerance {tolerance:.4g}."
            )
            families.append(
                {
                    "kind": label,
                    "reference": base_value,
                    "member_count": len(group),
                    "mean_ratio_error": mean_error,
                    "max_ratio_error": float(np.max(ratio_errors)),
                    "peak_ids": [peak.peak_id for peak in group],
                    "labels": [peak.label for peak in group],
                    "confidence": confidence,
                    "reason": reason,
                    "notes": "candidate family; simple multiples "
                    + ", ".join(ratios),
                }
            )
    families.extend(
        _inferred_step_multiple_families(
            positive,
            key=key,
            label=label,
            tolerance=tolerance,
        )
    )
    return _rank_unique_multiple_families(families)


def _inferred_step_multiple_families(
    peaks: list[StructurePeak],
    *,
    key: str,
    label: str,
    tolerance: float,
    max_order: int = 6,
) -> list[dict[str, Any]]:
    values = [
        (abs(float(getattr(peak, key))), peak)
        for peak in peaks
        if abs(float(getattr(peak, key))) > 1.0e-9
    ]
    if len(values) < 2:
        return []
    max_order = max(2, int(max_order))
    step_estimates = sorted(
        {
            value / float(order)
            for value, _peak in values
            for order in range(1, max_order + 1)
            if value > 1.0e-9
        }
    )
    families: list[dict[str, Any]] = []
    for step in step_estimates:
        group: list[StructurePeak] = []
        orders: list[int] = []
        ratio_errors: list[float] = []
        residuals: list[float] = []
        for value, peak in values:
            ratio = value / step
            order = int(round(ratio))
            if order < 1 or order > max_order:
                continue
            ratio_error = abs(ratio - float(order))
            if ratio_error > tolerance:
                continue
            group.append(peak)
            orders.append(order)
            ratio_errors.append(ratio_error)
            residuals.append(abs(value - step * float(order)))
        if len(group) < 2 or len(set(orders)) < 2:
            continue
        mean_error = float(np.mean(ratio_errors))
        max_error = float(np.max(ratio_errors))
        tightness = max(0.0, 1.0 - mean_error / max(float(tolerance), 1.0e-9))
        count_bonus = min(0.2, 0.06 * max(0, len(group) - 2))
        order_bonus = min(0.12, 0.03 * max(0, len(set(orders)) - 2))
        confidence = min(
            1.0, 0.58 + 0.3 * tightness + count_bonus + order_bonus
        )
        ordered = sorted(
            zip(orders, group, strict=False), key=lambda item: item[0]
        )
        reason = (
            f"{len(group)} peaks fit inferred |{key}| step {step:.4g}; "
            f"orders {', '.join(str(order) for order, _peak in ordered)} "
            f"with mean ratio error {mean_error:.4g} within tolerance "
            f"{tolerance:.4g}."
        )
        families.append(
            {
                "kind": label,
                "reference": float(step),
                "member_count": len(group),
                "mean_ratio_error": mean_error,
                "max_ratio_error": max_error,
                "mean_residual": float(np.mean(residuals)),
                "orders": [int(order) for order, _peak in ordered],
                "peak_ids": [peak.peak_id for _order, peak in ordered],
                "labels": [peak.label for _order, peak in ordered],
                "confidence": confidence,
                "reason": reason,
                "notes": "candidate family; inferred coordinate step "
                + ", ".join(str(order) for order, _peak in ordered),
            }
        )
    return families


def _rank_unique_multiple_families(
    families: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        families,
        key=lambda family: (
            -int(family.get("member_count", 0)),
            float(family.get("mean_ratio_error", 0.0)),
            float(family.get("max_ratio_error", 0.0)),
            float(family.get("reference", 0.0)),
        ),
    )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    accepted_sets: list[set[str]] = []
    for family in ranked:
        family_ids = {str(peak_id) for peak_id in family.get("peak_ids", [])}
        key_tuple = tuple(sorted(family_ids))
        if key_tuple in seen:
            continue
        if any(family_ids < accepted for accepted in accepted_sets):
            continue
        seen.add(key_tuple)
        accepted_sets.append(family_ids)
        unique.append(family)
    return unique


@lru_cache(maxsize=1)
def _load_wyckoff_position_data() -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    package = "pymatgen.analysis.prototypes"
    multiplicities_name = "wyckoff-position-multiplicities.json.gz"
    params_name = "wyckoff-position-params.json.gz"
    try:
        multiplicities = _load_gzip_json_resource(package, multiplicities_name)
        parameter_counts = _load_gzip_json_resource(package, params_name)
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Pymatgen Wyckoff-position resources are required for Wyckoff "
            "Setup registration."
        ) from exc
    return (
        {
            group: {letter: int(value) for letter, value in sites.items()}
            for group, sites in multiplicities.items()
            if group != "0"
        },
        {
            group: {letter: int(value) for letter, value in sites.items()}
            for group, sites in parameter_counts.items()
            if group != "0"
        },
    )


def _load_gzip_json_resource(
    package: str,
    resource_name: str,
) -> dict[str, dict[str, int]]:
    resource = resources.files(package).joinpath(resource_name)
    with resources.as_file(resource) as path:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)


@lru_cache(maxsize=230)
def _space_group_symbols(space_group_number: int) -> tuple[str, str]:
    number = int(space_group_number)
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        group = SpaceGroup.from_int_number(number)
        symbol = str(getattr(group, "symbol", f"SG{number}"))
        full_symbol = str(getattr(group, "full_symbol", symbol))
    except Exception:
        symbol = f"SG{number}"
        full_symbol = symbol
    return symbol, full_symbol


def _normalize_crystal_system_name(value: str) -> str:
    lowered = str(value).strip().lower()
    for crystal_system in WYCKOFF_CRYSTAL_SYSTEMS:
        if crystal_system.lower() == lowered:
            return crystal_system
    raise ValueError(f"Unknown crystal system: {value!r}.")


def _default_space_group_for_system(crystal_system: str) -> int:
    system = _normalize_crystal_system_name(crystal_system)
    return DEFAULT_SPACE_GROUP_BY_SYSTEM.get(
        system,
        next(iter(CRYSTAL_SYSTEM_SPACE_GROUP_RANGES[system])),
    )


def _basis_wyckoff_assignments(
    basis_labels: list[str],
    combination: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = list(combination.get("site_labels", []))
    multiplicities = list(combination.get("multiplicities", []))
    assignments = []
    for index, basis in enumerate(basis_labels or ["X"]):
        if labels:
            site_index = index % len(labels)
            site_label = labels[site_index]
            multiplicity = (
                int(multiplicities[site_index])
                if site_index < len(multiplicities)
                else None
            )
        else:
            site_label = ""
            multiplicity = None
        assignments.append(
            {
                "basis": basis,
                "site_label": site_label,
                "multiplicity": multiplicity,
            }
        )
    return assignments


def _basis_labels_from_atom_specs(
    atom_specs: list[dict[str, Any]],
    fallback_atoms: list[str],
) -> list[str]:
    if not atom_specs:
        return list(fallback_atoms)
    labels: list[str] = []
    seen_shared: set[str] = set()
    for spec in atom_specs:
        shared_site = str(spec.get("shared_site") or "").strip()
        if shared_site:
            if shared_site in seen_shared:
                continue
            seen_shared.add(shared_site)
            labels.append(shared_site)
        else:
            labels.append(str(spec.get("element") or "X"))
    return labels


def _wyckoff_combination_penalty(
    combination: dict[str, Any],
    site_count: int,
) -> float:
    if not combination:
        return 0.25
    total_multiplicity = int(combination.get("total_multiplicity", 0))
    free_params = int(combination.get("free_parameter_count", 0))
    count_delta = abs(int(combination.get("site_count", 0)) - site_count)
    return 0.005 * total_multiplicity + 0.015 * free_params + 0.1 * count_delta


def _composition_from_inputs(
    atoms: list[str],
    atom_specs: list[dict[str, Any]],
    molecules: list[dict[str, Any]],
    stoichiometry: str,
) -> dict[str, float]:
    molecule_formulas = {
        str(item.get("label") or item.get("name") or ""): str(
            item.get("formula") or ""
        )
        for item in molecules
        if item.get("label") or item.get("name")
    }
    molecule_formulas.update(
        {
            key: str(value.get("formula", ""))
            for key, value in REFERENCE_MOLECULES.items()
        }
    )
    if stoichiometry.strip():
        parsed = _parse_formula_composition(
            stoichiometry.strip(),
            molecule_formulas,
        )
        if parsed:
            return parsed
    composition: dict[str, float] = {}
    if atom_specs:
        for spec in atom_specs:
            element = _element_symbol(str(spec.get("element") or ""))
            if not element:
                continue
            composition[element] = composition.get(element, 0.0) + max(
                0.0,
                float(spec.get("stoichiometry", 1.0) or 0.0),
            )
    else:
        for atom in atoms:
            element = _element_symbol(atom)
            composition[element] = composition.get(element, 0.0) + 1.0
    for molecule in molecules:
        formula = str(molecule.get("formula") or "")
        for element, count in _parse_formula_composition(
            formula,
            molecule_formulas,
        ).items():
            composition[element] = composition.get(element, 0.0) + count
    return composition or {"X": 1.0}


def _normalized_molecule_list(
    molecules: Iterable[dict[str, Any]],
    stoichiometry: str,
) -> list[dict[str, Any]]:
    molecule_list = [
        _molecule_with_pose_restraints(item) for item in molecules
    ]
    seen = {
        str(item.get("label") or item.get("name") or "").strip()
        for item in molecule_list
    }
    token = _formula_token(stoichiometry)
    for label, metadata in REFERENCE_MOLECULES.items():
        if label in seen:
            continue
        if label.upper() in token:
            item = dict(metadata)
            item["label"] = label
            molecule_list.append(_molecule_with_pose_restraints(item))
            seen.add(label)
    return molecule_list


def _molecule_with_pose_restraints(item: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    label = str(enriched.get("label") or enriched.get("name") or "").strip()
    restraints = MOLECULE_POSE_RESTRAINTS.get(label.upper(), {})
    for key, value in restraints.items():
        enriched.setdefault(key, value)
    return enriched


def _normalize_atom_specs(
    atom_specs: Iterable[dict[str, Any]] | None,
    fallback_atoms: list[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, spec in enumerate(atom_specs or [], start=1):
        if not isinstance(spec, dict):
            continue
        element = _element_symbol(str(spec.get("element") or ""))
        if not element:
            continue
        normalized.append(
            {
                "element": element,
                "stoichiometry": _positive_float(
                    spec.get("stoichiometry", spec.get("count", 1.0)),
                    1.0,
                ),
                "shared_site": str(
                    spec.get("shared_site")
                    or spec.get("site_group")
                    or spec.get("site")
                    or ""
                ).strip(),
                "occupancy": _bounded_float(
                    spec.get("occupancy", 1.0),
                    1.0,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "label": str(spec.get("label") or f"{element}{index}"),
            }
        )
    if normalized:
        return normalized
    return [
        {
            "element": _element_symbol(atom),
            "stoichiometry": 1.0,
            "shared_site": "",
            "occupancy": 1.0,
            "label": f"{_element_symbol(atom)}{index}",
        }
        for index, atom in enumerate(fallback_atoms, start=1)
        if _element_symbol(atom)
    ]


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)


def _bounded_float(
    value: Any,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _parse_formula_composition(
    formula: str,
    molecule_formulas: dict[str, str] | None = None,
) -> dict[str, float]:
    molecule_formulas = molecule_formulas or {}

    def merge(
        target: dict[str, float],
        source: dict[str, float],
        multiplier: float,
    ) -> None:
        for element, count in source.items():
            target[element] = target.get(element, 0.0) + count * multiplier

    def parse_number(index: int) -> tuple[float, int]:
        start = index
        while index < len(formula) and (
            formula[index].isdigit() or formula[index] == "."
        ):
            index += 1
        if start == index:
            return 1.0, index
        try:
            return float(formula[start:index]), index
        except ValueError:
            return 1.0, index

    def parse_until(
        index: int,
        terminator: str = "",
    ) -> tuple[dict[str, float], int]:
        composition: dict[str, float] = {}
        molecule_tokens = sorted(molecule_formulas, key=len, reverse=True)
        while index < len(formula):
            char = formula[index]
            if terminator and char == terminator:
                return composition, index + 1
            if char in "([{":
                close = {"(": ")", "[": "]", "{": "}"}[char]
                group, index = parse_until(index + 1, close)
                multiplier, index = parse_number(index)
                merge(composition, group, multiplier)
                continue
            matched_molecule = ""
            for token in molecule_tokens:
                if token and formula.startswith(token, index):
                    matched_molecule = token
                    break
            if matched_molecule:
                sub = _parse_formula_composition(
                    molecule_formulas[matched_molecule],
                    molecule_formulas,
                )
                index += len(matched_molecule)
                multiplier, index = parse_number(index)
                merge(composition, sub, multiplier)
                continue
            if char.isupper():
                end = index + 1
                if end < len(formula) and formula[end].islower():
                    end += 1
                element = formula[index:end]
                index = end
                multiplier, index = parse_number(index)
                composition[element] = (
                    composition.get(element, 0.0) + multiplier
                )
                continue
            index += 1
        return composition, index

    parsed, _ = parse_until(0)
    return {
        _element_symbol(element): float(count)
        for element, count in parsed.items()
        if count > 0.0
    }


def _format_formula_sum(composition: dict[str, float]) -> str:
    pieces = []
    for element in _composition_order(composition, []):
        count = composition[element]
        count_text = (
            str(int(round(count)))
            if abs(count - round(count)) <= 1.0e-9
            else f"{count:.4g}"
        )
        pieces.append(f"{element}{count_text}")
    return " ".join(pieces)


def _explicit_template_rows(
    composition: dict[str, float],
    space_group: WyckoffSpaceGroupOption,
    stoichiometry: str,
) -> list[tuple[str, str, float, float, float]]:
    if space_group.number != 58:
        return []
    if _formula_token(stoichiometry) not in {
        "DMF2MA2I8PB3",
        "MA2DMF2PB3I8",
    }:
        return []
    required = {
        "C": 8.0,
        "H": 26.0,
        "I": 8.0,
        "N": 4.0,
        "O": 2.0,
        "Pb": 3.0,
    }
    if any(
        abs(composition.get(element, 0.0) - count) > 1.0e-9
        for element, count in required.items()
    ):
        return []
    rows: list[tuple[str, str, float, float, float]] = []
    seen: set[tuple[str, int, int, int]] = set()
    counters: dict[str, int] = {}
    for label, element, x_frac, y_frac, z_frac in MA2_DMF2_PB3I8_TEMPLATE_ROWS:
        for signs, offsets in PNNM_SYMMETRY_OPERATIONS:
            coords = (
                _wrap_fraction(signs[0] * x_frac + offsets[0]),
                _wrap_fraction(signs[1] * y_frac + offsets[1]),
                _wrap_fraction(signs[2] * z_frac + offsets[2]),
            )
            key = (
                element,
                round(coords[0] * 1_000_000),
                round(coords[1] * 1_000_000),
                round(coords[2] * 1_000_000),
            )
            if key in seen:
                continue
            seen.add(key)
            counters[element] = counters.get(element, 0) + 1
            rows.append((f"{label}_{counters[element]:02d}", element, *coords))
    return rows


def _formula_token(formula: str) -> str:
    return "".join(char.upper() for char in formula if char.isalnum())


def _wrap_fraction(value: float) -> float:
    wrapped = float(value) % 1.0
    if abs(wrapped - 1.0) <= 1.0e-9 or abs(wrapped) <= 1.0e-9:
        return 0.0
    return wrapped


def _composition_from_rows(
    rows: list[tuple[str, str, float, float, float]],
) -> dict[str, float]:
    composition: dict[str, float] = {}
    for _, element, *_ in rows:
        composition[element] = composition.get(element, 0.0) + 1.0
    return composition


def _expanded_cif_element_rows(
    composition: dict[str, float],
    *,
    preferred_order: list[str],
    molecule_labels: list[str],
    atom_specs: list[dict[str, Any]],
) -> list[tuple[str, str, str, float, str]]:
    rows: list[tuple[str, str, str, float, str]] = []
    atom_specs_by_element = _atom_specs_by_element(atom_specs)
    if atom_specs_by_element:
        handled_elements: set[str] = set()
        for raw_element in preferred_order:
            element = _element_symbol(raw_element)
            if (
                element not in atom_specs_by_element
                or element in handled_elements
            ):
                continue
            handled_elements.add(element)
            specs = atom_specs_by_element[element]
            total_spec_count = sum(
                max(float(spec.get("stoichiometry", 1.0) or 0.0), 0.0)
                for spec in specs
            )
            composition_count = float(
                composition.get(element, total_spec_count or len(specs))
            )
            for spec_index, spec in enumerate(specs, start=1):
                spec_count = max(
                    float(spec.get("stoichiometry", 1.0) or 0.0),
                    0.0,
                )
                fraction = (
                    spec_count / total_spec_count
                    if total_spec_count > 0.0
                    else 1.0 / max(1, len(specs))
                )
                row_count = max(1, int(round(composition_count * fraction)))
                shared_site = str(spec.get("shared_site") or "").strip()
                occupancy = _bounded_float(
                    spec.get("occupancy", 1.0),
                    1.0,
                    minimum=0.0,
                    maximum=1.0,
                )
                for copy_index in range(1, row_count + 1):
                    label_suffix = (
                        f"{spec_index}_{copy_index}"
                        if row_count > 1 or len(specs) > 1
                        else "1"
                    )
                    rows.append(
                        (
                            f"{element}{label_suffix}",
                            element,
                            shared_site or element,
                            occupancy,
                            shared_site,
                        )
                    )
    order = _composition_order(composition, preferred_order)
    preferred_elements = {_element_symbol(item) for item in preferred_order}
    for element in order:
        if element in atom_specs_by_element:
            continue
        count = max(1, int(round(float(composition[element]))))
        basis = element
        if element not in preferred_elements:
            basis = next(
                (label for label in molecule_labels if label), element
            )
        for index in range(1, count + 1):
            rows.append((f"{element}{index}", element, basis, 1.0, ""))
    return rows or [("X1", "X", "X", 1.0, "")]


def _atom_specs_by_element(
    atom_specs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in atom_specs:
        element = _element_symbol(str(spec.get("element") or ""))
        if not element:
            continue
        grouped.setdefault(element, []).append(spec)
    return grouped


def _occupancy_for_element(
    element: str,
    atom_specs: list[dict[str, Any]],
) -> float:
    specs = _atom_specs_by_element(atom_specs).get(
        _element_symbol(element), []
    )
    if not specs:
        return 1.0
    occupancies = [
        _bounded_float(
            spec.get("occupancy", 1.0),
            1.0,
            minimum=0.0,
            maximum=1.0,
        )
        for spec in specs
    ]
    return float(np.mean(occupancies)) if occupancies else 1.0


def _format_occupancy(value: float) -> str:
    value = _bounded_float(value, 1.0, minimum=0.0, maximum=1.0)
    if abs(value - round(value)) <= 1.0e-9:
        return f"{value:.1f}"
    return f"{value:.6g}"


def _composition_order(
    composition: dict[str, float],
    preferred_order: list[str],
) -> list[str]:
    preferred = []
    for atom in preferred_order:
        element = _element_symbol(atom)
        if element in composition and element not in preferred:
            preferred.append(element)
    hill = []
    if "C" in composition and "C" not in preferred:
        hill.append("C")
    if "H" in composition and "H" not in preferred:
        hill.append("H")
    rest = sorted(
        element
        for element in composition
        if element not in set(preferred + hill)
    )
    return preferred + hill + rest


def _molecule_display_text(
    molecules: list[dict[str, Any]],
    stoichiometry: str,
) -> str:
    labels = [
        str(item.get("label") or item.get("name") or "").strip()
        for item in molecules
        if str(item.get("label") or item.get("name") or "").strip()
    ]
    token = _formula_token(stoichiometry)
    for label in REFERENCE_MOLECULES:
        if label.upper() in token and label not in labels:
            labels.append(label)
    return ", ".join(labels) or "none"


def _draft_fractional_coordinates(
    index: int,
    total: int,
) -> tuple[float, float, float]:
    denominator = max(total, 1)
    base = (index - 1) / denominator
    return (
        base % 1.0,
        ((index * 5) % denominator) / denominator,
        ((index * 7) % denominator) / denominator,
    )


def _symmetry_fractional_sites(
    total: int,
    *,
    pair_offset: int = 0,
    self_offset: int = 0,
) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    """Return deterministic inversion-symmetric unit-cell placement
    sites."""

    count = max(1, int(total))
    sites: list[tuple[np.ndarray, tuple[float, float, float]]] = []
    if count % 2 == 1:
        self_sites = _symmetry_self_inverse_points()
        self_site = self_sites[int(self_offset) % len(self_sites)]
        sites.append((np.asarray(self_site, dtype=float), (1.0, 1.0, 1.0)))
    pair_seeds = _symmetry_pair_seed_points()
    offset = int(pair_offset) % len(pair_seeds)
    for base in pair_seeds[offset:] + pair_seeds[:offset]:
        if len(sites) >= count:
            break
        base_array = np.asarray(base, dtype=float)
        inverse = (-base_array) % 1.0
        sites.append((base_array, (1.0, 1.0, 1.0)))
        if len(sites) >= count:
            break
        sites.append((inverse, (-1.0, -1.0, -1.0)))
    return sites[:count]


def _symmetry_self_inverse_points() -> list[tuple[float, float, float]]:
    return [
        (0.5, 0.5, 0.5),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]


def _symmetry_pair_seed_points() -> list[tuple[float, float, float]]:
    values = (0.125, 0.25, 0.375, 0.2, 0.3, 0.4)
    seeds: list[tuple[float, float, float]] = []
    seen_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for point in product(values, repeat=3):
        inverse = tuple(float((-value) % 1.0) for value in point)
        key = tuple(int(round(value * 10_000.0)) for value in point)
        inverse_key = tuple(int(round(value * 10_000.0)) for value in inverse)
        pair_key = tuple(sorted((key, inverse_key)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        seeds.append(tuple(float(value) for value in point))
    return seeds


def _symmetry_sites_by_element(
    rows: list[tuple[str, str, str, float, str]],
) -> dict[str, list[tuple[np.ndarray, tuple[float, float, float]]]]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for _, element, *_ in rows:
        if element not in counts:
            order.append(element)
        counts[element] = counts.get(element, 0) + 1
    return {
        element: _symmetry_fractional_sites(
            counts[element],
            pair_offset=index * 17,
            self_offset=index,
        )
        for index, element in enumerate(order)
    }


def _inorganic_composition_for_rows(
    composition: dict[str, float],
    atoms: list[str],
    atom_specs: list[dict[str, Any]],
) -> dict[str, float]:
    if atom_specs:
        elements = {
            _element_symbol(str(spec.get("element") or ""))
            for spec in atom_specs
            if spec.get("element")
        }
    else:
        elements = {_element_symbol(atom) for atom in atoms if atom}
    filtered = {
        element: float(composition[element])
        for element in elements
        if element in composition
    }
    if filtered:
        return filtered
    return {
        element: 1.0 for element in elements if element and element != "X"
    } or {"X": 1.0}


def _molecule_counts_from_stoichiometry(
    molecules: list[dict[str, Any]],
    stoichiometry: str,
) -> dict[str, int]:
    formula = stoichiometry.strip()
    counts: dict[str, int] = {}
    for molecule in molecules:
        label = str(
            molecule.get("label") or molecule.get("name") or ""
        ).strip()
        if not label:
            continue
        count = _molecule_label_count(formula, label) if formula else 0
        counts[label] = max(1, int(round(count or 1.0)))
    return counts


def _molecule_label_count(formula: str, label: str) -> float:
    compact = "".join(formula.split())
    if not compact:
        return 0.0
    total = 0.0
    index = 0
    while index < len(compact):
        if compact[index] == "(" and compact.startswith(label, index + 1):
            close_index = index + 1 + len(label)
            if close_index < len(compact) and compact[close_index] == ")":
                multiplier, next_index = _parse_formula_number(
                    compact,
                    close_index + 1,
                )
                total += multiplier
                index = next_index
                continue
        if compact.startswith(label, index):
            multiplier, next_index = _parse_formula_number(
                compact,
                index + len(label),
            )
            total += multiplier
            index = next_index
            continue
        index += 1
    return total


def _parse_formula_number(formula: str, index: int) -> tuple[float, int]:
    start = index
    while index < len(formula) and (
        formula[index].isdigit() or formula[index] == "."
    ):
        index += 1
    if start == index:
        return 1.0, index
    try:
        return float(formula[start:index]), index
    except ValueError:
        return 1.0, index


def _molecular_atom_site_rows(
    candidate: LatticeCandidate,
    molecules: list[dict[str, Any]],
    molecule_counts: dict[str, int],
    *,
    start_index: int,
    occupied_sites: list[tuple[str, np.ndarray]] | None = None,
    assume_unit_cell_symmetry: bool = False,
) -> list[str]:
    if not molecules:
        return []
    rows: list[str] = []
    placed_sites = list(occupied_sites or [])
    total_copies = sum(
        max(1, int(molecule_counts.get(_molecule_label(item), 1)))
        for item in molecules
    )
    copy_index = 0
    for molecule_order, molecule in enumerate(molecules):
        label = _molecule_label(molecule)
        if not label:
            continue
        atom_template = _molecule_atom_template(molecule)
        copy_count = max(1, int(molecule_counts.get(label, 1)))
        symmetry_sites = (
            _symmetry_fractional_sites(
                copy_count,
                pair_offset=start_index + molecule_order * 17,
                self_offset=start_index + molecule_order,
            )
            if assume_unit_cell_symmetry
            else []
        )
        for molecule_copy in range(1, copy_count + 1):
            copy_index += 1
            if assume_unit_cell_symmetry:
                site = symmetry_sites[molecule_copy - 1]
                center = np.asarray(site[0], dtype=float)
                cartesian_transform = site[1]
            else:
                fallback_center = np.asarray(
                    _draft_fractional_coordinates(
                        start_index + copy_index - 1,
                        start_index + total_copies + len(molecules),
                    ),
                    dtype=float,
                )
                center = _best_molecule_seed_center(
                    candidate,
                    atom_template,
                    fallback_center,
                    placed_sites,
                )
                cartesian_transform = (1.0, 1.0, 1.0)
            sites = _positioned_molecule_sites(
                candidate,
                label,
                molecule_copy,
                atom_template,
                center,
                cartesian_transform=cartesian_transform,
            )
            rows.extend(_format_positioned_molecule_rows(sites))
            placed_sites.extend(
                (element, np.asarray((x_frac, y_frac, z_frac), dtype=float))
                for _, element, x_frac, y_frac, z_frac in sites
            )
    return rows


def _best_molecule_seed_center(
    candidate: LatticeCandidate,
    atoms: list[tuple[str, float, float, float]],
    fallback_center: np.ndarray,
    occupied_sites: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    centers = _molecule_seed_center_candidates(fallback_center)
    if not occupied_sites:
        return centers[0]
    lattice = _candidate_lattice_matrix(candidate)
    elements = [element for element, *_ in atoms]
    best_center = centers[0]
    best_score = -float("inf")
    for center in centers:
        coords = _molecule_fractional_coords_from_lattice(
            lattice,
            atoms,
            center,
        )
        score = _molecule_seed_center_score(
            coords,
            elements,
            occupied_sites,
            lattice,
        )
        if score > best_score:
            best_score = score
            best_center = center
    return np.asarray(best_center, dtype=float) % 1.0


def _molecule_seed_center_candidates(
    fallback_center: np.ndarray,
) -> list[np.ndarray]:
    grid = [
        (x, y, z)
        for x in (0.125, 0.375, 0.625, 0.875)
        for y in (0.125, 0.375, 0.625, 0.875)
        for z in (0.125, 0.375, 0.625, 0.875)
    ]
    symmetry_sites = [
        (x, y, z)
        for x in (0.0, 0.25, 0.5, 0.75)
        for y in (0.0, 0.25, 0.5, 0.75)
        for z in (0.0, 0.25, 0.5, 0.75)
        if (x, y, z).count(0.0) + (x, y, z).count(0.5) >= 2
    ]
    centers = [
        np.asarray(fallback_center, dtype=float),
        *map(np.asarray, grid),
    ]
    centers.extend(np.asarray(site, dtype=float) for site in symmetry_sites)
    deduped: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for center in centers:
        wrapped = np.asarray(center, dtype=float) % 1.0
        key = tuple(int(round(value * 1000.0)) for value in wrapped)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(wrapped)
    return deduped


def _molecule_seed_center_score(
    coords: np.ndarray,
    elements: list[str],
    occupied_sites: list[tuple[str, np.ndarray]],
    lattice: np.ndarray,
) -> float:
    min_distance = float("inf")
    clash_penalty = 0.0
    for coord, element in zip(coords, elements, strict=True):
        for occupied_element, occupied_coord in occupied_sites:
            distance = _seed_pbc_distance(coord, occupied_coord, lattice)
            min_distance = min(min_distance, distance)
            minimum = _seed_minimum_contact_distance(
                element,
                occupied_element,
            )
            if distance < minimum:
                heavy_pair = "H" not in {element, occupied_element}
                weight = 1.6 if heavy_pair else 0.8
                clash_penalty += weight * (
                    (minimum - distance) / max(minimum, 1.0e-9)
                )
    if not np.isfinite(min_distance):
        min_distance = 4.0
    return (
        min(min_distance, 4.0)
        - 3.5 * clash_penalty
        - 5.0 * _seed_unit_cell_boundary_penalty(coords)
    )


def _seed_minimum_contact_distance(
    left_element: str,
    right_element: str,
) -> float:
    pair = {str(left_element), str(right_element)}
    inorganic = pair & {"Pb", "Sn", "Ge", "I", "Br", "Cl", "F"}
    if inorganic:
        return 1.50 if "H" in pair else 2.20
    if "H" in pair:
        return 0.75
    return 1.65


def _seed_unit_cell_boundary_penalty(
    coords: np.ndarray,
    *,
    span_limit: float = 0.72,
) -> float:
    if coords.size == 0:
        return 0.0
    wrapped = np.asarray(coords, dtype=float) % 1.0
    spans = np.max(wrapped, axis=0) - np.min(wrapped, axis=0)
    return float(np.sum(np.clip(spans - span_limit, 0.0, None)))


def _molecule_fractional_coords(
    candidate: LatticeCandidate,
    atoms: list[tuple[str, float, float, float]],
    center: np.ndarray,
) -> np.ndarray:
    return _molecule_fractional_coords_from_lattice(
        _candidate_lattice_matrix(candidate),
        atoms,
        center,
    )


def _molecule_fractional_coords_from_lattice(
    matrix: np.ndarray,
    atoms: list[tuple[str, float, float, float]],
    center: np.ndarray,
) -> np.ndarray:
    center_array = np.asarray(center, dtype=float)
    coords = []
    for _, x_cart, y_cart, z_cart in atoms:
        frac_offset = np.linalg.solve(
            matrix.T,
            np.asarray([x_cart, y_cart, z_cart], dtype=float),
        )
        coords.append((center_array + frac_offset) % 1.0)
    return np.asarray(coords, dtype=float)


def _seed_pbc_distance(
    left_frac: np.ndarray,
    right_frac: np.ndarray,
    lattice: np.ndarray,
) -> float:
    delta = np.asarray(left_frac, dtype=float) - np.asarray(
        right_frac,
        dtype=float,
    )
    delta -= np.round(delta)
    return float(np.linalg.norm(delta @ lattice))


def _candidate_lattice_matrix(candidate: LatticeCandidate) -> np.ndarray:
    return np.asarray(
        Lattice(
            a=candidate.a,
            b=candidate.b,
            c=candidate.c,
            alpha=candidate.alpha,
            beta=candidate.beta,
            gamma=candidate.gamma,
        ).vectors(),
        dtype=float,
    )


def _positioned_molecule_sites(
    candidate: LatticeCandidate,
    molecule_label: str,
    molecule_copy: int,
    atoms: list[tuple[str, float, float, float]],
    center: tuple[float, float, float] | np.ndarray,
    *,
    cartesian_transform: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[tuple[str, str, float, float, float]]:
    matrix = _candidate_lattice_matrix(candidate)
    center_array = np.asarray(center, dtype=float)
    transform = np.asarray(cartesian_transform, dtype=float)
    counters: dict[str, int] = {}
    sites = []
    for element, x_cart, y_cart, z_cart in atoms:
        counters[element] = counters.get(element, 0) + 1
        cart_offset = (
            np.asarray([x_cart, y_cart, z_cart], dtype=float) * transform
        )
        frac_offset = np.linalg.solve(matrix.T, cart_offset)
        x_frac, y_frac, z_frac = (
            _wrap_fraction(value) for value in center_array + frac_offset
        )
        safe_label = _formula_token(molecule_label) or "MOL"
        atom_label = (
            f"{safe_label}{molecule_copy}_{element}{counters[element]}"
        )
        sites.append((atom_label, element, x_frac, y_frac, z_frac))
    return sites


def _format_positioned_molecule_rows(
    sites: list[tuple[str, str, float, float, float]],
) -> list[str]:
    rows = []
    for atom_label, element, x_frac, y_frac, z_frac in sites:
        rows.append(
            f"{atom_label} {element} {x_frac:.6f} {y_frac:.6f} "
            f"{z_frac:.6f} 1.0"
        )
    return rows


def _molecule_label(molecule: dict[str, Any]) -> str:
    return str(molecule.get("label") or molecule.get("name") or "").strip()


def _molecule_atom_template(
    molecule: dict[str, Any],
) -> list[tuple[str, float, float, float]]:
    atoms = molecule.get("atoms")
    if not atoms:
        label = _molecule_label(molecule)
        atoms = REFERENCE_MOLECULES.get(label, {}).get("atoms")
    if atoms:
        return [
            (str(atom[0]), float(atom[1]), float(atom[2]), float(atom[3]))
            for atom in atoms
            if len(atom) >= 4
        ]
    formula = str(molecule.get("formula") or "")
    composition = _parse_formula_composition(formula)
    rows: list[tuple[str, float, float, float]] = []
    for element in _composition_order(composition, []):
        count = max(1, int(round(composition[element])))
        for index in range(count):
            angle = 2.0 * np.pi * index / max(count, 1)
            radius = 0.35 + 0.12 * index
            rows.append(
                (
                    element,
                    radius * float(np.cos(angle)),
                    radius * float(np.sin(angle)),
                    0.15 * (index % 3 - 1),
                )
            )
    return rows or [("X", 0.0, 0.0, 0.0)]


def _positioned_molecule_rows(
    candidate: LatticeCandidate,
    molecule_label: str,
    molecule_copy: int,
    atoms: list[tuple[str, float, float, float]],
    center: tuple[float, float, float],
) -> list[str]:
    return _format_positioned_molecule_rows(
        _positioned_molecule_sites(
            candidate,
            molecule_label,
            molecule_copy,
            atoms,
            center,
        )
    )


def _element_symbol(symbol: str) -> str:
    letters = "".join(char for char in str(symbol) if char.isalpha())
    if not letters:
        return str(symbol)
    if len(letters) == 1:
        return letters.upper()
    return f"{letters[0].upper()}{letters[1:].lower()}"


def _draft_cif_text(
    candidate: LatticeCandidate,
    cif_id: str,
    atoms: list[str],
    molecules: list[dict[str, Any]],
    stoichiometry: str,
    *,
    space_group: WyckoffSpaceGroupOption,
    wyckoff_combination: dict[str, Any],
    wyckoff_assignments: list[dict[str, Any]],
    composition: dict[str, float] | None = None,
    atom_specs: list[dict[str, Any]] | None = None,
    allow_explicit_template: bool = True,
    assume_unit_cell_symmetry: bool = False,
) -> str:
    params = candidate.as_parameters()
    site_labels = ", ".join(wyckoff_combination.get("site_labels", []))
    composition = composition or {atom: 1.0 for atom in atoms}
    template_rows = (
        _explicit_template_rows(
            composition,
            space_group,
            stoichiometry,
        )
        if allow_explicit_template
        else []
    )
    if template_rows:
        return _draft_explicit_template_cif_text(
            candidate,
            cif_id,
            stoichiometry,
            space_group=space_group,
            wyckoff_site_labels=site_labels,
            molecules=molecules,
            rows=template_rows,
            atom_specs=atom_specs or [],
        )
    lines = [
        f"data_{cif_id}",
        "_symmetry_space_group_name_H-M 'P1'",
        "_space_group_IT_number 1",
        (
            f"# inferred parent space group: {space_group.symbol} "
            f"({space_group.number})"
        ),
        f"_chemical_formula_sum '{_format_formula_sum(composition)}'",
        f"_cell_length_a {params.a:.6f}",
        f"_cell_length_b {params.b:.6f}",
        f"_cell_length_c {params.c:.6f}",
        f"_cell_angle_alpha {params.alpha:.6f}",
        f"_cell_angle_beta {params.beta:.6f}",
        f"_cell_angle_gamma {params.gamma:.6f}",
        f"# composition constraint: {stoichiometry or 'unspecified'}",
        f"# Wyckoff combination: {site_labels or 'unassigned'}",
        "# molecular species: "
        + _molecule_display_text(molecules, stoichiometry),
        (
            "# unit-cell placement: inversion-symmetric atom and molecule "
            "centers"
            if assume_unit_cell_symmetry
            else "# explicit full-composition draft; parent symmetry is advisory"
        ),
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    atom_entries = atoms or ["X"]
    molecule_counts = _molecule_counts_from_stoichiometry(
        molecules,
        stoichiometry,
    )
    inorganic_composition = _inorganic_composition_for_rows(
        composition,
        atom_entries,
        atom_specs or [],
    )
    element_rows = _expanded_cif_element_rows(
        inorganic_composition,
        preferred_order=atom_entries,
        molecule_labels=[],
        atom_specs=atom_specs or [],
    )
    symmetry_sites = (
        _symmetry_sites_by_element(element_rows)
        if assume_unit_cell_symmetry
        else {}
    )
    symmetry_indices: dict[str, int] = {}
    shared_coordinates: dict[str, tuple[float, float, float]] = {}
    occupied_sites: list[tuple[str, np.ndarray]] = []
    for index, row in enumerate(element_rows, start=1):
        label, atom, basis, occupancy, shared_site = row
        if assume_unit_cell_symmetry:
            element_sites = symmetry_sites.get(atom) or []
            site_index = symmetry_indices.get(atom, 0)
            symmetry_indices[atom] = site_index + 1
            center = element_sites[site_index][0]
            x_frac, y_frac, z_frac = (
                float(value) for value in np.asarray(center, dtype=float)
            )
        elif shared_site:
            if shared_site not in shared_coordinates:
                shared_coordinates[shared_site] = (
                    _draft_fractional_coordinates(
                        index,
                        len(element_rows),
                    )
                )
            x_frac, y_frac, z_frac = shared_coordinates[shared_site]
        else:
            x_frac, y_frac, z_frac = _draft_fractional_coordinates(
                index,
                len(element_rows),
            )
        lines.append(
            f"{label} {atom} {x_frac:.6f} {y_frac:.6f} {z_frac:.6f} "
            f"{_format_occupancy(occupancy)}"
        )
        occupied_sites.append(
            (atom, np.asarray((x_frac, y_frac, z_frac), dtype=float))
        )
    molecular_rows = _molecular_atom_site_rows(
        candidate,
        molecules,
        molecule_counts,
        start_index=len(element_rows) + 1,
        occupied_sites=occupied_sites,
        assume_unit_cell_symmetry=assume_unit_cell_symmetry,
    )
    lines.extend(molecular_rows)
    return "\n".join(lines) + "\n"


def _draft_explicit_template_cif_text(
    candidate: LatticeCandidate,
    cif_id: str,
    stoichiometry: str,
    *,
    space_group: WyckoffSpaceGroupOption,
    wyckoff_site_labels: str,
    molecules: list[dict[str, Any]],
    rows: list[tuple[str, str, float, float, float]],
    atom_specs: list[dict[str, Any]] | None = None,
) -> str:
    """Write a full-cell draft to avoid accidental symmetry copies."""

    params = candidate.as_parameters()
    composition = _composition_from_rows(rows)
    lines = [
        f"data_{cif_id}",
        "_symmetry_space_group_name_H-M 'P1'",
        "_space_group_IT_number 1",
        (
            f"# inferred parent space group: {space_group.symbol} "
            f"({space_group.number})"
        ),
        f"_chemical_formula_sum '{_format_formula_sum(composition)}'",
        f"_cell_length_a {params.a:.6f}",
        f"_cell_length_b {params.b:.6f}",
        f"_cell_length_c {params.c:.6f}",
        f"_cell_angle_alpha {params.alpha:.6f}",
        f"_cell_angle_beta {params.beta:.6f}",
        f"_cell_angle_gamma {params.gamma:.6f}",
        f"# composition constraint: {stoichiometry or 'unspecified'}",
        f"# Wyckoff combination: {wyckoff_site_labels or 'unassigned'}",
        "# molecular species: "
        + _molecule_display_text(molecules, stoichiometry),
        "# explicit full unit cell generated from full MA and DMF motifs",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for label, element, x_frac, y_frac, z_frac in rows:
        occupancy = _occupancy_for_element(element, atom_specs or [])
        lines.append(
            f"{label} {element} {x_frac:.6f} {y_frac:.6f} "
            f"{z_frac:.6f} {_format_occupancy(occupancy)}"
        )
    return "\n".join(lines) + "\n"


def _peak_id(record: dict[str, Any]) -> str:
    return str(record.get("peak_id") or record.get("id") or "peak")


def _peak_qxy(record: dict[str, Any]) -> float:
    return float(record.get("qxy", record.get("qx", record.get("x", 0.0))))


def _peak_qz(record: dict[str, Any]) -> float:
    return float(record.get("qz", record.get("y", 0.0)))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
