"""Synthetic experimental-refinement diagnostics.

This workflow uses the reference CIFs only to create fiber-textured mock
measurements and to score the blind result afterward.  The solver receives the
same constrained inputs as the experimental refinement path: allowed inorganic
elements plus allowed molecule labels.
"""

from __future__ import annotations

import math
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ewald.analysis.structure import (
    DEFAULT_PHASE_TAG,
    REFERENCE_MOLECULES,
    LatticeCandidate,
    StructurePeak,
    generate_ranked_cif_records,
    group_peak_families,
    guess_lattice_candidates,
)
from ewald.benchmark.experimental_refinement import (
    DEFAULT_EXPERIMENTAL_OUTPUT_DIR,
    _expand_axis_scale_candidates,
    _formula_charge_balance,
    _perovskite_motif_metadata_by_formula,
    chemistry_stoichiometry_hypotheses,
    perovskite_scaffold_hypotheses,
)
from ewald.benchmark.structure_benchmark import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkStructureSpec,
    _append_logbook,
    _candidate_search_config,
    _cif_physical_chemistry_metrics,
    _comparison_sort_key,
    _composition_validation_metrics,
    _detect_structure_peaks,
    _fractional_center,
    _lattice_validation_metrics,
    _materialize_generated_cifs,
    _mock_experimental_qspace,
    _molecular_body_token,
    _molecule_records,
    _pair_distribution_validation_metrics,
    _path_chemistry_rank_fields,
    _peak_detection_recovery_metrics,
    _read_simple_cif_sites,
    _save_peak_detection_plot,
    _score_bragg_peak_intensity_match,
    _slug,
    _solve_constraints,
    _timestamp,
    _write_filtered_cif,
    _write_json,
    _write_lattice_disordered_cif,
    _write_mock_tiff,
    _write_physicalized_cif,
    _write_qspace_npz,
    _write_truth_peak_table,
)
from ewald.crystallography.cif import (
    compare_cif_atom_coordinates,
    extract_cif_lattice_parameters,
    infer_crystal_system_from_lattice,
)
from ewald.data.models import (
    PEAK_HKL_METADATA_KEY,
    STRUCTURE_ANALYSIS_KEY,
    ImageCorrectionState,
    ProjectState,
)
from ewald.io.project import save_project
from ewald.simulation.giwaxs import (
    GIWAXSSimulationParameters,
    calculate_giwaxs_peak_rows,
    compare_giwaxs_images,
    save_giwaxs_comparison_plot,
    simulate_giwaxs_image,
)

_ATOMIC_NUMBER_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
)
_ATOMIC_NUMBERS = {
    symbol: atomic_number
    for atomic_number, symbol in enumerate(_ATOMIC_NUMBER_SYMBOLS, start=1)
}
_ORGANIC_ELECTRON_PROXY_ELEMENTS = (
    "H",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
)
_ORGANIC_ELECTRON_PROXY_NUMBERS = {
    symbol: _ATOMIC_NUMBERS[symbol]
    for symbol in _ORGANIC_ELECTRON_PROXY_ELEMENTS
}
_ORGANIC_MOLECULE_NOMINAL_CHARGES = {
    "MA": 1.0,
    "FA": 1.0,
    "BA": 1.0,
    "DMF": 0.0,
    "DMSO": 0.0,
    "NMP": 0.0,
}


@dataclass(slots=True)
class SyntheticRefinementConfig(BenchmarkRunConfig):
    """Controls for synthetic refinement diagnostics."""

    output_dir: Path = DEFAULT_EXPERIMENTAL_OUTPUT_DIR
    seed: int = 20260519
    simulations_per_structure: int = 2
    detector_shape: tuple[int, int] = (128, 192)
    hkl_extent: int = 5
    peak_threshold_percentile: float = 99.55
    peak_max_peaks: int = 120
    candidate_hkl_max: int = 6
    candidate_q_tolerance: float = 0.07
    candidate_relative_tolerance: float = 0.045
    candidate_grid_points: int = 10
    candidate_max_candidates: int = 8
    max_generated_cifs_to_compare: int = 6
    comparison_plot_count: int = 1
    staged_refinement: bool = False
    family_tolerance: float = 0.045
    family_ratio_tolerance: float = 0.06
    records_per_stoichiometry: int = 1
    max_stoichiometry_hypotheses: int = 8
    max_scaffolds_to_validate: int = 10
    generate_organic_electron_proxies: bool = True
    max_organic_proxy_cifs_to_compare: int = 8
    generate_organic_replacement_structures: bool = True
    max_organic_replacement_cifs_to_compare: int = 6
    stage_simulation_max_cifs: int = 6
    organic_rmc_steps: int = 12
    organic_rmc_translation_step: float = 0.025
    organic_rmc_rotation_step_deg: float = 8.0
    assume_unit_cell_symmetry: bool = True
    candidate_axis_scale_variants: tuple[float, ...] = (1.0, 0.5, 2.0)
    rank_generated_cifs_with_image_fit: bool = False
    texture_modes: tuple[str, ...] = (
        "out_of_plane_stack",
        "in_plane_stack",
        "tilted_out_of_plane_stack",
        "tilted_in_plane_stack",
    )
    texture_azimuth_jitter_deg: float = 6.0
    oracle_lattice_parameters: bool = False
    oracle_peak_positions: bool = False
    oracle_stoichiometry_constraints: bool = False

    def __post_init__(self) -> None:
        BenchmarkRunConfig.__post_init__(self)
        self.family_tolerance = max(1.0e-9, float(self.family_tolerance))
        self.family_ratio_tolerance = max(
            1.0e-9,
            float(self.family_ratio_tolerance),
        )
        self.records_per_stoichiometry = max(
            1,
            int(self.records_per_stoichiometry),
        )
        self.max_stoichiometry_hypotheses = max(
            1,
            int(self.max_stoichiometry_hypotheses),
        )
        self.max_scaffolds_to_validate = max(
            1,
            int(self.max_scaffolds_to_validate),
        )
        self.generate_organic_electron_proxies = bool(
            self.generate_organic_electron_proxies
        )
        self.max_organic_proxy_cifs_to_compare = max(
            0,
            int(self.max_organic_proxy_cifs_to_compare),
        )
        self.generate_organic_replacement_structures = bool(
            self.generate_organic_replacement_structures
        )
        self.max_organic_replacement_cifs_to_compare = max(
            0,
            int(self.max_organic_replacement_cifs_to_compare),
        )
        self.stage_simulation_max_cifs = max(
            1,
            int(self.stage_simulation_max_cifs),
        )
        self.organic_rmc_steps = max(0, int(self.organic_rmc_steps))
        self.organic_rmc_translation_step = max(
            0.0,
            float(self.organic_rmc_translation_step),
        )
        self.organic_rmc_rotation_step_deg = max(
            0.0,
            float(self.organic_rmc_rotation_step_deg),
        )
        self.assume_unit_cell_symmetry = bool(self.assume_unit_cell_symmetry)
        self.candidate_axis_scale_variants = tuple(
            float(value)
            for value in self.candidate_axis_scale_variants
            if float(value) > 0.0
        ) or (1.0,)
        self.texture_modes = tuple(
            _normalize_texture_mode(mode)
            for mode in self.texture_modes
            if str(mode).strip()
        ) or ("out_of_plane_stack", "in_plane_stack")
        self.texture_azimuth_jitter_deg = max(
            0.0,
            float(self.texture_azimuth_jitter_deg),
        )
        self.oracle_lattice_parameters = bool(self.oracle_lattice_parameters)
        self.oracle_peak_positions = bool(self.oracle_peak_positions)
        self.oracle_stoichiometry_constraints = bool(
            self.oracle_stoichiometry_constraints
        )

    def as_dict(self) -> dict[str, Any]:
        payload = BenchmarkRunConfig.as_dict(self)
        payload.update(
            {
                "family_tolerance": self.family_tolerance,
                "family_ratio_tolerance": self.family_ratio_tolerance,
                "records_per_stoichiometry": self.records_per_stoichiometry,
                "max_stoichiometry_hypotheses": (
                    self.max_stoichiometry_hypotheses
                ),
                "max_scaffolds_to_validate": self.max_scaffolds_to_validate,
                "generate_organic_electron_proxies": (
                    self.generate_organic_electron_proxies
                ),
                "max_organic_proxy_cifs_to_compare": (
                    self.max_organic_proxy_cifs_to_compare
                ),
                "generate_organic_replacement_structures": (
                    self.generate_organic_replacement_structures
                ),
                "max_organic_replacement_cifs_to_compare": (
                    self.max_organic_replacement_cifs_to_compare
                ),
                "stage_simulation_max_cifs": self.stage_simulation_max_cifs,
                "organic_rmc_steps": self.organic_rmc_steps,
                "organic_rmc_translation_step": (
                    self.organic_rmc_translation_step
                ),
                "organic_rmc_rotation_step_deg": (
                    self.organic_rmc_rotation_step_deg
                ),
                "assume_unit_cell_symmetry": self.assume_unit_cell_symmetry,
                "candidate_axis_scale_variants": list(
                    self.candidate_axis_scale_variants
                ),
                "rank_generated_cifs_with_image_fit": (
                    self.rank_generated_cifs_with_image_fit
                ),
                "texture_modes": list(self.texture_modes),
                "texture_azimuth_jitter_deg": (
                    self.texture_azimuth_jitter_deg
                ),
                "oracle_lattice_parameters": self.oracle_lattice_parameters,
                "oracle_peak_positions": self.oracle_peak_positions,
                "oracle_stoichiometry_constraints": (
                    self.oracle_stoichiometry_constraints
                ),
            }
        )
        return payload


@dataclass(slots=True)
class SyntheticRefinementResult(BenchmarkRunResult):
    """Summary for one synthetic refinement-diagnostic batch."""

    aggregate: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = BenchmarkRunResult.as_dict(self)
        payload["aggregate"] = self.aggregate
        return payload


def run_synthetic_refinement_pipeline(
    specs: Iterable[BenchmarkStructureSpec],
    config: SyntheticRefinementConfig | None = None,
) -> SyntheticRefinementResult:
    """Run synthetic fiber-textured refinement diagnostics for CIF
    specs."""

    cfg = config or SyntheticRefinementConfig()
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_root = cfg.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    logbook = run_root / "LOGBOOK.md"
    rng = np.random.default_rng(cfg.seed)
    normalized_specs = [spec for spec in specs if spec.cif_path.exists()]
    tasks = [
        (spec, simulation_index)
        for spec in normalized_specs
        for simulation_index in range(
            1,
            max(1, cfg.simulations_per_structure) + 1,
        )
    ]
    rng.shuffle(tasks)
    result = SyntheticRefinementResult(run_id=run_id, output_dir=run_root)
    _progress(
        "Synthetic refinement "
        f"{run_id}: {len(tasks)} filesets, "
        f"detector={cfg.detector_shape[0]}x{cfg.detector_shape[1]}, "
        f"hkl_extent={cfg.hkl_extent}"
    )
    _write_json(
        run_root / "synthetic_refinement_manifest.json",
        {
            "run_id": run_id,
            "created_at": _timestamp(),
            "config": cfg.as_dict(),
            "structures": [spec.as_dict() for spec in normalized_specs],
            "solver_input_policy": (
                "Reference CIFs generate synthetic data and post-hoc truth "
                "metrics only. Lattice, stoichiometry, hkl, and coordinates "
                "are held out from the blind solver unless an explicit oracle "
                "diagnostic flag is enabled. Oracle diagnostics may provide "
                "lattice/peak/hkl/stoichiometry constraints, but atom "
                "coordinates are still not copied into generated structures."
            ),
        },
    )
    _append_logbook(
        logbook,
        [
            f"# EWALD Synthetic Refinement Diagnostics {run_id}",
            "",
            f"- Created: {_timestamp()}",
            f"- Seed: {cfg.seed}",
            f"- Structure count: {len(normalized_specs)}",
            f"- Simulations per structure: {cfg.simulations_per_structure}",
            (
                "- Texture: controlled in-plane/out-of-plane stacking "
                f"families: {', '.join(cfg.texture_modes)}"
            ),
            "",
        ],
    )
    for order_index, (spec, simulation_index) in enumerate(tasks, start=1):
        fileset = _run_one_synthetic_fileset(
            spec,
            simulation_index,
            order_index,
            run_root,
            logbook,
            rng,
            cfg,
        )
        result.filesets.append(fileset)
        result.aggregate = _aggregate_synthetic_findings(result.filesets)
        _write_json(run_root / "summary.json", result.as_dict())
        _progress(
            f"[{order_index}/{len(tasks)}] completed "
            f"{fileset.get('fileset_id')}: "
            f"peaks={fileset.get('peak_count')}, "
            "recall="
            f"{_format_progress_float(fileset.get('peak_recovery', {}).get('recall'))}, "
            "scaffold_rms="
            f"{_format_progress_float(_fileset_scaffold_rms(fileset))}"
        )
    result.aggregate = _aggregate_synthetic_findings(result.filesets)
    _write_json(run_root / "summary.json", result.as_dict())
    _append_logbook(
        logbook,
        [
            "",
            "## Aggregate Findings",
            "",
            *_aggregate_logbook_lines(result.aggregate),
        ],
    )
    _progress(f"Synthetic refinement {run_id}: finished")
    return result


def _progress(message: str) -> None:
    print(f"[synthetic-refine] {_timestamp()} {message}", flush=True)


def _format_progress_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.3f}"


def _fileset_scaffold_rms(fileset: dict[str, Any]) -> Any:
    validation = fileset.get("best_inorganic_scaffold_validation", {})
    if not isinstance(validation, dict):
        return None
    comparison = validation.get("cif_comparison", {})
    if not isinstance(comparison, dict):
        return None
    coordinate = comparison.get("coordinate_match", {})
    if not isinstance(coordinate, dict):
        return None
    return coordinate.get("cartesian_rms_angstrom")


def _oracle_structure_peaks_from_truth(
    truth_peaks: list[dict[str, Any]],
) -> list[StructurePeak]:
    """Convert synthetic truth Bragg rows into exact solver peak
    inputs."""

    peaks: list[StructurePeak] = []
    for index, truth in enumerate(truth_peaks, start=1):
        hkl = _truth_hkl(truth)
        intensity = _truth_peak_weight(truth, "intensity")
        amplitude = _truth_peak_weight(truth, "amplitude")
        observed_intensity = intensity if intensity > 0.0 else amplitude
        hkl_label = f"({hkl[0]} {hkl[1]} {hkl[2]})"
        peaks.append(
            StructurePeak(
                peak_id=f"oracle_peak_{index:04d}",
                label=f"Oracle {hkl_label}",
                qxy=abs(float(truth.get("qxy", 0.0))),
                qz=float(truth.get("qz", 0.0)),
                source="synthetic_truth_oracle",
                phase_tag=DEFAULT_PHASE_TAG,
                hkl_label=hkl_label,
                include=True,
                fit_quality=observed_intensity,
                status="oracle_truth_peak",
                notes=(
                    "Exact synthetic truth peak used for diagnostic solver "
                    "input; atom coordinates remain hidden."
                ),
                metadata={
                    PEAK_HKL_METADATA_KEY: {
                        "h": hkl[0],
                        "k": hkl[1],
                        "l": hkl[2],
                    },
                    "hkl": list(hkl),
                    "hkl_family": _plane_family_key(hkl),
                    "truth_index": int(index - 1),
                    "truth_qxy_signed": float(truth.get("qxy", 0.0)),
                    "truth_qxy_abs": abs(float(truth.get("qxy", 0.0))),
                    "truth_qz": float(truth.get("qz", 0.0)),
                    "amplitude": amplitude,
                    "intensity": intensity,
                    "integrated_intensity": observed_intensity,
                    "oracle_truth_peak": True,
                },
            )
        )
    peaks.sort(
        key=lambda peak: float(
            peak.metadata.get("integrated_intensity", peak.fit_quality or 0.0)
        ),
        reverse=True,
    )
    for index, peak in enumerate(peaks, start=1):
        peak.peak_id = f"oracle_peak_{index:04d}"
    return peaks


def _oracle_peak_families_from_peaks(
    peaks: list[StructurePeak],
) -> list[dict[str, Any]]:
    """Group oracle peaks by their exact synthetic hkl family."""

    grouped: dict[str, list[StructurePeak]] = {}
    for peak in peaks:
        hkl = _parse_hkl_triplet(peak.metadata.get("hkl") or peak.hkl_label)
        family_key = (
            _plane_family_key(hkl)
            if hkl is not None
            else str(peak.metadata.get("hkl_family") or "unknown")
        )
        grouped.setdefault(family_key, []).append(peak)
    families = []
    for index, (family_key, members) in enumerate(
        sorted(grouped.items()),
        start=1,
    ):
        intensities = [
            float(
                member.metadata.get(
                    "integrated_intensity",
                    member.fit_quality or 0.0,
                )
            )
            for member in members
        ]
        families.append(
            {
                "family_id": f"oracle_hkl_family_{index:03d}",
                "kind": "oracle hkl family",
                "phase_tag": DEFAULT_PHASE_TAG,
                "reference": family_key,
                "hkl_family": family_key,
                "member_count": len(members),
                "peak_ids": [member.peak_id for member in members],
                "labels": [member.label for member in members],
                "mean_qxy": float(np.mean([member.qxy for member in members])),
                "mean_qz": float(np.mean([member.qz for member in members])),
                "integrated_intensity_sum": float(np.sum(intensities)),
                "confidence": 1.0,
                "reason": "exact synthetic truth hkl family",
                "notes": (
                    "Oracle diagnostic family; hkl grouping supplied from "
                    "truth table without using reference atom coordinates."
                ),
            }
        )
    families.sort(
        key=lambda item: (
            -float(item.get("integrated_intensity_sum", 0.0)),
            str(item.get("hkl_family")),
        )
    )
    return families


def _oracle_lattice_candidate_from_reference(
    spec: BenchmarkStructureSpec,
    peaks: list[StructurePeak],
) -> LatticeCandidate | None:
    """Return the exact reference cell as a diagnostic lattice
    candidate."""

    try:
        lattice = extract_cif_lattice_parameters(spec.cif_path)
    except Exception:
        return None
    crystal_system = infer_crystal_system_from_lattice(lattice)
    assignments = []
    for peak in peaks:
        hkl = _parse_hkl_triplet(peak.metadata.get("hkl") or peak.hkl_label)
        if hkl is None or hkl == (0, 0, 0):
            continue
        assignments.append(
            {
                "peak_id": peak.peak_id,
                "label": peak.label,
                "q_observed": float(peak.q_magnitude),
                "qxy_observed": float(peak.qxy),
                "qz_observed": float(peak.qz),
                "projected_delta_q": 0.0,
                "delta_q": 0.0,
                "abs_delta_q": 0.0,
                "hkl": list(hkl),
                "hkl_family": _plane_family_key(hkl),
                "source": "oracle_truth_hkl",
            }
        )
    return LatticeCandidate(
        candidate_id="oracle_reference_lattice",
        crystal_system=crystal_system,
        a=float(lattice["a"]),
        b=float(lattice["b"]),
        c=float(lattice["c"]),
        alpha=float(lattice["alpha"]),
        beta=float(lattice["beta"]),
        gamma=float(lattice["gamma"]),
        score=0.0,
        rms_error=0.0,
        matched_count=len(assignments),
        outlier_count=0,
        method="oracle_reference_lattice_parameters",
        assignments=assignments,
        notes=(
            "Exact reference lattice and truth hkl assignments for diagnostic "
            "upper-bound test; atom coordinates are not used."
        ),
    )


def _oracle_stoichiometry_hypotheses(
    spec: BenchmarkStructureSpec,
    fallback: list[str],
) -> list[str]:
    """Prepend exact reference stoichiometry hypotheses without
    coordinates."""

    exact = _reference_stoichiometry_hypotheses(spec)
    return list(dict.fromkeys([*exact, *fallback]))


def _reference_stoichiometry_hypotheses(
    spec: BenchmarkStructureSpec,
) -> list[str]:
    counts = _reference_composition_counts(spec.cif_path)
    if not counts:
        return []
    inorganic_order = [
        _element_symbol(atom)
        for atom in spec.inorganic_atoms
        if _element_symbol(atom)
    ]
    inorganic_counts = {
        element: counts.get(element, 0.0) for element in inorganic_order
    }
    inorganic_formula = _formula_from_counts_ordered(
        inorganic_counts,
        inorganic_order,
    )
    molecule_counts = _infer_reference_molecule_counts(
        counts,
        spec.organic_molecules,
        inorganic_order,
    )
    hypotheses = []
    if molecule_counts and inorganic_formula:
        hypotheses.append(
            _molecule_formula_from_counts(
                molecule_counts,
                spec.organic_molecules,
            )
            + inorganic_formula
        )
    if inorganic_formula:
        hypotheses.append(inorganic_formula)
    for formula in list(hypotheses):
        reduced = _reduced_stoichiometry_formula(
            formula,
            spec.organic_molecules,
            inorganic_order,
        )
        if reduced and reduced != formula:
            hypotheses.append(reduced)
    return list(dict.fromkeys(item for item in hypotheses if item))


def _reference_composition_counts(path: Path) -> dict[str, float]:
    try:
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(str(path))
        return {
            _element_symbol(element): float(count)
            for element, count in structure.composition.as_dict().items()
        }
    except Exception:
        rows, _ = _read_simple_cif_sites(path)
        counts: dict[str, float] = {}
        for row in rows:
            element = _element_symbol(str(row.get("element") or ""))
            if not element:
                continue
            counts[element] = counts.get(element, 0.0) + 1.0
        return counts


def _infer_reference_molecule_counts(
    element_counts: dict[str, float],
    organic_molecules: Iterable[str],
    inorganic_elements: Iterable[str],
) -> dict[str, float]:
    labels = [
        _canonical_molecule_label(label)
        for label in organic_molecules
        if _canonical_molecule_label(label)
    ]
    if not labels:
        return {}
    inorganic = {_element_symbol(element) for element in inorganic_elements}
    molecule_formulas = {
        label: _parse_element_formula_counts(
            str(_reference_molecule_metadata(label).get("formula") or "")
        )
        for label in labels
    }
    molecule_formulas = {
        label: counts for label, counts in molecule_formulas.items() if counts
    }
    if not molecule_formulas:
        return {}
    elements = sorted(
        {
            element
            for counts in molecule_formulas.values()
            for element in counts
            if element not in inorganic
        }
    )
    if not elements:
        return {}
    matrix = np.asarray(
        [
            [molecule_formulas[label].get(element, 0.0) for label in labels]
            for element in elements
        ],
        dtype=float,
    )
    target = np.asarray(
        [float(element_counts.get(element, 0.0)) for element in elements],
        dtype=float,
    )
    if matrix.size == 0 or float(np.linalg.norm(target)) <= 1.0e-12:
        return {}
    try:
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return {}
    predicted = matrix @ solution
    residual = float(np.linalg.norm(predicted - target))
    relative = residual / max(float(np.linalg.norm(target)), 1.0)
    if relative > 0.15:
        return {}
    counts: dict[str, float] = {}
    for label, raw_value in zip(labels, solution, strict=True):
        value = max(0.0, float(raw_value))
        rounded = round(value * 4.0) / 4.0
        if rounded > 1.0e-9:
            counts[label] = rounded
    return counts


def _molecule_formula_from_counts(
    counts: dict[str, float],
    preferred_order: Iterable[str],
) -> str:
    pieces = []
    for raw_label in preferred_order:
        label = _canonical_molecule_label(raw_label)
        count = float(counts.get(label, 0.0))
        if count <= 0.0:
            continue
        pieces.append(f"{label}{_format_formula_count(count)}")
    return "".join(pieces)


def _formula_from_counts_ordered(
    counts: dict[str, float],
    preferred_order: Iterable[str],
) -> str:
    pieces = []
    seen: set[str] = set()
    for raw_element in preferred_order:
        element = _element_symbol(raw_element)
        seen.add(element)
        count = float(counts.get(element, 0.0))
        if count > 0.0:
            pieces.append(f"{element}{_format_formula_count(count)}")
    for element in sorted(counts):
        symbol = _element_symbol(element)
        if symbol in seen:
            continue
        count = float(counts.get(symbol, counts.get(element, 0.0)))
        if count > 0.0:
            pieces.append(f"{symbol}{_format_formula_count(count)}")
    return "".join(pieces)


def _reduced_stoichiometry_formula(
    formula: str,
    organic_molecules: Iterable[str],
    inorganic_order: Iterable[str],
) -> str:
    molecule_counts = _organic_molecule_counts_from_hypothesis(
        formula,
        organic_molecules,
    )
    inorganic_counts = _parse_element_formula_counts(
        _inorganic_formula_from_hypothesis(
            formula,
            inorganic_order,
            organic_molecules,
        )
    )
    combined = [
        *molecule_counts.values(),
        *inorganic_counts.values(),
    ]
    if not combined:
        return ""
    scale = 4
    integers = [int(round(float(value) * scale)) for value in combined]
    if any(
        abs(integer / scale - float(value)) > 1.0e-6
        for integer, value in zip(integers, combined, strict=True)
    ):
        return formula
    divisor = 0
    for integer in integers:
        divisor = math.gcd(divisor, abs(integer))
    if divisor <= scale:
        return formula
    reduced_molecules = {
        label: count * scale / divisor
        for label, count in molecule_counts.items()
    }
    reduced_inorganic = {
        element: count * scale / divisor
        for element, count in inorganic_counts.items()
    }
    return _molecule_formula_from_counts(
        reduced_molecules, organic_molecules
    ) + _formula_from_counts_ordered(reduced_inorganic, inorganic_order)


def _scheduled_texture_mode(
    cfg: SyntheticRefinementConfig,
    order_index: int,
) -> str:
    modes = cfg.texture_modes or ("out_of_plane_stack", "in_plane_stack")
    return modes[(max(1, int(order_index)) - 1) % len(modes)]


def _synthetic_texture_parameters(
    rng: np.random.Generator,
    cfg: SyntheticRefinementConfig,
    texture_mode: str,
) -> GIWAXSSimulationParameters:
    mode = _normalize_texture_mode(texture_mode)
    theta_x_center, azimuths = _texture_orientation_centers(mode)
    theta_x = theta_x_center + float(
        rng.uniform(-cfg.fiber_tilt_jitter_deg, cfg.fiber_tilt_jitter_deg)
    )
    theta_y_center = float(azimuths[int(rng.integers(0, len(azimuths)))])
    theta_y = theta_y_center + float(
        rng.uniform(
            -cfg.texture_azimuth_jitter_deg,
            cfg.texture_azimuth_jitter_deg,
        )
    )
    theta_x = float(np.clip(theta_x, 0.0, 180.0))
    theta_y = float(theta_y % 360.0)
    return GIWAXSSimulationParameters(
        sigma_theta=cfg.sigma_theta,
        sigma_phi=cfg.sigma_phi,
        sigma_r=cfg.sigma_r,
        hkl_extent=cfg.hkl_extent,
        theta_x_deg=theta_x,
        theta_y_deg=theta_y,
        qxy_min=cfg.qxy_range[0],
        qxy_max=cfg.qxy_range[1],
        qz_min=cfg.qz_range[0],
        qz_max=cfg.qz_range[1],
        resolution_z=cfg.detector_shape[0],
        resolution_x=cfg.detector_shape[1],
    )


def _normalize_texture_mode(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "oop": "out_of_plane_stack",
        "out_of_plane": "out_of_plane_stack",
        "oop_stack": "out_of_plane_stack",
        "outofplane": "out_of_plane_stack",
        "ip": "in_plane_stack",
        "in_plane": "in_plane_stack",
        "ip_stack": "in_plane_stack",
        "inplane": "in_plane_stack",
        "tilted_oop": "tilted_out_of_plane_stack",
        "tilted_out_of_plane": "tilted_out_of_plane_stack",
        "tilted_oop_stack": "tilted_out_of_plane_stack",
        "tilted_ip": "tilted_in_plane_stack",
        "tilted_in_plane": "tilted_in_plane_stack",
        "tilted_ip_stack": "tilted_in_plane_stack",
        "mixed": "tilted_in_plane_stack",
    }
    return aliases.get(token, token or "out_of_plane_stack")


def _texture_orientation_centers(mode: str) -> tuple[float, tuple[float, ...]]:
    if mode == "out_of_plane_stack":
        return 0.0, (0.0, 90.0, 180.0, 270.0)
    if mode == "in_plane_stack":
        return 90.0, (0.0, 90.0, 180.0, 270.0)
    if mode == "tilted_out_of_plane_stack":
        return 25.0, (0.0, 90.0, 180.0, 270.0)
    if mode == "tilted_in_plane_stack":
        return 65.0, (0.0, 90.0, 180.0, 270.0)
    return 90.0, (0.0, 90.0, 180.0, 270.0)


def _run_one_synthetic_fileset(
    spec: BenchmarkStructureSpec,
    simulation_index: int,
    order_index: int,
    run_root: Path,
    logbook: Path,
    rng: np.random.Generator,
    cfg: SyntheticRefinementConfig,
) -> dict[str, Any]:
    fileset_id = f"{spec.structure_id}_synthetic_{simulation_index:02d}"
    _progress(
        f"[{order_index}] starting {fileset_id} from {spec.cif_path.name}"
    )
    fileset_dir = (
        run_root / spec.structure_id / f"synthetic_{simulation_index:02d}"
    )
    simulation_dir = fileset_dir / "simulations"
    qspace_dir = fileset_dir / "qspace"
    plots_dir = fileset_dir / "plots"
    rankings_dir = fileset_dir / "rankings"
    generated_dir = fileset_dir / "generated_structures"
    scaffold_dir = fileset_dir / "inorganic_scaffolds"
    organic_proxy_dir = fileset_dir / "organic_proxy_structures"
    organic_replacement_dir = fileset_dir / "organic_replacement_structures"
    organic_rmc_dir = fileset_dir / "organic_rmc_structures"
    best_dir = fileset_dir / "best_fit_generated_structures"
    for directory in (
        simulation_dir,
        qspace_dir,
        plots_dir,
        rankings_dir,
        generated_dir,
        scaffold_dir,
        organic_proxy_dir,
        organic_replacement_dir,
        organic_rmc_dir,
        best_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    texture = _scheduled_texture_mode(cfg, order_index)
    _progress(f"[{order_index}] simulate {fileset_id} texture={texture}")
    params = _synthetic_texture_parameters(rng, cfg, texture)
    simulation_cif_path = _write_lattice_disordered_cif(
        spec.cif_path,
        simulation_dir / f"{fileset_id}_simulation_lattice.cif",
        rng,
        cfg,
    )
    clean = simulate_giwaxs_image(simulation_cif_path, params)
    clean_nc = simulation_dir / f"{fileset_id}_clean.nc"
    clean.to_netcdf(clean_nc)
    truth_peak_path = simulation_dir / f"{fileset_id}_truth_peaks.json"
    truth_peaks = _write_truth_peak_table(truth_peak_path, clean, cfg)
    noisy = _mock_experimental_qspace(clean, rng, cfg)
    tiff_path = fileset_dir / f"{fileset_id}_mock_experiment.tiff"
    qspace_path = qspace_dir / f"{fileset_id}_mock_qspace.npz"
    _write_mock_tiff(tiff_path, noisy, spec, params)
    _write_qspace_npz(qspace_path, noisy, spec, params)

    _progress(f"[{order_index}] detect peaks {fileset_id}")
    project = ProjectState(name=f"Synthetic refinement {fileset_id}")
    data_file = project.add_data_file(
        tiff_path,
        benchmark_fileset_id=fileset_id,
        reference_cif=str(spec.cif_path),
        synthetic=True,
    )
    project.processed_products[data_file.data_id] = str(qspace_path)
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=str(data_file.data_id),
            confirmed=True,
            metadata={
                "synthetic_refinement_fileset_id": fileset_id,
                "qspace_product": str(qspace_path),
            },
        )
    )
    project.simulations[f"{fileset_id}_reference"] = {
        "simulation_id": f"{fileset_id}_reference",
        "simulation_mode": "synthetic_refinement_reference_clean",
        "data_id": data_file.data_id,
        "structure_name": spec.label or spec.cif_path.stem,
        "structure_path": str(simulation_cif_path),
        "cif_path": str(simulation_cif_path),
        "dataset_uri": str(clean_nc),
        "parameters": params.as_dict(),
        "metadata": {
            "created_at": _timestamp(),
            "synthetic_measurement": str(tiff_path),
            "reference_validation_cif": str(spec.cif_path),
            "truth_peak_table": str(truth_peak_path),
        },
    }

    peaks = _detect_structure_peaks(noisy, cfg)
    _progress(f"[{order_index}] detected {len(peaks)} peaks for {fileset_id}")
    peak_plot = _save_peak_detection_plot(
        noisy,
        peaks,
        plots_dir / "peak_detection.png",
        title=fileset_id,
    )
    peak_recovery = _peak_detection_recovery_metrics(peaks, truth_peaks, cfg)
    peak_truth = match_detected_peaks_to_truth(
        peaks,
        truth_peaks,
        tolerance=float(peak_recovery.get("match_tolerance", 0.07)),
    )
    solver_peaks = peaks
    solver_input_mode = "detected_peak_finder_output"
    if cfg.oracle_peak_positions:
        solver_peaks = _oracle_structure_peaks_from_truth(truth_peaks)
        solver_input_mode = "oracle_truth_peak_positions_and_hkl"
        _write_json(
            rankings_dir / "oracle_peak_inputs.json",
            {
                "policy": (
                    "Exact synthetic truth peak positions and hkl labels are "
                    "used as solver inputs. Reference atom coordinates are "
                    "not used for scaffold construction."
                ),
                "peak_count": len(solver_peaks),
                "peaks": [peak.as_dict() for peak in solver_peaks],
            },
        )
        _write_json(
            rankings_dir / "detected_peak_truth_matches.json", peak_truth
        )
        peak_truth = match_detected_peaks_to_truth(
            solver_peaks,
            truth_peaks,
            tolerance=max(float(cfg.peak_deduplicate_tolerance), 1.0e-9),
        )
    _write_json(
        rankings_dir / "peak_position_errors.json",
        peak_truth["position_error_summary"],
    )
    if cfg.oracle_peak_positions:
        families = _oracle_peak_families_from_peaks(solver_peaks)
    else:
        families = group_peak_families(
            solver_peaks,
            tolerance=cfg.family_tolerance,
            ratio_tolerance=cfg.family_ratio_tolerance,
            phase_tag=DEFAULT_PHASE_TAG,
        )
    family_assessment = assess_peak_families_against_truth(
        families,
        peak_truth["matches"],
    )
    _write_json(rankings_dir / "peak_truth_matches.json", peak_truth)
    _write_json(rankings_dir / "peak_families.json", families)
    _write_json(
        rankings_dir / "peak_family_assessment.json", family_assessment
    )

    _progress(f"[{order_index}] solve lattice candidates {fileset_id}")
    candidates = guess_lattice_candidates(
        solver_peaks,
        _candidate_search_config(cfg),
    )
    if cfg.oracle_lattice_parameters:
        oracle_candidate = _oracle_lattice_candidate_from_reference(
            spec,
            solver_peaks,
        )
        if oracle_candidate is not None:
            candidates = [oracle_candidate, *candidates]
    candidates = _expand_axis_scale_candidates(candidates, cfg)
    _progress(
        f"[{order_index}] generated {len(candidates)} lattice candidates "
        f"for {fileset_id}"
    )
    candidate_records = [candidate.as_dict() for candidate in candidates]
    candidate_truth = assess_candidate_assignments_against_truth(
        candidate_records,
        peak_truth["matches"],
    )
    for record in candidate_records:
        metrics = candidate_truth.get(str(record.get("candidate_id")), {})
        record["truth_assignment_metrics"] = metrics
    _write_json(rankings_dir / "lattice_candidates.json", candidate_records)
    _write_json(
        rankings_dir / "candidate_truth_assessment.json", candidate_truth
    )

    stoichiometry_hypotheses = chemistry_stoichiometry_hypotheses(
        spec.inorganic_atoms,
        spec.organic_molecules,
        limit=cfg.max_stoichiometry_hypotheses,
    )
    if cfg.oracle_stoichiometry_constraints:
        stoichiometry_hypotheses = _oracle_stoichiometry_hypotheses(
            spec,
            stoichiometry_hypotheses,
        )
    perovskite_motifs = perovskite_scaffold_hypotheses(
        spec.inorganic_atoms,
        spec.organic_molecules,
        limit=max(cfg.max_stoichiometry_hypotheses * 3, 24),
    )
    _write_json(
        rankings_dir / "stoichiometry_hypotheses.json",
        {
            "policy": (
                "chemistry-derived search hypotheses from allowed atom and "
                "molecule labels; reference CIF stoichiometry read only when "
                "oracle stoichiometry diagnostics are enabled"
            ),
            "hypotheses": stoichiometry_hypotheses,
            "perovskite_motif_hypotheses": perovskite_motifs,
            "oracle_stoichiometry_constraints": (
                cfg.oracle_stoichiometry_constraints
            ),
        },
    )
    _write_json(
        rankings_dir / "perovskite_motif_hypotheses.json",
        perovskite_motifs,
    )
    _progress(f"[{order_index}] generate scaffold CIFs {fileset_id}")
    generated_records = generate_scaffold_candidate_cifs(
        spec,
        candidates,
        stoichiometry_hypotheses,
        cfg,
    )
    generated_records = sorted(
        generated_records,
        key=lambda item: float(item.get("score", math.inf)),
    )[: max(cfg.max_generated_cifs_to_compare, cfg.max_scaffolds_to_validate)]
    generated_records = _materialize_generated_cifs(
        generated_records,
        generated_dir,
        data_file_id=str(data_file.data_id),
        fileset_id=fileset_id,
    )
    project.reference_cifs.setdefault("generated", {}).update(
        {str(record["cif_id"]): dict(record) for record in generated_records}
    )
    for record in generated_records:
        project.structures[str(record["cif_id"])] = dict(record)

    _progress(
        f"[{order_index}] validate {len(generated_records)} scaffolds "
        f"{fileset_id}"
    )
    scaffold_rankings = rank_inorganic_scaffolds(
        generated_records,
        spec,
        cfg,
        scaffold_dir,
        rankings_dir / "scaffold_validation",
        best_dir,
        run_root / "best_fit_generated_structures" / fileset_id,
        peak_intensity_peaks=solver_peaks,
        peak_intensity_params=params,
    )
    _write_json(
        rankings_dir / "inorganic_scaffold_rankings.json", scaffold_rankings
    )
    stage_rankings: dict[str, list[dict[str, Any]]] = {}
    stage_rankings["inorganic_scaffold"] = _rank_stage_cifs_with_simulation(
        noisy,
        [_stage_record_from_scaffold(item) for item in scaffold_rankings],
        params,
        cfg,
        stage_name="inorganic_scaffold",
        rankings_dir=rankings_dir,
        plots_dir=plots_dir,
        peaks=solver_peaks,
    )
    organic_proxy_rankings: list[dict[str, Any]] = []
    if (
        cfg.generate_organic_electron_proxies
        and cfg.max_organic_proxy_cifs_to_compare > 0
    ):
        _progress(f"[{order_index}] generate organic proxy CIFs {fileset_id}")
        organic_proxy_records = generate_organic_electron_proxy_cifs(
            spec,
            candidates,
            stoichiometry_hypotheses,
            cfg,
        )
        organic_proxy_rankings = sorted(
            organic_proxy_records,
            key=lambda item: float(item.get("score", math.inf)),
        )[: cfg.max_organic_proxy_cifs_to_compare]
        organic_proxy_rankings = _materialize_generated_cifs(
            organic_proxy_rankings,
            organic_proxy_dir,
            data_file_id=str(data_file.data_id),
            fileset_id=fileset_id,
        )
        stage_rankings["organic_electron_proxy"] = (
            _rank_stage_cifs_with_simulation(
                noisy,
                organic_proxy_rankings,
                params,
                cfg,
                stage_name="organic_electron_proxy",
                rankings_dir=rankings_dir,
                plots_dir=plots_dir,
                peaks=solver_peaks,
            )
        )
    _write_json(
        rankings_dir / "organic_electron_proxy_rankings.json",
        organic_proxy_rankings,
    )
    project.reference_cifs.setdefault("organic_proxy", {}).update(
        {
            str(record["cif_id"]): dict(record)
            for record in organic_proxy_rankings
        }
    )
    for record in organic_proxy_rankings:
        project.structures[str(record["cif_id"])] = dict(record)
    organic_replacement_rankings: list[dict[str, Any]] = []
    if (
        cfg.generate_organic_replacement_structures
        and cfg.max_organic_replacement_cifs_to_compare > 0
    ):
        _progress(
            f"[{order_index}] generate organic replacement CIFs {fileset_id}"
        )
        organic_replacement_records = generate_organic_replacement_cifs(
            spec,
            candidates,
            stoichiometry_hypotheses,
            cfg,
        )
        organic_replacement_records = sorted(
            organic_replacement_records,
            key=lambda item: float(item.get("score", math.inf)),
        )[: cfg.max_organic_replacement_cifs_to_compare]
        organic_replacement_records = _materialize_generated_cifs(
            organic_replacement_records,
            organic_replacement_dir / "raw",
            data_file_id=str(data_file.data_id),
            fileset_id=fileset_id,
        )
        organic_replacement_records = _physicalize_stage_records(
            organic_replacement_records,
            organic_replacement_dir,
            suffix="organic_replacement_geometry",
        )
        _write_json(
            rankings_dir / "organic_replacement_candidates.json",
            organic_replacement_records,
        )
        organic_replacement_rankings = _rank_stage_cifs_with_simulation(
            noisy,
            organic_replacement_records,
            params,
            cfg,
            stage_name="organic_replacement",
            rankings_dir=rankings_dir,
            plots_dir=plots_dir,
            peaks=solver_peaks,
        )
        stage_rankings["organic_replacement"] = organic_replacement_rankings
        project.reference_cifs.setdefault("organic_replacement", {}).update(
            {
                str(record["cif_id"]): dict(record)
                for record in organic_replacement_records
            }
        )
        for record in organic_replacement_records:
            project.structures[str(record["cif_id"])] = dict(record)
    organic_rmc_rankings: list[dict[str, Any]] = []
    if organic_replacement_rankings and cfg.organic_rmc_steps > 0:
        _progress(f"[{order_index}] organic RMC variants {fileset_id}")
        organic_rmc_records = generate_organic_rmc_variant_cifs(
            Path(str(organic_replacement_rankings[0].get("path"))),
            fileset_id=fileset_id,
            rng=rng,
            cfg=cfg,
            output_dir=organic_rmc_dir,
        )
        _write_json(
            rankings_dir / "organic_rmc_candidates.json",
            organic_rmc_records,
        )
        organic_rmc_rankings = _rank_stage_cifs_with_simulation(
            noisy,
            organic_rmc_records,
            params,
            cfg,
            stage_name="organic_rmc",
            rankings_dir=rankings_dir,
            plots_dir=plots_dir,
            peaks=solver_peaks,
        )
        stage_rankings["organic_rmc"] = organic_rmc_rankings
        project.reference_cifs.setdefault("organic_rmc", {}).update(
            {
                str(record["cif_id"]): dict(record)
                for record in organic_rmc_records
            }
        )
        for record in organic_rmc_records:
            project.structures[str(record["cif_id"])] = dict(record)
    reference_stage_comparisons = _attach_reference_structure_comparisons(
        stage_rankings,
        spec,
        rankings_dir / "reference_structure_comparisons",
    )
    _write_json(
        rankings_dir / "refinement_stage_rankings.json", stage_rankings
    )
    _progress(f"[{order_index}] store project {fileset_id}")
    image_rankings: list[dict[str, Any]] = []
    if cfg.rank_generated_cifs_with_image_fit:
        from ewald.benchmark.structure_benchmark import (
            _attach_refined_cif_records,
            _rank_generated_cifs,
        )

        image_rankings = _rank_generated_cifs(
            noisy,
            generated_records[: cfg.max_generated_cifs_to_compare],
            params,
            cfg,
            rankings_dir,
            plots_dir,
            peak_intensity_peaks=solver_peaks,
        )
        _attach_refined_cif_records(project, image_rankings)

    _store_synthetic_project_analysis(
        project,
        str(data_file.data_id),
        fileset_id,
        spec,
        peaks,
        families,
        candidate_records,
        stoichiometry_hypotheses,
        perovskite_motifs,
        peak_truth,
        family_assessment,
        candidate_truth,
        scaffold_rankings,
        organic_proxy_rankings,
        organic_replacement_rankings,
        organic_rmc_rankings,
        stage_rankings,
        image_rankings,
    )
    project_path = save_project(project, fileset_dir / f"{fileset_id}.ewld")
    best_scaffold = scaffold_rankings[0] if scaffold_rankings else {}
    fileset_summary = {
        "fileset_id": fileset_id,
        "structure": spec.as_dict(),
        "solve_order": order_index,
        "simulation_index": simulation_index,
        "mock_tiff": str(tiff_path),
        "mock_qspace": str(qspace_path),
        "clean_simulation": str(clean_nc),
        "simulation_cif": str(simulation_cif_path),
        "truth_peak_table": str(truth_peak_path),
        "peak_detection_plot": str(peak_plot),
        "project": str(project_path),
        "readable_project": str(project_path.with_suffix(".ewald.json")),
        "solver_input_mode": solver_input_mode,
        "solver_peak_count": len(solver_peaks),
        "oracle_diagnostic": {
            "lattice_parameters": cfg.oracle_lattice_parameters,
            "peak_positions_and_hkl": cfg.oracle_peak_positions,
            "stoichiometry_constraints": (
                cfg.oracle_stoichiometry_constraints
            ),
            "atom_coordinates_allowed": False,
        },
        "orientation": {
            "texture_mode": texture,
            "theta_x_deg": params.theta_x_deg,
            "theta_y_deg": params.theta_y_deg,
            "sigma_theta": params.sigma_theta,
            "sigma_phi": params.sigma_phi,
            "sigma_r": params.sigma_r,
        },
        "peak_count": len(peaks),
        "truth_peak_count": len(truth_peaks),
        "peak_recovery": peak_recovery,
        "peak_truth_match_summary": peak_truth["summary"],
        "peak_position_error_summary": peak_truth["position_error_summary"],
        "family_count": len(families),
        "family_assessment": family_assessment["summary"],
        "top_candidate": candidate_records[0] if candidate_records else None,
        "top_perovskite_motif": (
            perovskite_motifs[0] if perovskite_motifs else None
        ),
        "top_candidate_truth_assignment": (
            candidate_records[0].get("truth_assignment_metrics", {})
            if candidate_records
            else {}
        ),
        "best_inorganic_scaffold": best_scaffold.get("path"),
        "best_inorganic_scaffold_validation": best_scaffold.get(
            "validation", {}
        ),
        "best_bragg_intensity_match": best_scaffold.get(
            "bragg_intensity_match",
            {},
        ),
        "best_bragg_intensity_penalty": best_scaffold.get(
            "bragg_intensity_penalty",
            0.0,
        ),
        "organic_electron_proxy_rankings": organic_proxy_rankings[:3],
        "organic_replacement_rankings": organic_replacement_rankings[:3],
        "organic_rmc_rankings": organic_rmc_rankings[:3],
        "refinement_stage_rankings": {
            key: value[:3] for key, value in stage_rankings.items()
        },
        "reference_stage_comparisons": reference_stage_comparisons,
        "best_organic_replacement": (
            organic_replacement_rankings[0]
            if organic_replacement_rankings
            else {}
        ),
        "best_organic_rmc": (
            organic_rmc_rankings[0] if organic_rmc_rankings else {}
        ),
        "image_rankings": image_rankings[:3],
    }
    _append_synthetic_fileset_logbook(
        logbook,
        fileset_summary,
        scaffold_rankings,
    )
    return fileset_summary


def match_detected_peaks_to_truth(
    detected: list[StructurePeak],
    truth_peaks: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Greedily match detected q-space peaks to simulated truth
    peaks."""

    pairs: list[tuple[float, int, int]] = []
    for detected_index, peak in enumerate(detected):
        for truth_index, truth in enumerate(truth_peaks):
            distance = math.hypot(
                abs(float(peak.qxy)) - abs(float(truth.get("qxy", 0.0))),
                float(peak.qz) - float(truth.get("qz", 0.0)),
            )
            if distance <= tolerance:
                pairs.append((distance, detected_index, truth_index))
    matched_detected: set[int] = set()
    matched_truth: set[int] = set()
    assignments: dict[int, tuple[float, int]] = {}
    for distance, detected_index, truth_index in sorted(pairs):
        if detected_index in matched_detected or truth_index in matched_truth:
            continue
        matched_detected.add(detected_index)
        matched_truth.add(truth_index)
        assignments[detected_index] = (distance, truth_index)
    matches = []
    for detected_index, peak in enumerate(detected):
        distance_truth = assignments.get(detected_index)
        truth = None
        distance = None
        position_error = None
        if distance_truth is not None:
            distance, truth_index = distance_truth
            truth = truth_peaks[truth_index]
            position_error = _peak_position_error_record(peak, truth)
        matches.append(
            {
                "peak_id": peak.peak_id,
                "label": peak.label,
                "qxy": float(peak.qxy),
                "qz": float(peak.qz),
                "truth_distance": distance,
                "position_error": position_error,
                "truth": truth,
                "truth_hkl": _truth_hkl(truth) if truth else None,
                "truth_hkl_family": (
                    _plane_family_key(_truth_hkl(truth)) if truth else None
                ),
            }
        )
    position_error_summary = _peak_position_error_summary(matches)
    truth_rank_diagnostics = _truth_rank_recovery_diagnostics(
        truth_peaks,
        matched_truth,
    )
    summary = {
        "detected_peak_count": len(detected),
        "truth_peak_count": len(truth_peaks),
        "matched_detected_peak_count": len(matched_detected),
        "matched_truth_peak_count": len(matched_truth),
        "precision": len(matched_detected) / max(len(detected), 1),
        "recall": len(matched_truth) / max(len(truth_peaks), 1),
        "match_tolerance": float(tolerance),
        "mean_radial_position_error": position_error_summary.get(
            "mean_radial_error",
        ),
        "median_radial_position_error": position_error_summary.get(
            "median_radial_error",
        ),
        "qxy_abs_bias": position_error_summary.get("mean_delta_abs_qxy"),
        "qz_bias": position_error_summary.get("mean_delta_qz"),
        "amplitude_weighted_recall": truth_rank_diagnostics.get(
            "amplitude_weighted_recall"
        ),
        "intensity_weighted_recall": truth_rank_diagnostics.get(
            "intensity_weighted_recall"
        ),
        "top25_truth_recall": truth_rank_diagnostics.get(
            "top_truth_recall",
            {},
        )
        .get("top_25", {})
        .get("recall"),
        "top50_truth_recall": truth_rank_diagnostics.get(
            "top_truth_recall",
            {},
        )
        .get("top_50", {})
        .get("recall"),
        "top100_truth_recall": truth_rank_diagnostics.get(
            "top_truth_recall",
            {},
        )
        .get("top_100", {})
        .get("recall"),
    }
    return {
        "summary": summary,
        "matches": matches,
        "position_error_summary": position_error_summary,
        "truth_rank_diagnostics": truth_rank_diagnostics,
    }


def _peak_position_error_record(
    peak: StructurePeak,
    truth: dict[str, Any],
) -> dict[str, Any]:
    detected_qxy = float(peak.qxy)
    detected_qz = float(peak.qz)
    truth_qxy = float(truth.get("qxy", 0.0))
    truth_qz = float(truth.get("qz", 0.0))
    delta_signed_qxy = detected_qxy - truth_qxy
    delta_abs_qxy = abs(detected_qxy) - abs(truth_qxy)
    delta_qz = detected_qz - truth_qz
    return {
        "detected_qxy": detected_qxy,
        "detected_qz": detected_qz,
        "truth_qxy": truth_qxy,
        "truth_qz": truth_qz,
        "delta_signed_qxy": delta_signed_qxy,
        "delta_abs_qxy": delta_abs_qxy,
        "delta_qz": delta_qz,
        "abs_delta_signed_qxy": abs(delta_signed_qxy),
        "abs_delta_abs_qxy": abs(delta_abs_qxy),
        "abs_delta_qz": abs(delta_qz),
        "radial_error": math.hypot(delta_abs_qxy, delta_qz),
        "signed_radial_error": math.hypot(delta_signed_qxy, delta_qz),
    }


def _peak_position_error_summary(
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = [
        item["position_error"]
        for item in matches
        if isinstance(item.get("position_error"), dict)
    ]
    if not errors:
        return {
            "matched_peak_count": 0,
            "mean_delta_abs_qxy": None,
            "mean_delta_qz": None,
            "mean_radial_error": None,
            "median_radial_error": None,
            "rms_radial_error": None,
        }
    keys = [
        "delta_abs_qxy",
        "delta_signed_qxy",
        "delta_qz",
        "abs_delta_abs_qxy",
        "abs_delta_qz",
        "radial_error",
        "signed_radial_error",
    ]
    arrays = {
        key: np.asarray([float(error[key]) for error in errors], dtype=float)
        for key in keys
    }
    return {
        "matched_peak_count": len(errors),
        "mean_delta_abs_qxy": float(np.mean(arrays["delta_abs_qxy"])),
        "mean_delta_signed_qxy": float(np.mean(arrays["delta_signed_qxy"])),
        "mean_delta_qz": float(np.mean(arrays["delta_qz"])),
        "mean_abs_delta_abs_qxy": float(np.mean(arrays["abs_delta_abs_qxy"])),
        "mean_abs_delta_qz": float(np.mean(arrays["abs_delta_qz"])),
        "mean_radial_error": float(np.mean(arrays["radial_error"])),
        "median_radial_error": float(np.median(arrays["radial_error"])),
        "max_radial_error": float(np.max(arrays["radial_error"])),
        "rms_radial_error": float(
            np.sqrt(np.mean(arrays["radial_error"] ** 2))
        ),
        "mean_signed_radial_error": float(
            np.mean(arrays["signed_radial_error"])
        ),
    }


def _truth_rank_recovery_diagnostics(
    truth_peaks: list[dict[str, Any]],
    matched_truth_indices: set[int],
) -> dict[str, Any]:
    """Summarize recovery of the strongest simulated truth
    reflections."""

    if not truth_peaks:
        return {
            "truth_peak_count": 0,
            "matched_truth_peak_count": 0,
            "top_truth_recall": {},
            "amplitude_weighted_recall": 0.0,
            "intensity_weighted_recall": 0.0,
            "unmatched_strong_truth_peaks": [],
        }
    ordered = sorted(
        enumerate(truth_peaks),
        key=lambda item: _truth_peak_weight(item[1], "amplitude"),
        reverse=True,
    )
    top_truth_recall: dict[str, Any] = {}
    for limit in (25, 50, 100, 250):
        selected = ordered[: min(limit, len(ordered))]
        matched_count = sum(
            1 for index, _ in selected if index in matched_truth_indices
        )
        top_truth_recall[f"top_{limit}"] = {
            "truth_count": len(selected),
            "matched_truth_peak_count": matched_count,
            "recall": matched_count / max(len(selected), 1),
        }
    amplitude_total = sum(
        _truth_peak_weight(peak, "amplitude") for peak in truth_peaks
    )
    intensity_total = sum(
        _truth_peak_weight(peak, "intensity") for peak in truth_peaks
    )
    amplitude_matched = sum(
        _truth_peak_weight(peak, "amplitude")
        for index, peak in enumerate(truth_peaks)
        if index in matched_truth_indices
    )
    intensity_matched = sum(
        _truth_peak_weight(peak, "intensity")
        for index, peak in enumerate(truth_peaks)
        if index in matched_truth_indices
    )
    unmatched = [
        _truth_peak_compact_record(index, peak)
        for index, peak in ordered
        if index not in matched_truth_indices
    ][:12]
    return {
        "truth_peak_count": len(truth_peaks),
        "matched_truth_peak_count": len(matched_truth_indices),
        "top_truth_recall": top_truth_recall,
        "amplitude_weighted_recall": (
            amplitude_matched / amplitude_total
            if amplitude_total > 0.0
            else 0.0
        ),
        "intensity_weighted_recall": (
            intensity_matched / intensity_total
            if intensity_total > 0.0
            else 0.0
        ),
        "unmatched_strong_truth_peaks": unmatched,
    }


def _truth_peak_weight(peak: dict[str, Any], key: str) -> float:
    fallback_key = "intensity" if key == "amplitude" else "amplitude"
    value = peak.get(key, peak.get(fallback_key, 1.0))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 1.0
    return max(parsed, 0.0) if np.isfinite(parsed) else 0.0


def _truth_peak_compact_record(
    index: int,
    peak: dict[str, Any],
) -> dict[str, Any]:
    return {
        "truth_index": int(index),
        "hkl": [
            int(peak.get("h", 0)),
            int(peak.get("k", 0)),
            int(peak.get("l", 0)),
        ],
        "hkl_family": _plane_family_key(
            (
                int(peak.get("h", 0)),
                int(peak.get("k", 0)),
                int(peak.get("l", 0)),
            )
        ),
        "qxy": float(peak.get("qxy", 0.0)),
        "qz": float(peak.get("qz", 0.0)),
        "amplitude": _truth_peak_weight(peak, "amplitude"),
        "intensity": _truth_peak_weight(peak, "intensity"),
    }


def assess_candidate_assignments_against_truth(
    candidate_records: list[dict[str, Any]],
    peak_truth_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_peak = {
        str(item["peak_id"]): item
        for item in peak_truth_matches
        if item.get("truth_hkl") is not None
    }
    results: dict[str, Any] = {}
    for candidate in candidate_records:
        assignments = candidate.get("assignments", [])
        compared = 0
        family_matches = 0
        exact_abs_matches = 0
        rows = []
        for assignment in assignments:
            peak_id = str(assignment.get("peak_id") or "")
            truth = truth_by_peak.get(peak_id)
            predicted = _parse_hkl_triplet(assignment.get("hkl"))
            if truth is None or predicted is None:
                continue
            truth_hkl = tuple(int(value) for value in truth["truth_hkl"])
            predicted_family = _plane_family_key(predicted)
            truth_family = _plane_family_key(truth_hkl)
            exact_abs = tuple(abs(value) for value in predicted) == tuple(
                abs(value) for value in truth_hkl
            )
            family_match = predicted_family == truth_family
            compared += 1
            family_matches += int(family_match)
            exact_abs_matches += int(exact_abs)
            rows.append(
                {
                    "peak_id": peak_id,
                    "predicted_hkl": list(predicted),
                    "truth_hkl": list(truth_hkl),
                    "predicted_hkl_family": predicted_family,
                    "truth_hkl_family": truth_family,
                    "family_match": family_match,
                    "exact_abs_match": exact_abs,
                    "projected_delta_q": assignment.get("projected_delta_q"),
                }
            )
        results[str(candidate.get("candidate_id"))] = {
            "compared_assignment_count": compared,
            "hkl_family_match_count": family_matches,
            "exact_abs_hkl_match_count": exact_abs_matches,
            "hkl_family_accuracy": family_matches / max(compared, 1),
            "exact_abs_hkl_accuracy": exact_abs_matches / max(compared, 1),
            "assignments": rows[:20],
        }
    return results


def assess_peak_families_against_truth(
    families: list[dict[str, Any]],
    peak_truth_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_peak = {
        str(item["peak_id"]): item
        for item in peak_truth_matches
        if item.get("truth_hkl_family") is not None
    }
    records = []
    purities = []
    multi_truth_purities = []
    weighted_dominant = 0
    weighted_total = 0
    for family in families:
        truth_keys = [
            str(truth_by_peak[peak_id]["truth_hkl_family"])
            for peak_id in family.get("peak_ids", [])
            if peak_id in truth_by_peak
        ]
        if not truth_keys:
            records.append({**family, "truth_match_count": 0})
            continue
        counts = {
            key: truth_keys.count(key) for key in sorted(set(truth_keys))
        }
        dominant_key, dominant_count = max(
            counts.items(),
            key=lambda item: item[1],
        )
        purity = dominant_count / max(len(truth_keys), 1)
        purities.append(purity)
        weighted_dominant += int(dominant_count)
        weighted_total += len(truth_keys)
        if len(truth_keys) >= 2:
            multi_truth_purities.append(purity)
        records.append(
            {
                **family,
                "truth_match_count": len(truth_keys),
                "dominant_truth_hkl_family": dominant_key,
                "truth_hkl_family_counts": counts,
                "truth_family_purity": purity,
            }
        )
    summary = {
        "family_count": len(families),
        "truth_scored_family_count": len(purities),
        "mean_truth_family_purity": (
            float(np.mean(purities)) if purities else 0.0
        ),
        "mean_multi_truth_family_purity": (
            float(np.mean(multi_truth_purities))
            if multi_truth_purities
            else 0.0
        ),
        "weighted_mean_truth_family_purity": (
            weighted_dominant / weighted_total if weighted_total else 0.0
        ),
        "multi_truth_scored_family_count": len(multi_truth_purities),
        "high_purity_family_count": sum(
            1 for value in purities if value >= 0.75
        ),
        "mixed_truth_family_count": sum(
            1 for value in purities if value < 0.75
        ),
    }
    return {"summary": summary, "families": records}


def rank_inorganic_scaffolds(
    generated_records: list[dict[str, Any]],
    spec: BenchmarkStructureSpec,
    cfg: SyntheticRefinementConfig,
    scaffold_dir: Path,
    validation_dir: Path,
    local_best_dir: Path,
    global_best_dir: Path,
    *,
    peak_intensity_peaks: list[StructurePeak] | None = None,
    peak_intensity_params: GIWAXSSimulationParameters | None = None,
) -> list[dict[str, Any]]:
    """Write, score, and validate inorganic-only scaffold candidates."""

    scaffold_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    records = []
    keep_elements = {
        _element_symbol(element) for element in spec.inorganic_atoms
    }
    for record in generated_records[: cfg.max_scaffolds_to_validate]:
        source_path = Path(
            str(record.get("path") or record.get("structure_path"))
        )
        if not source_path.exists():
            continue
        cif_id = _slug(str(record.get("cif_id") or source_path.stem))
        raw_scaffold = scaffold_dir / f"{cif_id}_inorganic_scaffold.cif"
        physical_scaffold = (
            scaffold_dir / f"{cif_id}_inorganic_chemical_geometry.cif"
        )
        _write_filtered_cif(
            source_path, raw_scaffold, keep_elements=keep_elements
        )
        _write_physicalized_cif(raw_scaffold, physical_scaffold)
        chemistry_metrics = _cif_physical_chemistry_metrics(physical_scaffold)
        physical_penalty = float(
            chemistry_metrics.get("physical_penalty", 0.0) or 0.0
        )
        bragg_intensity_match: dict[str, Any] = {}
        bragg_intensity_penalty = 0.0
        if (
            peak_intensity_peaks
            and peak_intensity_params is not None
            and cfg.bragg_intensity_weight > 0.0
        ):
            try:
                simulated_rows = calculate_giwaxs_peak_rows(
                    physical_scaffold,
                    peak_intensity_params,
                )
                bragg_intensity_match = _score_bragg_peak_intensity_match(
                    peak_intensity_peaks,
                    simulated_rows,
                    tolerance=cfg.bragg_intensity_tolerance,
                    max_peaks=cfg.bragg_intensity_max_peaks,
                )
                bragg_intensity_penalty = cfg.bragg_intensity_weight * float(
                    bragg_intensity_match.get(
                        "intensity_match_penalty",
                        0.0,
                    )
                    or 0.0
                )
            except Exception as exc:
                bragg_intensity_match = {
                    "status": "error",
                    "error": str(exc),
                    "intensity_match_penalty": 0.0,
                }
        solver_score = (
            float(record.get("score", math.inf))
            + float(record.get("charge_balance_penalty", 0.0) or 0.0)
            + float(record.get("lattice_prior_penalty", 0.0) or 0.0)
            + float(record.get("stoichiometry_prior_penalty", 0.0) or 0.0)
            + physical_penalty
            + bragg_intensity_penalty
        )
        validation = validate_inorganic_scaffold(
            physical_scaffold,
            spec.cif_path,
            keep_elements,
            validation_dir / cif_id,
        )
        records.append(
            {
                "generated_cif_id": cif_id,
                "source_path": str(source_path),
                "raw_scaffold_path": str(raw_scaffold),
                "path": str(physical_scaffold),
                "solver_score": solver_score,
                "source_score": record.get("score"),
                "charge_penalty": record.get("charge_balance_penalty", 0.0),
                "stoichiometry_prior_penalty": record.get(
                    "stoichiometry_prior_penalty",
                    0.0,
                ),
                "lattice_prior_penalty": record.get(
                    "lattice_prior_penalty",
                    0.0,
                ),
                "bragg_intensity_weight": cfg.bragg_intensity_weight,
                "bragg_intensity_penalty": bragg_intensity_penalty,
                "bragg_intensity_match": bragg_intensity_match,
                "physical_penalty": physical_penalty,
                "chemistry_metrics": chemistry_metrics,
                "validation": validation,
            }
        )
    records.sort(
        key=lambda item: (
            float(item.get("solver_score", math.inf)),
            float(item.get("physical_penalty", math.inf)),
        )
    )
    for rank, record in enumerate(records, start=1):
        record["scaffold_rank"] = rank
    if records:
        local_best_dir.mkdir(parents=True, exist_ok=True)
        global_best_dir.mkdir(parents=True, exist_ok=True)
        best_path = Path(str(records[0]["path"]))
        local_copy = local_best_dir / best_path.name
        global_copy = global_best_dir / best_path.name
        shutil.copy2(best_path, local_copy)
        shutil.copy2(best_path, global_copy)
        records[0]["best_local_copy"] = str(local_copy)
        records[0]["best_global_copy"] = str(global_copy)
    return records


def _stage_record_from_scaffold(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "cif_id": record.get("generated_cif_id"),
        "path": record.get("path"),
        "source_path": record.get("source_path"),
        "coordinate_model": "inorganic_scaffold_only",
        "source_score": record.get("source_score"),
        "solver_score": record.get("solver_score"),
        "charge_balance_penalty": record.get("charge_penalty", 0.0),
        "lattice_prior_penalty": record.get("lattice_prior_penalty", 0.0),
        "bragg_intensity_penalty": record.get("bragg_intensity_penalty", 0.0),
        "bragg_intensity_match": record.get("bragg_intensity_match", {}),
        "validation": record.get("validation", {}),
    }


def _rank_stage_cifs_with_simulation(
    target: Any,
    records: list[dict[str, Any]],
    target_params: GIWAXSSimulationParameters,
    cfg: SyntheticRefinementConfig,
    *,
    stage_name: str,
    rankings_dir: Path,
    plots_dir: Path,
    peaks: list[StructurePeak],
) -> list[dict[str, Any]]:
    """Simulate and rank one explicit refinement stage."""

    stage_records = [
        dict(record)
        for record in records[: max(1, int(cfg.stage_simulation_max_cifs))]
        if record.get("path")
    ]
    stage_rankings_dir = rankings_dir / "stage_simulations" / stage_name
    stage_plot_dir = plots_dir / "stage_simulations" / stage_name
    stage_rankings_dir.mkdir(parents=True, exist_ok=True)
    stage_plot_dir.mkdir(parents=True, exist_ok=True)
    if not stage_records:
        _write_json(
            rankings_dir / f"{stage_name}_stage_fit_rankings.json",
            [],
        )
        return []

    params = _stage_simulation_parameters(target_params, cfg, stage_name)
    rankings: list[dict[str, Any]] = []
    for index, record in enumerate(stage_records, start=1):
        path = Path(str(record.get("path") or ""))
        if not path.exists():
            continue
        cif_id = _slug(str(record.get("cif_id") or path.stem))
        chemistry_fields = _path_chemistry_rank_fields(path)
        bragg_intensity_match: dict[str, Any] = {}
        bragg_intensity_penalty = 0.0
        comparison = None
        simulated_path = stage_rankings_dir / (
            f"{index:02d}_{cif_id}_simulated.nc"
        )
        try:
            simulated = simulate_giwaxs_image(path, params)
            simulated.to_netcdf(simulated_path)
            comparison = compare_giwaxs_images(
                target,
                simulated,
                target_label="Mock experiment",
                simulated_label=f"{stage_name}:{cif_id}",
            )
            simulated_rows = calculate_giwaxs_peak_rows(path, params)
            bragg_intensity_match = _score_bragg_peak_intensity_match(
                peaks,
                simulated_rows,
                tolerance=cfg.bragg_intensity_tolerance,
                max_peaks=cfg.bragg_intensity_max_peaks,
            )
            bragg_intensity_penalty = cfg.bragg_intensity_weight * float(
                bragg_intensity_match.get("intensity_match_penalty", 0.0)
                or 0.0
            )
            metrics = dict(comparison.metrics)
            metrics.update(
                {
                    "bragg_intensity_weighted_penalty": (
                        bragg_intensity_penalty
                    ),
                    "bragg_intensity_match_penalty": (
                        bragg_intensity_match.get("intensity_match_penalty")
                    ),
                    "bragg_intensity_relative_l1": (
                        bragg_intensity_match.get("relative_l1")
                    ),
                    "bragg_intensity_correlation": (
                        bragg_intensity_match.get("log_intensity_correlation")
                    ),
                    "bragg_intensity_matched_fraction": (
                        bragg_intensity_match.get("matched_peak_fraction")
                    ),
                }
            )
            item = {
                "stage": stage_name,
                "stage_policy": _stage_policy(stage_name),
                "generated_cif_id": cif_id,
                "path": str(path),
                "source_path": str(record.get("source_path") or path),
                "simulation_path": str(simulated_path),
                "parameters": params.as_dict(),
                "metrics": metrics,
                "bragg_intensity_match": bragg_intensity_match,
                "bragg_intensity_penalty": bragg_intensity_penalty,
                **chemistry_fields,
                "source_record": _stage_source_record_summary(record),
            }
        except Exception as exc:
            item = {
                "stage": stage_name,
                "stage_policy": _stage_policy(stage_name),
                "generated_cif_id": cif_id,
                "path": str(path),
                "source_path": str(record.get("source_path") or path),
                "simulation_path": str(simulated_path),
                "parameters": params.as_dict(),
                "metrics": {},
                "bragg_intensity_match": bragg_intensity_match,
                "bragg_intensity_penalty": bragg_intensity_penalty,
                **chemistry_fields,
                "source_record": _stage_source_record_summary(record),
                "error": str(exc),
            }
        item["_comparison"] = comparison
        rankings.append(item)

    rankings.sort(key=_comparison_sort_key)
    serializable = []
    for rank, item in enumerate(rankings, start=1):
        comparison = item.pop("_comparison", None)
        item["stage_rank"] = rank
        serializable.append(dict(item))
        if comparison is not None and rank <= cfg.comparison_plot_count:
            plot_path = stage_plot_dir / f"{stage_name}_rank_{rank:02d}.png"
            save_giwaxs_comparison_plot(
                comparison,
                plot_path,
                title=f"{stage_name} rank {rank}",
            )
            item["comparison_plot"] = str(plot_path)
            serializable[-1]["comparison_plot"] = str(plot_path)
    _write_json(
        rankings_dir / f"{stage_name}_stage_fit_rankings.json",
        serializable,
    )
    return serializable


def _stage_simulation_parameters(
    target_params: GIWAXSSimulationParameters,
    cfg: SyntheticRefinementConfig,
    stage_name: str,
) -> GIWAXSSimulationParameters:
    resolution_factor, hkl_factor = {
        "inorganic_scaffold": (0.50, 0.70),
        "organic_electron_proxy": (0.65, 0.82),
        "organic_replacement": (0.82, 0.92),
        "organic_rmc": (1.00, 1.00),
    }.get(stage_name, (0.70, 0.85))
    rows = max(48, int(round(cfg.detector_shape[0] * resolution_factor)))
    cols = max(64, int(round(cfg.detector_shape[1] * resolution_factor)))
    return GIWAXSSimulationParameters(
        sigma_theta=target_params.sigma_theta,
        sigma_phi=target_params.sigma_phi,
        sigma_r=target_params.sigma_r,
        hkl_extent=max(1, int(round(cfg.hkl_extent * hkl_factor))),
        theta_x_deg=target_params.theta_x_deg,
        theta_y_deg=target_params.theta_y_deg,
        qxy_min=cfg.qxy_range[0],
        qxy_max=cfg.qxy_range[1],
        qz_min=cfg.qz_range[0],
        qz_max=cfg.qz_range[1],
        resolution_z=rows,
        resolution_x=cols,
    )


def _stage_policy(stage_name: str) -> str:
    policies = {
        "inorganic_scaffold": (
            "independent Pb/Sn-halide scaffold, physical coordination "
            "constraints, Bragg position and relative intensity scoring"
        ),
        "organic_electron_proxy": (
            "charge-balanced organic centers represented by nearest-Z "
            "inorganic substitute atoms with steric/void limits"
        ),
        "organic_replacement": (
            "full molecules replace proxy centers under unit-cell symmetry, "
            "steric, molecular geometry, and hydrogen-bond constraints"
        ),
        "organic_rmc": (
            "bounded stochastic rigid-body molecular rotations/translations "
            "followed by chemistry and simulated-intensity scoring"
        ),
    }
    return policies.get(stage_name, "simulated stage comparison")


def _stage_source_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cif_id",
        "generated_cif_id",
        "coordinate_model",
        "solve_stoichiometry_hypothesis",
        "inorganic_stoichiometry",
        "organic_proxy_stoichiometry",
        "charge_balance",
        "charge_balance_penalty",
        "perovskite_motif_hypothesis",
        "solver_score",
        "score",
    )
    return {key: record[key] for key in keys if key in record}


def _physicalize_stage_records(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    suffix: str,
) -> list[dict[str, Any]]:
    physicalized: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source_path = Path(str(record.get("path") or ""))
        if not source_path.exists():
            continue
        cif_id = _slug(str(record.get("cif_id") or source_path.stem))
        output_path = output_dir / f"{cif_id}_{suffix}.cif"
        _write_physicalized_cif(source_path, output_path)
        payload = dict(record)
        payload["source_path"] = str(source_path)
        payload["raw_path"] = str(source_path)
        payload["path"] = str(output_path)
        payload["local_path"] = str(output_path)
        payload["structure_path"] = str(output_path)
        payload["cif_text"] = output_path.read_text(encoding="utf-8")
        payload["coordinate_model"] = (
            f"{payload.get('coordinate_model', 'stage')}_physicalized"
        )
        payload["chemistry_metrics"] = _cif_physical_chemistry_metrics(
            output_path
        )
        physicalized.append(payload)
    return physicalized


def generate_scaffold_candidate_cifs(
    spec: BenchmarkStructureSpec,
    candidates: list[Any],
    stoichiometry_hypotheses: list[str],
    cfg: SyntheticRefinementConfig,
) -> list[dict[str, Any]]:
    """Generate inorganic-only CIF records for fast scaffold
    diagnostics."""

    records: list[dict[str, Any]] = []
    stoichiometry_prior_weight = _stoichiometry_prior_weight(
        spec.organic_molecules,
    )
    motif_by_formula = _perovskite_motif_metadata_by_formula(
        spec.inorganic_atoms,
        spec.organic_molecules,
        limit=max(len(stoichiometry_hypotheses) * 3, 24),
    )
    for candidate in candidates:
        for hypothesis_rank, hypothesis in enumerate(
            stoichiometry_hypotheses,
            start=1,
        ):
            inorganic_formula = _inorganic_formula_from_hypothesis(
                hypothesis,
                spec.inorganic_atoms,
                spec.organic_molecules,
            )
            if not inorganic_formula:
                continue
            for record in generate_ranked_cif_records(
                candidate,
                atoms=spec.inorganic_atoms,
                molecules=[],
                stoichiometry=inorganic_formula,
                limit=cfg.records_per_stoichiometry,
                allow_explicit_templates=False,
                assume_unit_cell_symmetry=cfg.assume_unit_cell_symmetry,
            ):
                payload = dict(record)
                payload["cif_id"] = _slug(
                    f"{payload.get('cif_id')}_{_formula_token(inorganic_formula)}"
                )
                payload["solve_stoichiometry_hypothesis"] = hypothesis
                payload["inorganic_stoichiometry"] = inorganic_formula
                payload["solve_stoichiometry_hypothesis_rank"] = (
                    hypothesis_rank
                )
                payload["template_coordinates_allowed"] = False
                payload["coordinate_model"] = "inorganic_scaffold_only"
                payload["unit_cell_symmetry_assumed"] = (
                    cfg.assume_unit_cell_symmetry
                )
                motif = dict(motif_by_formula.get(hypothesis, {}))
                payload["perovskite_motif_hypothesis"] = motif or None
                payload["perovskite_motif_prior_penalty"] = float(
                    motif.get("motif_prior_penalty", 0.12) if motif else 0.12
                )
                payload["stoichiometry_prior_penalty"] = (
                    stoichiometry_prior_weight * (hypothesis_rank - 1)
                )
                payload["score"] = float(payload.get("score", math.inf)) + (
                    payload["stoichiometry_prior_penalty"]
                    + payload["perovskite_motif_prior_penalty"]
                )
                records.append(payload)
    records.sort(key=lambda item: float(item.get("score", math.inf)))
    return _diverse_scaffold_record_subset(
        records,
        limit=max(1, cfg.max_scaffolds_to_validate),
    )


def generate_organic_electron_proxy_cifs(
    spec: BenchmarkStructureSpec,
    candidates: list[Any],
    stoichiometry_hypotheses: list[str],
    cfg: SyntheticRefinementConfig,
) -> list[dict[str, Any]]:
    """Generate full-lattice drafts with organic molecules as Z proxies.

    The proxy atoms are a deliberately cheap electron-density stand-in: a
    charge-balanced stoichiometry hypothesis gives the number of molecule
    centers, and each molecule is replaced by the real element with the nearest
    atomic number to its charge-adjusted electron count.
    """

    if not tuple(spec.organic_molecules):
        return []
    records: list[dict[str, Any]] = []
    stoichiometry_prior_weight = _stoichiometry_prior_weight(
        spec.organic_molecules,
    )
    motif_by_formula = _perovskite_motif_metadata_by_formula(
        spec.inorganic_atoms,
        spec.organic_molecules,
        limit=max(len(stoichiometry_hypotheses) * 3, 24),
    )
    for candidate in candidates:
        for hypothesis_rank, hypothesis in enumerate(
            stoichiometry_hypotheses,
            start=1,
        ):
            inorganic_formula = _inorganic_formula_from_hypothesis(
                hypothesis,
                spec.inorganic_atoms,
                spec.organic_molecules,
            )
            if not inorganic_formula:
                continue
            proxy_plan = organic_electron_proxy_plan(
                hypothesis,
                spec.organic_molecules,
            )
            proxy_formula = str(proxy_plan.get("organic_proxy_formula") or "")
            proxy_atoms = proxy_plan.get("proxy_atoms", [])
            if not proxy_formula or not proxy_atoms:
                continue
            organic_proxy_stoichiometry = f"{inorganic_formula}{proxy_formula}"
            proxy_elements = [
                str(item.get("proxy_element") or "")
                for item in proxy_atoms
                if item.get("proxy_element")
            ]
            atom_list = _unique_element_sequence(
                tuple(spec.inorganic_atoms) + tuple(proxy_elements)
            )
            charge_balance = _formula_charge_balance(hypothesis)
            charge_balance_penalty = float(
                charge_balance.get("penalty", 0.0) or 0.0
            )
            stoichiometry_prior_penalty = stoichiometry_prior_weight * (
                hypothesis_rank - 1
            )
            for record in generate_ranked_cif_records(
                candidate,
                atoms=atom_list,
                molecules=[],
                stoichiometry=organic_proxy_stoichiometry,
                limit=cfg.records_per_stoichiometry,
                allow_explicit_templates=False,
                assume_unit_cell_symmetry=cfg.assume_unit_cell_symmetry,
            ):
                payload = dict(record)
                payload["cif_id"] = _slug(
                    f"{payload.get('cif_id')}_"
                    f"{_formula_token(organic_proxy_stoichiometry)}_"
                    "organic_proxy"
                )
                payload["solve_stoichiometry_hypothesis"] = hypothesis
                payload["solve_stoichiometry_hypothesis_rank"] = (
                    hypothesis_rank
                )
                payload["charge_balance"] = charge_balance
                payload["charge_balance_penalty"] = charge_balance_penalty
                payload["inorganic_stoichiometry"] = inorganic_formula
                payload["organic_proxy_formula"] = proxy_formula
                payload["organic_proxy_stoichiometry"] = (
                    organic_proxy_stoichiometry
                )
                payload["organic_electron_proxy_plan"] = proxy_plan
                payload["template_coordinates_allowed"] = False
                payload["coordinate_model"] = "organic_electron_proxy"
                payload["unit_cell_symmetry_assumed"] = (
                    cfg.assume_unit_cell_symmetry
                )
                motif = dict(motif_by_formula.get(hypothesis, {}))
                payload["perovskite_motif_hypothesis"] = motif or None
                payload["perovskite_motif_prior_penalty"] = float(
                    motif.get("motif_prior_penalty", 0.12) if motif else 0.12
                )
                payload["stoichiometry_prior_penalty"] = (
                    stoichiometry_prior_penalty
                )
                payload["score"] = (
                    float(payload.get("score", math.inf))
                    + stoichiometry_prior_penalty
                    + charge_balance_penalty
                    + payload["perovskite_motif_prior_penalty"]
                )
                records.append(payload)
    records.sort(key=lambda item: float(item.get("score", math.inf)))
    return _diverse_organic_proxy_record_subset(
        records,
        limit=max(1, cfg.max_organic_proxy_cifs_to_compare),
    )


def generate_organic_replacement_cifs(
    spec: BenchmarkStructureSpec,
    candidates: list[Any],
    stoichiometry_hypotheses: list[str],
    cfg: SyntheticRefinementConfig,
) -> list[dict[str, Any]]:
    """Generate full organic replacement CIFs from charge-balanced
    guesses."""

    molecules = _molecule_records(spec.organic_molecules)
    if not molecules:
        return []
    records: list[dict[str, Any]] = []
    stoichiometry_prior_weight = _stoichiometry_prior_weight(
        spec.organic_molecules,
    )
    motif_by_formula = _perovskite_motif_metadata_by_formula(
        spec.inorganic_atoms,
        spec.organic_molecules,
        limit=max(len(stoichiometry_hypotheses) * 3, 24),
    )
    for candidate in candidates:
        for hypothesis_rank, hypothesis in enumerate(
            stoichiometry_hypotheses,
            start=1,
        ):
            charge_balance = _formula_charge_balance(hypothesis)
            charge_balance_penalty = float(
                charge_balance.get("penalty", 0.0) or 0.0
            )
            stoichiometry_prior_penalty = stoichiometry_prior_weight * (
                hypothesis_rank - 1
            )
            motif = dict(motif_by_formula.get(hypothesis, {}))
            motif_penalty = float(
                motif.get("motif_prior_penalty", 0.12) if motif else 0.12
            )
            for record in generate_ranked_cif_records(
                candidate,
                atoms=spec.inorganic_atoms,
                molecules=molecules,
                stoichiometry=hypothesis,
                limit=cfg.records_per_stoichiometry,
                allow_explicit_templates=False,
                assume_unit_cell_symmetry=cfg.assume_unit_cell_symmetry,
            ):
                payload = dict(record)
                payload["cif_id"] = _slug(
                    f"{payload.get('cif_id')}_{_formula_token(hypothesis)}_"
                    "organic_replacement"
                )
                payload["solve_stoichiometry_hypothesis"] = hypothesis
                payload["solve_stoichiometry_hypothesis_rank"] = (
                    hypothesis_rank
                )
                payload["charge_balance"] = charge_balance
                payload["charge_balance_penalty"] = charge_balance_penalty
                payload["template_coordinates_allowed"] = False
                payload["coordinate_model"] = "full_organic_replacement"
                payload["unit_cell_symmetry_assumed"] = (
                    cfg.assume_unit_cell_symmetry
                )
                payload["organic_replacement_policy"] = (
                    "full molecules placed after charge-balanced proxy "
                    "stage; molecule bodies are checked against steric, "
                    "DFIX/DANG/SADI/FLAT-like, and hydrogen-bond priors"
                )
                payload["perovskite_motif_hypothesis"] = motif or None
                payload["perovskite_motif_prior_penalty"] = motif_penalty
                payload["stoichiometry_prior_penalty"] = (
                    stoichiometry_prior_penalty
                )
                payload["score"] = (
                    float(payload.get("score", math.inf))
                    + stoichiometry_prior_penalty
                    + charge_balance_penalty
                    + motif_penalty
                )
                records.append(payload)
    records.sort(key=lambda item: float(item.get("score", math.inf)))
    return _diverse_organic_replacement_record_subset(
        records,
        limit=max(1, cfg.max_organic_replacement_cifs_to_compare),
    )


def generate_organic_rmc_variant_cifs(
    source_path: Path,
    *,
    fileset_id: str,
    rng: np.random.Generator,
    cfg: SyntheticRefinementConfig,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Generate stochastic rigid-body organic variants for final
    refinement."""

    if not source_path.exists():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for step_index in range(1, cfg.organic_rmc_steps + 1):
        cif_id = _slug(
            f"{fileset_id}_{source_path.stem}_organic_rmc_{step_index:03d}"
        )
        raw_path = raw_dir / f"{cif_id}_raw.cif"
        refined_path = output_dir / f"{cif_id}_geometry.cif"
        transform = _write_organic_rmc_variant_cif(
            source_path,
            raw_path,
            rng=rng,
            translation_step=cfg.organic_rmc_translation_step,
            rotation_step_deg=cfg.organic_rmc_rotation_step_deg,
        )
        _write_physicalized_cif(raw_path, refined_path)
        chemistry_metrics = _cif_physical_chemistry_metrics(refined_path)
        physical_penalty = float(
            chemistry_metrics.get("physical_penalty", 0.0) or 0.0
        )
        records.append(
            {
                "cif_id": cif_id,
                "path": str(refined_path),
                "local_path": str(refined_path),
                "structure_path": str(refined_path),
                "source_path": str(source_path),
                "raw_path": str(raw_path),
                "coordinate_model": "organic_rmc_stochastic",
                "score": physical_penalty + 0.001 * step_index,
                "physical_penalty": physical_penalty,
                "chemistry_metrics": chemistry_metrics,
                "organic_rmc_transform": transform,
                "cif_text": refined_path.read_text(encoding="utf-8"),
            }
        )
    return records


def organic_electron_proxy_plan(
    stoichiometry_hypothesis: str,
    organic_molecules: Iterable[str],
) -> dict[str, Any]:
    """Return the proxy atom plan implied by one stoichiometry
    hypothesis."""

    molecule_counts = _organic_molecule_counts_from_hypothesis(
        stoichiometry_hypothesis,
        organic_molecules,
    )
    proxy_atoms: list[dict[str, Any]] = []
    proxy_counts: dict[str, float] = {}
    total_electrons = 0.0
    for label in sorted(molecule_counts):
        count = float(molecule_counts[label])
        if count <= 0.0:
            continue
        metadata = _reference_molecule_metadata(label)
        formula = str(metadata.get("formula") or "")
        if not formula:
            continue
        neutral_electrons = formula_electron_count(formula)
        nominal_charge = _organic_molecule_nominal_charge(label)
        charge_adjusted = max(
            1,
            int(round(neutral_electrons - nominal_charge)),
        )
        proxy = electron_proxy_element_for_count(charge_adjusted)
        proxy_element = str(proxy["element"])
        proxy_counts[proxy_element] = proxy_counts.get(proxy_element, 0.0) + (
            count
        )
        total_electrons += count * float(charge_adjusted)
        proxy_atoms.append(
            {
                "molecule_label": label,
                "molecule_count": count,
                "molecule_formula": formula,
                "neutral_electron_count": neutral_electrons,
                "nominal_charge": nominal_charge,
                "charge_adjusted_electron_count": charge_adjusted,
                "proxy_element": proxy_element,
                "proxy_atomic_number": proxy["atomic_number"],
                "electron_count_error": proxy["electron_count_error"],
                "proxy_selection_model": proxy["selection_model"],
            }
        )
    return {
        "stoichiometry_hypothesis": stoichiometry_hypothesis,
        "electron_count_model": "neutral_formula_electrons_minus_nominal_charge",
        "molecule_counts": molecule_counts,
        "proxy_atoms": proxy_atoms,
        "proxy_element_counts": proxy_counts,
        "organic_proxy_formula": _formula_from_counts(proxy_counts),
        "total_proxy_sites": float(sum(proxy_counts.values())),
        "total_charge_adjusted_electrons": total_electrons,
    }


def formula_electron_count(formula: str) -> int:
    """Count neutral formula electrons from element atomic numbers."""

    total = 0.0
    for element, count in _parse_element_formula_counts(formula).items():
        total += _ATOMIC_NUMBERS.get(element, 0) * float(count)
    return int(round(total))


def electron_proxy_element_for_count(
    electron_count: int | float,
) -> dict[str, Any]:
    """Map an organic electron count to a chemically usable proxy
    element."""

    target = int(round(float(electron_count)))
    target = max(1, target)
    element, atomic_number = min(
        _ORGANIC_ELECTRON_PROXY_NUMBERS.items(),
        key=lambda item: (abs(item[1] - target), item[1]),
    )
    return {
        "element": element,
        "atomic_number": atomic_number,
        "target_electron_count": target,
        "electron_count_error": atomic_number - target,
        "selection_model": (
            "nearest finite-electronegativity organic-center proxy; "
            "noble gases and halides excluded"
        ),
    }


def _diverse_scaffold_record_subset(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(record: dict[str, Any]) -> None:
        if len(selected) >= limit:
            return
        cif_id = str(record.get("cif_id") or "")
        if cif_id in selected_ids:
            return
        selected.append(record)
        selected_ids.add(cif_id)

    seen_stoichiometry: set[str] = set()
    for record in records:
        stoichiometry = str(record.get("inorganic_stoichiometry") or "")
        if stoichiometry in seen_stoichiometry:
            continue
        add(record)
        seen_stoichiometry.add(stoichiometry)
        if len(seen_stoichiometry) >= max(1, limit // 2):
            break

    seen_candidates: set[str] = set()
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id in seen_candidates:
            continue
        add(record)
        seen_candidates.add(candidate_id)
        if len(selected) >= limit:
            break

    for record in records:
        add(record)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: float(item.get("score", math.inf)))
    return selected


def _diverse_organic_proxy_record_subset(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(record: dict[str, Any]) -> None:
        if len(selected) >= limit:
            return
        cif_id = str(record.get("cif_id") or "")
        if cif_id in selected_ids:
            return
        selected.append(record)
        selected_ids.add(cif_id)

    seen_proxy_formulas: set[str] = set()
    for record in records:
        formula = str(record.get("organic_proxy_stoichiometry") or "")
        if formula in seen_proxy_formulas:
            continue
        add(record)
        seen_proxy_formulas.add(formula)
        if len(selected) >= max(1, limit // 2):
            break

    seen_candidates: set[str] = set()
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id in seen_candidates:
            continue
        add(record)
        seen_candidates.add(candidate_id)
        if len(selected) >= limit:
            break

    for record in records:
        add(record)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: float(item.get("score", math.inf)))
    return selected


def _diverse_organic_replacement_record_subset(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(record: dict[str, Any]) -> None:
        if len(selected) >= limit:
            return
        cif_id = str(record.get("cif_id") or "")
        if cif_id in selected_ids:
            return
        selected.append(record)
        selected_ids.add(cif_id)

    seen_hypotheses: set[str] = set()
    for record in records:
        hypothesis = str(record.get("solve_stoichiometry_hypothesis") or "")
        if hypothesis in seen_hypotheses:
            continue
        add(record)
        seen_hypotheses.add(hypothesis)
        if len(selected) >= max(1, limit // 2):
            break

    seen_candidates: set[str] = set()
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id in seen_candidates:
            continue
        add(record)
        seen_candidates.add(candidate_id)
        if len(selected) >= limit:
            break

    for record in records:
        add(record)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: float(item.get("score", math.inf)))
    return selected


def _write_organic_rmc_variant_cif(
    source_path: Path,
    output_path: Path,
    *,
    rng: np.random.Generator,
    translation_step: float,
    rotation_step_deg: float,
) -> dict[str, Any]:
    rows, lattice = _read_simple_cif_sites(source_path)
    lines = source_path.read_text(encoding="utf-8").splitlines()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if lattice is None or not rows:
        output_path.write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding="utf-8",
        )
        return {"status": "no_lattice_or_sites", "body_count": 0}

    bodies: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        token = _molecular_body_token(str(row["label"]))
        if token:
            bodies.setdefault(token, []).append(row)
    if not bodies:
        output_path.write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding="utf-8",
        )
        return {"status": "no_organic_bodies", "body_count": 0}

    inv_lattice = np.linalg.inv(lattice)
    transforms = []
    max_translation = max(0.0, float(translation_step))
    max_rotation = math.radians(max(0.0, float(rotation_step_deg)))
    for body_token, body_rows in sorted(bodies.items()):
        coords = np.asarray(
            [np.asarray(row["frac"], dtype=float) for row in body_rows],
            dtype=float,
        )
        center = _fractional_center(coords)
        translation = rng.uniform(
            -max_translation,
            max_translation,
            size=3,
        )
        axis = rng.normal(size=3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-12:
            axis = np.asarray([0.0, 0.0, 1.0])
        else:
            axis = axis / axis_norm
        angle = float(rng.uniform(-max_rotation, max_rotation))
        rotation = _axis_angle_matrix(axis, angle)
        offsets = coords - center
        offsets -= np.round(offsets)
        rotated_offsets = (offsets @ lattice) @ rotation.T
        rotated_frac = rotated_offsets @ inv_lattice
        transformed = (center + translation + rotated_frac) % 1.0
        for row, frac in zip(body_rows, transformed, strict=True):
            row_index = int(row["row_index"])
            if row_index < 0 or row_index >= len(lines):
                continue
            parts = lines[row_index].split()
            if len(parts) < 6:
                continue
            parts[2:5] = [f"{value:.6f}" for value in frac]
            lines[row_index] = " ".join(parts)
        transforms.append(
            {
                "body": body_token,
                "translation_fractional": [
                    float(value) for value in translation
                ],
                "rotation_axis": [float(value) for value in axis],
                "rotation_angle_deg": math.degrees(angle),
            }
        )
    output_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return {
        "status": "computed",
        "body_count": len(bodies),
        "max_translation_fractional": max_translation,
        "max_rotation_deg": math.degrees(max_rotation),
        "transforms": transforms,
    }


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z = axis / norm
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=float,
    )


def _stoichiometry_prior_weight(
    organic_molecules: Iterable[str],
) -> float:
    donor_cations = {"MA", "FA", "BA"}
    labels = {
        _formula_token(label).upper()
        for label in organic_molecules
        if str(label).strip()
    }
    if labels & donor_cations:
        return 0.30
    return 0.15


def validate_inorganic_scaffold(
    generated_scaffold: Path,
    reference_cif: Path,
    inorganic_elements: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare an inorganic scaffold against the reference inorganic
    sites."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_scaffold = output_dir / "reference_inorganic_scaffold.cif"
    _write_normalized_inorganic_cif(
        reference_cif,
        reference_scaffold,
        inorganic_elements,
    )
    try:
        comparison = compare_cif_atom_coordinates(
            generated_scaffold,
            reference_scaffold,
        )
        lattice_metrics = _lattice_validation_metrics(comparison)
        composition_metrics = _composition_validation_metrics(comparison)
        status = "validated"
    except Exception as exc:
        comparison = {"error": str(exc)}
        lattice_metrics = {}
        composition_metrics = {}
        status = "validation_error"
    pair_distribution = _pair_distribution_validation_metrics(
        generated_scaffold,
        reference_scaffold,
    )
    coordinate = (
        comparison.get("coordinate_match", {})
        if isinstance(comparison, dict)
        else {}
    )
    cartesian_rms = coordinate.get("cartesian_rms_angstrom")
    attained = bool(
        status == "validated"
        and float(lattice_metrics.get("sorted_abc_relative_error", math.inf))
        <= 0.15
        and float(
            composition_metrics.get("element_count_relative_error", math.inf)
        )
        <= 0.25
        and (cartesian_rms is None or float(cartesian_rms) <= 2.0)
    )
    return {
        "status": status,
        "attained_inorganic_scaffold": attained,
        "generated_scaffold": str(generated_scaffold),
        "reference_scaffold": str(reference_scaffold),
        "cif_comparison": comparison,
        "lattice_metrics": lattice_metrics,
        "composition_metrics": composition_metrics,
        "pair_distribution_metrics": pair_distribution,
    }


def _attach_reference_structure_comparisons(
    stage_rankings: dict[str, list[dict[str, Any]]],
    spec: BenchmarkStructureSpec,
    output_dir: Path,
) -> dict[str, Any]:
    """Attach full-reference comparisons to comparable staged
    outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for stage_name in ("organic_replacement", "organic_rmc"):
        records = stage_rankings.get(stage_name, [])
        stage_rows = []
        for rank, record in enumerate(records[:3], start=1):
            path = Path(str(record.get("path") or ""))
            comparison = _reference_structure_comparison(
                path,
                spec.cif_path,
            )
            record["reference_structure_comparison"] = comparison
            stage_rows.append(
                {
                    "stage_rank": rank,
                    "path": str(path),
                    "comparison": comparison,
                }
            )
        if stage_rows:
            _write_json(
                output_dir / f"{stage_name}_known_structure_comparison.json",
                stage_rows,
            )
            best = stage_rows[0]["comparison"]
            coordinate = best.get("cif_comparison", {}).get(
                "coordinate_match",
                {},
            )
            summary[stage_name] = {
                "status": best.get("status"),
                "rank_1_path": stage_rows[0]["path"],
                "rank_1_cartesian_rms_angstrom": coordinate.get(
                    "cartesian_rms_angstrom"
                ),
                "rank_1_unmatched_count": coordinate.get("unmatched_count"),
                "rank_1_sorted_abc_relative_error": best.get(
                    "lattice_metrics",
                    {},
                ).get("sorted_abc_relative_error"),
                "rank_1_element_count_relative_error": best.get(
                    "composition_metrics",
                    {},
                ).get("element_count_relative_error"),
            }
    if summary:
        _write_json(output_dir / "summary.json", summary)
    return summary


def _reference_structure_comparison(
    generated_path: Path,
    reference_cif: Path,
) -> dict[str, Any]:
    if not generated_path.exists():
        return {
            "status": "missing_generated_cif",
            "generated": str(generated_path),
            "reference": str(reference_cif),
        }
    try:
        comparison = compare_cif_atom_coordinates(
            generated_path,
            reference_cif,
        )
        return {
            "status": "validated",
            "generated": str(generated_path),
            "reference": str(reference_cif),
            "cif_comparison": comparison,
            "lattice_metrics": _lattice_validation_metrics(comparison),
            "composition_metrics": _composition_validation_metrics(comparison),
        }
    except Exception as exc:
        return {
            "status": "validation_error",
            "generated": str(generated_path),
            "reference": str(reference_cif),
            "error": str(exc),
        }


def _store_synthetic_project_analysis(
    project: ProjectState,
    data_id: str,
    fileset_id: str,
    spec: BenchmarkStructureSpec,
    peaks: list[StructurePeak],
    families: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    stoichiometry_hypotheses: list[str],
    perovskite_motif_hypotheses: list[dict[str, Any]],
    peak_truth: dict[str, Any],
    family_assessment: dict[str, Any],
    candidate_truth: dict[str, Any],
    scaffold_rankings: list[dict[str, Any]],
    organic_proxy_rankings: list[dict[str, Any]],
    organic_replacement_rankings: list[dict[str, Any]],
    organic_rmc_rankings: list[dict[str, Any]],
    stage_rankings: dict[str, list[dict[str, Any]]],
    image_rankings: list[dict[str, Any]],
) -> None:
    from ewald.benchmark.structure_benchmark import _store_peak_records

    _store_peak_records(project, data_id, peaks)
    structure = project.analysis_results.setdefault(STRUCTURE_ANALYSIS_KEY, {})
    analysis = structure.setdefault(data_id, {})
    analysis["families"] = families
    analysis["candidates"] = candidates
    project.analysis_results.setdefault("synthetic_refinement", {})[
        fileset_id
    ] = {
        "fileset_id": fileset_id,
        "constraints": _solve_constraints(spec),
        "peak_truth": peak_truth,
        "peak_family_assessment": family_assessment,
        "candidate_truth_assessment": candidate_truth,
        "stoichiometry_hypotheses": stoichiometry_hypotheses,
        "perovskite_motif_hypotheses": perovskite_motif_hypotheses,
        "inorganic_scaffold_rankings": scaffold_rankings,
        "organic_electron_proxy_rankings": organic_proxy_rankings,
        "organic_replacement_rankings": organic_replacement_rankings,
        "organic_rmc_rankings": organic_rmc_rankings,
        "refinement_stage_rankings": stage_rankings,
        "generated_cif_rankings": image_rankings,
    }


def _append_synthetic_fileset_logbook(
    logbook: Path,
    summary: dict[str, Any],
    scaffold_rankings: list[dict[str, Any]],
) -> None:
    top = summary.get("top_candidate") or {}
    truth = summary.get("top_candidate_truth_assignment") or {}
    scaffold_validation = (
        summary.get("best_inorganic_scaffold_validation") or {}
    )
    lattice = scaffold_validation.get("lattice_metrics", {})
    coordinate = (
        scaffold_validation.get("cif_comparison", {}).get(
            "coordinate_match", {}
        )
        if isinstance(scaffold_validation.get("cif_comparison"), dict)
        else {}
    )
    peak_truth = summary.get("peak_truth_match_summary", {})
    family = summary.get("family_assessment", {})
    best = scaffold_rankings[0] if scaffold_rankings else {}
    proxy = (summary.get("organic_electron_proxy_rankings") or [{}])[0]
    proxy_plan = proxy.get("organic_electron_proxy_plan", {})
    bragg = summary.get("best_bragg_intensity_match") or {}
    replacement = summary.get("best_organic_replacement") or {}
    rmc = summary.get("best_organic_rmc") or {}
    lines = [
        "",
        f"## {summary['fileset_id']}",
        "",
        f"- Solve order: {summary['solve_order']}",
        f"- Mock TIFF: `{summary['mock_tiff']}`",
        f"- Project: `{summary['project']}`",
        f"- Peak detection plot: `{summary['peak_detection_plot']}`",
        f"- Solver input mode: `{summary.get('solver_input_mode', 'unknown')}`",
        (
            "- Orientation: "
            f"theta_x={summary['orientation']['theta_x_deg']:.3f} deg, "
            f"theta_y={summary['orientation']['theta_y_deg']:.3f} deg"
        ),
        (
            "- Peak truth match: "
            f"precision={float(peak_truth.get('precision', math.nan)):.3g}, "
            f"recall={float(peak_truth.get('recall', math.nan)):.3g}"
        ),
        (
            "- Family assessment: "
            f"{family.get('high_purity_family_count', 0)} high-purity "
            f"families out of {family.get('truth_scored_family_count', 0)} "
            "truth-scored families"
        ),
        (
            "- Top lattice candidate: "
            + (
                f"{top.get('crystal_system')} a={top.get('a'):.4g} "
                f"b={top.get('b'):.4g} c={top.get('c'):.4g} "
                f"score={top.get('score'):.4g}"
                if top
                else "none"
            )
        ),
        (
            "- Top-candidate hkl-family accuracy: "
            f"{float(truth.get('hkl_family_accuracy', math.nan)):.3g}"
        ),
        ("- Best inorganic scaffold: " f"`{best.get('path', 'none')}`"),
        (
            "- Inorganic scaffold validation: sorted abc relative error="
            f"{float(lattice.get('sorted_abc_relative_error', math.nan)):.4g}, "
            "cartesian RMS="
            f"{float(coordinate.get('cartesian_rms_angstrom', math.nan)):.4g} A"
        ),
        (
            "- Bragg intensity match: "
            f"penalty={float(summary.get('best_bragg_intensity_penalty', math.nan)):.3g}, "
            f"corr={float(bragg.get('log_intensity_correlation', math.nan)):.3g}, "
            f"matched={float(bragg.get('matched_peak_fraction', math.nan)):.3g}"
        ),
        (
            "- Organic electron proxy: "
            + (
                f"{proxy.get('organic_proxy_stoichiometry')} "
                f"({proxy_plan.get('total_charge_adjusted_electrons', 0):.4g} e)"
                if proxy
                else "none"
            )
        ),
        (
            "- Organic replacement stage: "
            + (
                f"rank-1 `{replacement.get('path')}` "
                f"corr={float((replacement.get('metrics') or {}).get('correlation', math.nan)):.3g}"
                if replacement
                else "none"
            )
        ),
        (
            "- Organic RMC stage: "
            + (
                f"rank-1 `{rmc.get('path')}` "
                f"corr={float((rmc.get('metrics') or {}).get('correlation', math.nan)):.3g}"
                if rmc
                else "none"
            )
        ),
        "",
    ]
    _append_logbook(logbook, lines)


def _aggregate_synthetic_findings(
    filesets: list[dict[str, Any]],
) -> dict[str, Any]:
    if not filesets:
        return {}

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for fileset in filesets:
            value: Any = fileset
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                values.append(parsed)
        return float(np.mean(values)) if values else math.nan

    scaffold_rms = []
    scaffold_lattice_error = []
    attained = 0
    for fileset in filesets:
        validation = fileset.get("best_inorganic_scaffold_validation", {})
        if validation.get("attained_inorganic_scaffold"):
            attained += 1
        lattice = validation.get("lattice_metrics", {})
        try:
            value = float(lattice.get("sorted_abc_relative_error"))
            if np.isfinite(value):
                scaffold_lattice_error.append(value)
        except (TypeError, ValueError):
            pass
        comparison = validation.get("cif_comparison", {})
        coordinate = (
            comparison.get("coordinate_match", {})
            if isinstance(comparison, dict)
            else {}
        )
        try:
            value = float(coordinate.get("cartesian_rms_angstrom"))
            if np.isfinite(value):
                scaffold_rms.append(value)
        except (TypeError, ValueError):
            pass
    peak_precision = mean(("peak_truth_match_summary", "precision"))
    peak_recall = mean(("peak_truth_match_summary", "recall"))
    intensity_weighted_recall = mean(
        ("peak_truth_match_summary", "intensity_weighted_recall")
    )
    top50_truth_recall = mean(
        ("peak_truth_match_summary", "top50_truth_recall")
    )
    top100_truth_recall = mean(
        ("peak_truth_match_summary", "top100_truth_recall")
    )
    peak_position_mean_radial_error = mean(
        ("peak_position_error_summary", "mean_radial_error")
    )
    peak_position_rms_radial_error = mean(
        ("peak_position_error_summary", "rms_radial_error")
    )
    peak_position_qxy_bias = mean(
        ("peak_position_error_summary", "mean_delta_abs_qxy")
    )
    peak_position_qz_bias = mean(
        ("peak_position_error_summary", "mean_delta_qz")
    )
    hkl_accuracy = mean(
        ("top_candidate_truth_assignment", "hkl_family_accuracy")
    )
    bragg_intensity_penalty = mean(("best_bragg_intensity_penalty",))
    bragg_intensity_correlation = mean(
        ("best_bragg_intensity_match", "log_intensity_correlation")
    )
    bragg_intensity_matched_fraction = mean(
        ("best_bragg_intensity_match", "matched_peak_fraction")
    )
    family_purity = mean(("family_assessment", "mean_truth_family_purity"))
    weighted_family_purity = mean(
        ("family_assessment", "weighted_mean_truth_family_purity")
    )
    multi_family_purity = mean(
        ("family_assessment", "mean_multi_truth_family_purity")
    )
    texture_breakdown = _aggregate_by_texture_mode(filesets)
    observations = []
    if peak_recall < 0.35 and (
        not np.isfinite(intensity_weighted_recall)
        or intensity_weighted_recall < 0.75
    ):
        observations.append(
            "Peak detection is sparse even after intensity weighting; "
            "peak-shape fitting should be tested before broadening lattice "
            "search."
        )
    elif peak_recall < 0.35:
        observations.append(
            "Raw peak recall is sparse, but intensity-weighted/top-ranked "
            "truth recovery is much better; use strong-peak diagnostics when "
            "judging the peak finder."
        )
    if np.isfinite(weighted_family_purity) and weighted_family_purity < 0.60:
        observations.append(
            "Peak-family grouping is the current weak link; mixed hkl-family "
            "clusters should be split or down-weighted before indexing."
        )
    if (
        np.isfinite(bragg_intensity_correlation)
        and bragg_intensity_correlation < 0.35
    ):
        observations.append(
            "Bragg peak intensity matching is now active in scaffold ranking; "
            "low intensity correlation suggests atom placement or motif "
            "weights need refinement."
        )
    if hkl_accuracy < 0.40:
        observations.append(
            "Top lattice candidates still carry hkl-family ambiguity; the "
            "axis-step cap reduces oversized-cell bias, but candidate ranking "
            "should keep penalizing high-order assignments."
        )
    if scaffold_rms and float(np.nanmean(scaffold_rms)) > 2.0:
        observations.append(
            "Inorganic scaffold coordinates remain the limiting error after "
            "cell recovery; stronger Pb/Sn-halide motif placement is the next "
            "highest-leverage algorithm target."
        )
    if not observations:
        observations.append(
            "Peak, family, hkl, and scaffold metrics are internally "
            "consistent for this batch."
        )
    return {
        "fileset_count": len(filesets),
        "mean_peak_precision": peak_precision,
        "mean_peak_recall": peak_recall,
        "mean_peak_intensity_weighted_recall": intensity_weighted_recall,
        "mean_peak_top50_truth_recall": top50_truth_recall,
        "mean_peak_top100_truth_recall": top100_truth_recall,
        "mean_peak_position_radial_error": peak_position_mean_radial_error,
        "mean_peak_position_rms_radial_error": (
            peak_position_rms_radial_error
        ),
        "mean_peak_position_qxy_abs_bias": peak_position_qxy_bias,
        "mean_peak_position_qz_bias": peak_position_qz_bias,
        "mean_top_candidate_hkl_family_accuracy": hkl_accuracy,
        "mean_best_bragg_intensity_penalty": bragg_intensity_penalty,
        "mean_best_bragg_intensity_correlation": bragg_intensity_correlation,
        "mean_best_bragg_intensity_matched_fraction": (
            bragg_intensity_matched_fraction
        ),
        "mean_peak_family_truth_purity": family_purity,
        "mean_peak_family_truth_weighted_purity": weighted_family_purity,
        "mean_multi_truth_peak_family_purity": multi_family_purity,
        "mean_inorganic_scaffold_cartesian_rms_angstrom": (
            float(np.mean(scaffold_rms)) if scaffold_rms else math.nan
        ),
        "mean_inorganic_scaffold_sorted_abc_relative_error": (
            float(np.mean(scaffold_lattice_error))
            if scaffold_lattice_error
            else math.nan
        ),
        "attained_inorganic_scaffold_count": attained,
        "texture_mode_breakdown": texture_breakdown,
        "observations": observations,
    }


def _aggregate_by_texture_mode(
    filesets: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fileset in filesets:
        orientation = fileset.get("orientation", {})
        texture = (
            str(orientation.get("texture_mode") or "unknown")
            if isinstance(orientation, dict)
            else "unknown"
        )
        grouped.setdefault(texture, []).append(fileset)

    def finite_mean(values: list[Any]) -> float:
        parsed = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                parsed.append(number)
        return float(np.mean(parsed)) if parsed else math.nan

    records: dict[str, Any] = {}
    for texture, texture_filesets in sorted(grouped.items()):
        records[texture] = {
            "fileset_count": len(texture_filesets),
            "mean_peak_precision": finite_mean(
                [
                    item.get("peak_truth_match_summary", {}).get("precision")
                    for item in texture_filesets
                ]
            ),
            "mean_peak_recall": finite_mean(
                [
                    item.get("peak_truth_match_summary", {}).get("recall")
                    for item in texture_filesets
                ]
            ),
            "mean_peak_intensity_weighted_recall": finite_mean(
                [
                    item.get("peak_truth_match_summary", {}).get(
                        "intensity_weighted_recall"
                    )
                    for item in texture_filesets
                ]
            ),
            "mean_peak_top50_truth_recall": finite_mean(
                [
                    item.get("peak_truth_match_summary", {}).get(
                        "top50_truth_recall"
                    )
                    for item in texture_filesets
                ]
            ),
            "mean_peak_position_radial_error": finite_mean(
                [
                    item.get("peak_position_error_summary", {}).get(
                        "mean_radial_error"
                    )
                    for item in texture_filesets
                ]
            ),
            "mean_peak_position_qxy_abs_bias": finite_mean(
                [
                    item.get("peak_position_error_summary", {}).get(
                        "mean_delta_abs_qxy"
                    )
                    for item in texture_filesets
                ]
            ),
            "mean_peak_position_qz_bias": finite_mean(
                [
                    item.get("peak_position_error_summary", {}).get(
                        "mean_delta_qz"
                    )
                    for item in texture_filesets
                ]
            ),
            "mean_best_bragg_intensity_penalty": finite_mean(
                [
                    item.get("best_bragg_intensity_penalty")
                    for item in texture_filesets
                ]
            ),
            "mean_best_bragg_intensity_correlation": finite_mean(
                [
                    item.get("best_bragg_intensity_match", {}).get(
                        "log_intensity_correlation"
                    )
                    for item in texture_filesets
                ]
            ),
            "mean_scaffold_rms": finite_mean(
                [_fileset_scaffold_rms(item) for item in texture_filesets]
            ),
        }
    return records


def _aggregate_logbook_lines(aggregate: dict[str, Any]) -> list[str]:
    if not aggregate:
        return ["No filesets were processed.", ""]
    lines = [
        f"- Filesets: {aggregate.get('fileset_count', 0)}",
        (
            "- Mean peak precision/recall: "
            f"{aggregate.get('mean_peak_precision', math.nan):.3g}/"
            f"{aggregate.get('mean_peak_recall', math.nan):.3g}"
        ),
        (
            "- Mean peak intensity-weighted/top-50 recall: "
            f"{aggregate.get('mean_peak_intensity_weighted_recall', math.nan):.3g}/"
            f"{aggregate.get('mean_peak_top50_truth_recall', math.nan):.3g}"
        ),
        (
            "- Mean peak position radial/RMS error: "
            f"{aggregate.get('mean_peak_position_radial_error', math.nan):.3g}/"
            f"{aggregate.get('mean_peak_position_rms_radial_error', math.nan):.3g}"
        ),
        (
            "- Mean peak position qxy/qz bias: "
            f"{aggregate.get('mean_peak_position_qxy_abs_bias', math.nan):.3g}/"
            f"{aggregate.get('mean_peak_position_qz_bias', math.nan):.3g}"
        ),
        (
            "- Mean top-candidate hkl-family accuracy: "
            f"{aggregate.get('mean_top_candidate_hkl_family_accuracy', math.nan):.3g}"
        ),
        (
            "- Mean best-scaffold Bragg intensity penalty/correlation: "
            f"{aggregate.get('mean_best_bragg_intensity_penalty', math.nan):.3g}/"
            f"{aggregate.get('mean_best_bragg_intensity_correlation', math.nan):.3g}"
        ),
        (
            "- Mean peak-family weighted purity: "
            f"{aggregate.get('mean_peak_family_truth_weighted_purity', math.nan):.3g}"
        ),
        (
            "- Mean scaffold cartesian RMS: "
            f"{aggregate.get('mean_inorganic_scaffold_cartesian_rms_angstrom', math.nan):.3g} A"
        ),
        "",
        "Conclusions:",
        "",
    ]
    lines.extend(f"- {item}" for item in aggregate.get("observations", []))
    lines.append("")
    return lines


def _write_normalized_inorganic_cif(
    source_path: Path,
    output_path: Path,
    inorganic_elements: set[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source = Structure.from_file(str(source_path))
        species = []
        frac_coords = []
        for site in source:
            symbol = str(site.specie.symbol)
            if symbol in inorganic_elements:
                species.append(symbol)
                frac_coords.append(site.frac_coords)
        if species:
            filtered = Structure(source.lattice, species, frac_coords)
            output_path.write_text(filtered.to(fmt="cif"), encoding="utf-8")
            return
    except Exception:
        pass
    _write_simple_normalized_inorganic_cif(
        source_path,
        output_path,
        inorganic_elements,
    )


def _write_simple_normalized_inorganic_cif(
    source_path: Path,
    output_path: Path,
    inorganic_elements: set[str],
) -> None:
    cell: dict[str, float] = {}
    rows: list[tuple[str, str, float, float, float]] = []
    in_atom_rows = False
    for line in source_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] in {
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        }:
            try:
                cell[parts[0]] = _parse_cif_float(parts[1])
            except (IndexError, ValueError):
                pass
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            not in_atom_rows
            or stripped.startswith(("_", "#", "loop_"))
            or len(parts) < 5
            or parts[1] not in inorganic_elements
        ):
            continue
        try:
            coords = tuple(
                _parse_cif_float(parts[index]) for index in (2, 3, 4)
            )
        except ValueError:
            continue
        rows.append((str(parts[0]), str(parts[1]), *coords))
    output_path.write_text(
        _normalized_cif_text(source_path.stem, cell, rows),
        encoding="utf-8",
    )


def _normalized_cif_text(
    name: str,
    cell: dict[str, float],
    rows: list[tuple[str, str, float, float, float]],
) -> str:
    composition: dict[str, int] = {}
    for _, element, *_ in rows:
        composition[element] = composition.get(element, 0) + 1
    lines = [
        f"data_{_formula_token(name) or 'scaffold'}",
        "_symmetry_space_group_name_H-M 'P1'",
        "_space_group_IT_number 1",
        f"_chemical_formula_sum '{_formula_sum(composition)}'",
        f"_cell_length_a {cell.get('_cell_length_a', 1.0):.6f}",
        f"_cell_length_b {cell.get('_cell_length_b', 1.0):.6f}",
        f"_cell_length_c {cell.get('_cell_length_c', 1.0):.6f}",
        f"_cell_angle_alpha {cell.get('_cell_angle_alpha', 90.0):.6f}",
        f"_cell_angle_beta {cell.get('_cell_angle_beta', 90.0):.6f}",
        f"_cell_angle_gamma {cell.get('_cell_angle_gamma', 90.0):.6f}",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for label, element, x_frac, y_frac, z_frac in rows:
        lines.append(
            f"{label} {element} {x_frac:.6f} {y_frac:.6f} " f"{z_frac:.6f} 1.0"
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_cif_float(value: str) -> float:
    return float(str(value).strip().split("(")[0].strip("'\""))


def _formula_sum(composition: dict[str, int]) -> str:
    return (
        " ".join(
            f"{element}{composition[element]}"
            for element in sorted(composition)
            if composition[element] > 0
        )
        or "X1"
    )


def _truth_hkl(truth: dict[str, Any] | None) -> tuple[int, int, int]:
    truth = truth or {}
    return (
        int(truth.get("h", 0)),
        int(truth.get("k", 0)),
        int(truth.get("l", 0)),
    )


def _parse_hkl_triplet(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        pieces = value.strip().strip("()").replace(",", " ").split()
    else:
        pieces = list(value)
    if len(pieces) != 3:
        return None
    try:
        return tuple(int(float(piece)) for piece in pieces)
    except (TypeError, ValueError):
        return None


def _plane_family_key(hkl: Iterable[int]) -> str:
    values = [abs(int(value)) for value in hkl]
    divisor = 0
    for value in values:
        divisor = math.gcd(divisor, value)
    if divisor > 1:
        values = [value // divisor for value in values]
    values = sorted(values, reverse=True)
    return "(" + " ".join(str(value) for value in values) + ")"


def _inorganic_formula_from_hypothesis(
    hypothesis: str,
    inorganic_atoms: Iterable[str],
    organic_molecules: Iterable[str],
) -> str:
    text = "".join(str(hypothesis).split())
    for label in sorted(
        {str(item).strip() for item in organic_molecules if str(item).strip()},
        key=len,
        reverse=True,
    ):
        text = _remove_formula_label(text, label)
    elements = [
        _element_symbol(atom)
        for atom in inorganic_atoms
        if _element_symbol(atom)
    ]
    counts = {element: 0.0 for element in elements}
    index = 0
    ordered = sorted(elements, key=len, reverse=True)
    while index < len(text):
        matched = ""
        for element in ordered:
            if text.startswith(element, index):
                matched = element
                break
        if not matched:
            index += 1
            continue
        index += len(matched)
        number, index = _parse_formula_count(text, index)
        counts[matched] = counts.get(matched, 0.0) + number
    pieces = []
    for element in elements:
        count = counts.get(element, 0.0)
        if count <= 0.0:
            continue
        pieces.append(f"{element}{_format_formula_count(count)}")
    return "".join(pieces)


def _organic_molecule_counts_from_hypothesis(
    hypothesis: str,
    organic_molecules: Iterable[str],
) -> dict[str, float]:
    text = "".join(str(hypothesis).split())
    counts: dict[str, float] = {}
    for raw_label in sorted(
        {str(item).strip() for item in organic_molecules if str(item).strip()},
        key=len,
        reverse=True,
    ):
        label = _canonical_molecule_label(raw_label)
        total, text = _pop_formula_label_count(text, label)
        if total > 0.0:
            counts[label] = counts.get(label, 0.0) + total
    return counts


def _pop_formula_label_count(formula: str, label: str) -> tuple[float, str]:
    result = formula
    total = 0.0
    for token in (f"({label})", label):
        while True:
            index = result.find(token)
            if index < 0:
                break
            count, end = _parse_formula_count(result, index + len(token))
            total += count
            result = result[:index] + result[end:]
    return total, result


def _parse_element_formula_counts(formula: str) -> dict[str, float]:
    text = "".join(str(formula).split())
    counts: dict[str, float] = {}
    index = 0
    while index < len(text):
        char = text[index]
        if not char.isupper():
            index += 1
            continue
        end = index + 1
        if end < len(text) and text[end].islower():
            end += 1
        element = _element_symbol(text[index:end])
        count, index = _parse_formula_count(text, end)
        counts[element] = counts.get(element, 0.0) + count
    return counts


def _reference_molecule_metadata(label: str) -> dict[str, Any]:
    canonical = _canonical_molecule_label(label)
    return dict(REFERENCE_MOLECULES.get(canonical, {}))


def _canonical_molecule_label(label: str) -> str:
    token = str(label).strip()
    return token.upper() if token else token


def _organic_molecule_nominal_charge(label: str) -> float:
    return float(
        _ORGANIC_MOLECULE_NOMINAL_CHARGES.get(
            _canonical_molecule_label(label),
            0.0,
        )
    )


def _formula_from_counts(counts: dict[str, float]) -> str:
    pieces = []
    for element in sorted(
        counts,
        key=lambda item: (
            _ATOMIC_NUMBERS.get(_element_symbol(item), 999),
            item,
        ),
    ):
        symbol = _element_symbol(element)
        count = float(counts[element])
        if count <= 0.0:
            continue
        pieces.append(f"{symbol}{_format_formula_count(count)}")
    return "".join(pieces)


def _unique_element_sequence(elements: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in elements:
        element = _element_symbol(str(raw))
        if not element or element in seen:
            continue
        ordered.append(element)
        seen.add(element)
    return tuple(ordered)


def _remove_formula_label(formula: str, label: str) -> str:
    result = formula
    while True:
        changed = False
        parenthesized = f"({label})"
        index = result.find(parenthesized)
        if index >= 0:
            _, end = _parse_formula_count(result, index + len(parenthesized))
            result = result[:index] + result[end:]
            changed = True
        index = result.find(label)
        if index >= 0:
            _, end = _parse_formula_count(result, index + len(label))
            result = result[:index] + result[end:]
            changed = True
        if not changed:
            return result


def _parse_formula_count(formula: str, index: int) -> tuple[float, int]:
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


def _format_formula_count(count: float) -> str:
    if abs(count - round(count)) <= 1.0e-9:
        integer = int(round(count))
        return "" if integer == 1 else str(integer)
    return f"{count:.4g}"


def _formula_token(formula: str) -> str:
    return "".join(char.lower() for char in str(formula) if char.isalnum())


def _element_symbol(symbol: str) -> str:
    letters = "".join(char for char in str(symbol) if char.isalpha())
    if not letters:
        return str(symbol)
    if len(letters) == 1:
        return letters.upper()
    return f"{letters[0].upper()}{letters[1:].lower()}"


__all__ = [
    "SyntheticRefinementConfig",
    "SyntheticRefinementResult",
    "assess_candidate_assignments_against_truth",
    "assess_peak_families_against_truth",
    "electron_proxy_element_for_count",
    "formula_electron_count",
    "generate_organic_electron_proxy_cifs",
    "generate_scaffold_candidate_cifs",
    "match_detected_peaks_to_truth",
    "organic_electron_proxy_plan",
    "rank_inorganic_scaffolds",
    "run_synthetic_refinement_pipeline",
    "validate_inorganic_scaffold",
]
