"""Headless experimental GIWAXS structure-refinement workflow.

The synthetic benchmark harness creates controlled measurements.  This module
starts from real detector data plus calibration assets, maps the image into
q-space, detects peak families, generates chemically plausible inorganic
scaffold hypotheses, and then runs the same staged scaffold/molecule refinement
used by the benchmark.  Reference CIFs are accepted only as validation targets;
their cell constants, coordinates, and stoichiometry are not fed to the solver.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from ewald.analysis.structure import (
    DEFAULT_PHASE_TAG,
    LatticeCandidate,
    generate_ranked_cif_records,
    group_peak_families,
    guess_lattice_candidates,
)
from ewald.benchmark.structure_benchmark import (
    BenchmarkRunConfig,
    BenchmarkStructureSpec,
    _append_fileset_logbook,
    _append_logbook,
    _attach_refined_cif_records,
    _candidate_search_config,
    _detect_structure_peaks,
    _json_safe,
    _materialize_generated_cifs,
    _molecule_records,
    _rank_generated_cifs,
    _save_peak_detection_plot,
    _slug,
    _solve_constraints,
    _timestamp,
    _validate_best_generated_structure,
    _write_json,
)
from ewald.data.models import (
    ImageCorrectionState,
    ProjectState,
    CorrectionAssetRef,
)
from ewald.io.project import save_project
from ewald.processing.qspace import (
    GrazingIncidenceConfig,
    map_grazing_incidence_qspace,
    xray_energy_kev_from_wavelength_m,
)
from ewald.simulation.giwaxs import GIWAXSSimulationParameters

DEFAULT_EXPERIMENTAL_OUTPUT_DIR = Path(
    "example/projects/experimental_refinement"
)
ATOMIC_MASSES_AMU = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Br": 79.904,
    "Rb": 85.468,
    "Sn": 118.710,
    "I": 126.904,
    "Cs": 132.905,
    "Pb": 207.2,
    "Bi": 208.980,
}
PEROVSKITE_B_SITE_CATIONS = {"Pb", "Sn", "Ge"}
PEROVSKITE_HALIDES = {"I", "Br", "Cl", "F"}
PEROVSKITE_INORGANIC_A_SITE_CATIONS = {"Cs", "Rb", "K", "Na"}
PEROVSKITE_ORGANIC_A_SITE_CATIONS = {"MA", "FA", "BA"}
PEROVSKITE_NEUTRAL_SOLVENTS = {"DMF", "DMSO", "NMP"}
PEROVSKITE_MOTIF_LIBRARY: tuple[dict[str, Any], ...] = (
    {
        "motif_id": "3d_corner_abx3",
        "label": "3D corner-sharing perovskite",
        "dimensionality": "3D",
        "connectivity": "corner_sharing",
        "b_count": 1,
        "x_count": 3,
        "motif_prior_penalty": 0.00,
    },
    {
        "motif_id": "2d_corner_layer_n1",
        "label": "2D single-layer corner-sharing perovskite",
        "dimensionality": "2D",
        "connectivity": "corner_sharing",
        "b_count": 1,
        "x_count": 4,
        "motif_prior_penalty": 0.02,
    },
    {
        "motif_id": "2d_corner_layer_n2",
        "label": "2D double-layer corner-sharing perovskite",
        "dimensionality": "2D",
        "connectivity": "corner_sharing",
        "b_count": 2,
        "x_count": 7,
        "motif_prior_penalty": 0.03,
    },
    {
        "motif_id": "2d_corner_layer_n3",
        "label": "2D triple-layer corner-sharing perovskite",
        "dimensionality": "2D",
        "connectivity": "corner_sharing",
        "b_count": 3,
        "x_count": 10,
        "motif_prior_penalty": 0.04,
    },
    {
        "motif_id": "1d_corner_chain",
        "label": "1D corner-sharing octahedral chain",
        "dimensionality": "1D",
        "connectivity": "corner_sharing",
        "b_count": 1,
        "x_count": 5,
        "motif_prior_penalty": 0.05,
    },
    {
        "motif_id": "1d_edge_chain",
        "label": "1D edge-sharing octahedral chain",
        "dimensionality": "1D",
        "connectivity": "edge_sharing",
        "b_count": 1,
        "x_count": 4,
        "motif_prior_penalty": 0.04,
    },
    {
        "motif_id": "1d_face_chain",
        "label": "1D face-sharing octahedral chain",
        "dimensionality": "1D",
        "connectivity": "face_sharing",
        "b_count": 1,
        "x_count": 3,
        "motif_prior_penalty": 0.05,
    },
    {
        "motif_id": "0d_isolated_octahedron",
        "label": "0D isolated BX6 octahedron",
        "dimensionality": "0D",
        "connectivity": "isolated_octahedra",
        "b_count": 1,
        "x_count": 6,
        "motif_prior_penalty": 0.07,
    },
    {
        "motif_id": "0d_face_shared_bioctahedron",
        "label": "0D face-sharing bioctahedron",
        "dimensionality": "0D",
        "connectivity": "face_sharing",
        "b_count": 2,
        "x_count": 9,
        "motif_prior_penalty": 0.08,
    },
    {
        "motif_id": "0d_edge_shared_dimer",
        "label": "0D edge-sharing octahedral dimer",
        "dimensionality": "0D",
        "connectivity": "edge_sharing",
        "b_count": 2,
        "x_count": 10,
        "motif_prior_penalty": 0.09,
    },
    {
        "motif_id": "2d_binary_halide_sheet",
        "label": "Pb/Sn halide sheet or solvate precursor",
        "dimensionality": "2D",
        "connectivity": "edge_sharing_or_sheet",
        "b_count": 1,
        "x_count": 2,
        "motif_prior_penalty": 0.06,
    },
    {
        "motif_id": "mixed_condensed_m3x8_solvate",
        "label": "Condensed perovskite-solvate derivative M3X8",
        "dimensionality": "1D/2D",
        "connectivity": "mixed_corner_edge_face",
        "b_count": 3,
        "x_count": 8,
        "motif_prior_penalty": 0.01,
        "prefer_with_neutral_solvent": True,
    },
)


@dataclass(slots=True)
class ExperimentalRefinementConfig:
    """Controls for solving a real detector image with validation CIF."""

    output_dir: Path = DEFAULT_EXPERIMENTAL_OUTPUT_DIR
    qspace_shape: tuple[int, int] = (180, 220)
    qspace_ip_range: tuple[float, float] | None = None
    qspace_oop_range: tuple[float, float] | None = None
    xray_energy_kev: float | None = None
    incident_angle_deg: float = 0.3
    tilt_angle_deg: float = 0.0
    sample_orientation: int = 4
    correct_solid_angle: bool = True
    polarization_factor: float | None = 0.95
    normalization_factor: float = 1.0
    peak_threshold_percentile: float = 99.72
    peak_adaptive_threshold: bool = True
    peak_adaptive_floor_percentile: float = 95.0
    peak_min_snr: float = 3.0
    peak_max_peaks: int = 120
    peak_min_distance_px: int = 4
    peak_deduplicate_tolerance: float = 0.035
    family_tolerance: float = 0.04
    family_ratio_tolerance: float = 0.06
    candidate_hkl_max: int = 7
    candidate_q_tolerance: float = 0.075
    candidate_relative_tolerance: float = 0.045
    candidate_lattice_min: float = 2.5
    candidate_lattice_max: float = 60.0
    candidate_grid_points: int = 14
    candidate_max_candidates: int = 12
    candidate_axis_scale_variants: tuple[float, ...] = (1.0, 0.5)
    hkl_extent: int = 7
    records_per_stoichiometry: int = 1
    max_stoichiometry_hypotheses: int = 12
    max_generated_cifs_to_compare: int = 12
    comparison_theta_x_offsets: tuple[float, ...] = (
        -10.0,
        -4.0,
        0.0,
        4.0,
        10.0,
    )
    comparison_theta_y_values: tuple[float, ...] = (
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
    )
    comparison_plot_count: int = 4
    staged_refinement: bool = True
    refinement_coarse_detector_shape: tuple[int, int] = (72, 96)
    refinement_coarse_hkl_extent: int = 4
    refinement_fractional_step: float = 0.06
    refinement_scaffold_variant_count: int = 0
    bragg_intensity_weight: float = 0.35
    bragg_intensity_tolerance: float = 0.08
    bragg_intensity_max_peaks: int = 80
    assume_unit_cell_symmetry: bool = True
    sigma_theta: float = 0.035
    sigma_phi: float = 0.35
    sigma_r: float = 0.030
    fiber_tilt_center_deg: float = 90.0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.qspace_shape = (
            int(self.qspace_shape[0]),
            int(self.qspace_shape[1]),
        )
        if self.qspace_ip_range is not None:
            self.qspace_ip_range = (
                float(self.qspace_ip_range[0]),
                float(self.qspace_ip_range[1]),
            )
        if self.qspace_oop_range is not None:
            self.qspace_oop_range = (
                float(self.qspace_oop_range[0]),
                float(self.qspace_oop_range[1]),
            )
        self.comparison_theta_x_offsets = tuple(
            float(value) for value in self.comparison_theta_x_offsets
        )
        self.comparison_theta_y_values = tuple(
            float(value) for value in self.comparison_theta_y_values
        )
        self.candidate_axis_scale_variants = tuple(
            float(value)
            for value in self.candidate_axis_scale_variants
            if float(value) > 0.0
        ) or (1.0,)
        self.refinement_coarse_detector_shape = (
            int(self.refinement_coarse_detector_shape[0]),
            int(self.refinement_coarse_detector_shape[1]),
        )
        self.refinement_scaffold_variant_count = max(
            0,
            int(self.refinement_scaffold_variant_count),
        )
        self.bragg_intensity_weight = max(
            0.0, float(self.bragg_intensity_weight)
        )
        self.bragg_intensity_tolerance = max(
            1.0e-9, float(self.bragg_intensity_tolerance)
        )
        self.bragg_intensity_max_peaks = max(
            1, int(self.bragg_intensity_max_peaks)
        )
        self.assume_unit_cell_symmetry = bool(self.assume_unit_cell_symmetry)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["qspace_shape"] = list(self.qspace_shape)
        payload["qspace_ip_range"] = (
            list(self.qspace_ip_range)
            if self.qspace_ip_range is not None
            else None
        )
        payload["qspace_oop_range"] = (
            list(self.qspace_oop_range)
            if self.qspace_oop_range is not None
            else None
        )
        payload["comparison_theta_x_offsets"] = list(
            self.comparison_theta_x_offsets
        )
        payload["comparison_theta_y_values"] = list(
            self.comparison_theta_y_values
        )
        payload["candidate_axis_scale_variants"] = list(
            self.candidate_axis_scale_variants
        )
        payload["refinement_coarse_detector_shape"] = list(
            self.refinement_coarse_detector_shape
        )
        return payload


@dataclass(slots=True)
class ExperimentalRefinementResult:
    """Summary for one experimental refinement run."""

    run_id: str
    output_dir: Path
    fileset: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "fileset": self.fileset,
        }


def run_experimental_refinement(
    *,
    reference_cif: str | Path,
    detector_image: str | Path,
    poni_file: str | Path,
    mask_file: str | Path | None,
    inorganic_atoms: Iterable[str],
    organic_molecules: Iterable[str] = (),
    config: ExperimentalRefinementConfig | None = None,
    label: str | None = None,
) -> ExperimentalRefinementResult:
    """Solve one real detector image and validate against a known CIF."""

    cfg = config or ExperimentalRefinementConfig()
    reference_path = Path(reference_cif)
    image_path = Path(detector_image)
    poni_path = Path(poni_file)
    mask_path = Path(mask_file) if mask_file else None
    spec = BenchmarkStructureSpec(
        cif_path=reference_path,
        inorganic_atoms=tuple(inorganic_atoms),
        organic_molecules=tuple(organic_molecules),
        label=label or reference_path.stem,
    )
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_root = cfg.output_dir / run_id
    fileset_id = _slug(label or reference_path.stem)
    fileset_dir = run_root / fileset_id
    qspace_dir = fileset_dir / "qspace"
    plots_dir = fileset_dir / "plots"
    rankings_dir = fileset_dir / "rankings"
    generated_dir = fileset_dir / "generated_structures"
    best_dir = fileset_dir / "best_fit_generated_structures"
    for directory in (
        qspace_dir,
        plots_dir,
        rankings_dir,
        generated_dir,
        best_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    logbook = run_root / "LOGBOOK.md"
    _append_experimental_header(
        logbook,
        run_id=run_id,
        spec=spec,
        detector_image=image_path,
        poni_file=poni_path,
        mask_file=mask_path,
        cfg=cfg,
    )
    raw_image = _load_detector_image(image_path)
    mask = _load_mask_file(mask_path) if mask_path is not None else None
    qspace = _map_detector_to_qspace(
        raw_image,
        poni_path,
        mask,
        cfg,
    )
    target = _standardize_qspace(qspace)
    qspace_path = qspace_dir / f"{fileset_id}_mapped_qspace.npz"
    _write_experiment_qspace_npz(
        qspace_path,
        target,
        spec,
        image_path,
        poni_path,
        mask_path,
        cfg,
    )

    bench_cfg = _benchmark_config_for_experiment(cfg, target)
    target_params = _simulation_params_for_target(target, bench_cfg)
    project = _experimental_project(
        spec,
        image_path,
        poni_path,
        mask_path,
        qspace_path,
        cfg,
        target,
    )
    data_file = project.data_files[0]

    peaks = _detect_structure_peaks(target, bench_cfg)
    families = group_peak_families(
        peaks,
        tolerance=cfg.family_tolerance,
        ratio_tolerance=cfg.family_ratio_tolerance,
        phase_tag=DEFAULT_PHASE_TAG,
    )
    peak_plot = _save_peak_detection_plot(
        target,
        peaks,
        plots_dir / "peak_detection.png",
        title=fileset_id,
    )
    _store_experimental_peak_records(
        project,
        str(data_file.data_id),
        peaks,
        families,
    )
    _write_json(rankings_dir / "peak_families.json", families)

    raw_candidates = guess_lattice_candidates(
        peaks,
        _candidate_search_config(bench_cfg),
    )
    candidates = _expand_axis_scale_candidates(raw_candidates, cfg)
    candidate_records = [candidate.as_dict() for candidate in candidates]
    _write_json(rankings_dir / "lattice_candidates.json", candidate_records)

    stoichiometry_hypotheses = chemistry_stoichiometry_hypotheses(
        spec.inorganic_atoms,
        spec.organic_molecules,
        limit=cfg.max_stoichiometry_hypotheses,
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
                "molecule labels; reference CIF stoichiometry not read"
            ),
            "hypotheses": stoichiometry_hypotheses,
            "perovskite_motif_hypotheses": perovskite_motifs,
        },
    )
    _write_json(
        rankings_dir / "perovskite_motif_hypotheses.json",
        perovskite_motifs,
    )
    generated_records = _generate_chemistry_candidate_cifs(
        spec,
        candidates,
        stoichiometry_hypotheses,
        cfg,
    )
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

    comparisons = _rank_generated_cifs(
        target,
        generated_records,
        target_params,
        bench_cfg,
        rankings_dir,
        plots_dir,
        peak_intensity_peaks=peaks,
    )
    _attach_refined_cif_records(project, comparisons)
    validation = _validate_best_generated_structure(
        comparisons,
        reference_path,
        best_dir,
        run_root / "best_fit_generated_structures" / fileset_id,
    )
    project.analysis_results.setdefault("experimental_refinement", {})[
        fileset_id
    ] = {
        "fileset_id": fileset_id,
        "constraints": _solve_constraints(spec),
        "peak_count": len(peaks),
        "peak_families": families,
        "stoichiometry_hypotheses": stoichiometry_hypotheses,
        "perovskite_motif_hypotheses": perovskite_motifs,
        "lattice_candidates": candidate_records,
        "generated_cif_rankings": comparisons,
        "validation": validation,
    }
    project_path = save_project(project, fileset_dir / f"{fileset_id}.ewld")
    fileset_summary = {
        "fileset_id": fileset_id,
        "structure": spec.as_dict(),
        "detector_image": str(image_path),
        "poni_file": str(poni_path),
        "mask_file": str(mask_path) if mask_path else None,
        "mapped_qspace": str(qspace_path),
        "peak_detection_plot": str(peak_plot),
        "project": str(project_path),
        "readable_project": str(project_path.with_suffix(".ewald.json")),
        "peak_count": len(peaks),
        "family_count": len(families),
        "top_candidate": candidate_records[0] if candidate_records else None,
        "top_perovskite_motif": (
            perovskite_motifs[0] if perovskite_motifs else None
        ),
        "best_generated_cif": validation.get("best_generated_cif"),
        "validation": validation,
    }
    _write_json(rankings_dir / "validation.json", validation)
    _write_json(
        run_root / "summary.json",
        {"run_id": run_id, "fileset": fileset_summary},
    )
    _append_experimental_logbook(
        logbook,
        fileset_summary,
        target_params,
        peaks,
        families,
        candidate_records,
        stoichiometry_hypotheses,
        comparisons,
    )
    return ExperimentalRefinementResult(
        run_id=run_id,
        output_dir=run_root,
        fileset=fileset_summary,
    )


def chemistry_stoichiometry_hypotheses(
    inorganic_atoms: Iterable[str],
    organic_molecules: Iterable[str] = (),
    *,
    limit: int = 12,
) -> list[str]:
    """Return broad perovskite-derivative stoichiometry hypotheses."""

    motif_records = perovskite_scaffold_hypotheses(
        inorganic_atoms,
        organic_molecules,
        limit=max(1, int(limit)) * 3,
    )
    motif_formulas = [str(record["formula"]) for record in motif_records]
    atoms = [
        _element_symbol(atom) for atom in inorganic_atoms if str(atom).strip()
    ]
    molecules = [
        _molecule_token(label)
        for label in organic_molecules
        if _molecule_token(label) and _molecule_token(label).lower() != "none"
    ]
    b_sites = [atom for atom in atoms if atom in {"Pb", "Sn", "Ge"}]
    halides = [atom for atom in atoms if atom in {"I", "Br", "Cl", "F"}]
    inorganic_first = [
        atom for atom in atoms if atom not in set(b_sites) | set(halides)
    ]
    b_site = b_sites[0] if b_sites else (atoms[0] if atoms else "X")
    halide = halides[0] if halides else (atoms[1] if len(atoms) > 1 else "X")
    molecule_patterns = _molecule_count_patterns(molecules)
    donor_cations = {"MA", "FA", "BA"}
    has_organic_cation = any(label in donor_cations for label in molecules)
    inorganic_patterns = (
        [
            (3, 8),
            (1, 3),
            (1, 2),
            (1, 4),
            (2, 5),
            (2, 6),
            (3, 9),
            (4, 10),
            (4, 12),
        ]
        if has_organic_cation
        else [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (3, 8),
            (3, 9),
            (4, 10),
            (4, 12),
        ]
    )
    priority_legacy: list[str] = []
    if molecule_patterns:
        for b_count, x_count in ((1, 2), (1, 3)):
            prefix = "".join(
                _formula_piece(label, count)
                for label, count in molecule_patterns[0]
            )
            core = "".join(_formula_piece(atom, 1) for atom in inorganic_first)
            core += _formula_piece(b_site, b_count)
            core += _formula_piece(halide, x_count)
            priority_legacy.append(prefix + core)
    hypotheses: list[str] = (
        motif_formulas[:2]
        + list(dict.fromkeys(priority_legacy))
        + motif_formulas[2:]
    )
    z_multipliers = (1, 2)
    for molecule_counts in molecule_patterns:
        for b_count, x_count in inorganic_patterns:
            for z_multiplier in z_multipliers:
                prefix = "".join(
                    _formula_piece(label, count * z_multiplier)
                    for label, count in molecule_counts
                )
                core = "".join(
                    _formula_piece(atom, z_multiplier)
                    for atom in inorganic_first
                )
                core += _formula_piece(b_site, b_count * z_multiplier)
                core += _formula_piece(halide, x_count * z_multiplier)
                hypotheses.append(prefix + core)
    hypotheses.append("".join(_formula_piece(atom, 1) for atom in atoms))
    deduped = list(dict.fromkeys(item for item in hypotheses if item))
    return deduped[: max(1, int(limit))]


def perovskite_scaffold_hypotheses(
    inorganic_atoms: Iterable[str],
    organic_molecules: Iterable[str] = (),
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Return named perovskite/perovskitoid scaffold hypotheses.

    The motifs are formulated for divalent Pb/Sn/Ge centers and monovalent
    halides.  Organic and inorganic A-site cations are used for charge balance
    when available, while neutral coordinating solvents are retained as
    solvate/structure-directing species without contributing charge.
    """

    atoms = [
        _element_symbol(atom) for atom in inorganic_atoms if str(atom).strip()
    ]
    molecules = [
        _molecule_token(label)
        for label in organic_molecules
        if _molecule_token(label) and _molecule_token(label).lower() != "none"
    ]
    b_sites = [atom for atom in atoms if atom in PEROVSKITE_B_SITE_CATIONS]
    halides = [atom for atom in atoms if atom in PEROVSKITE_HALIDES]
    if not b_sites or not halides:
        return []
    b_site = b_sites[0]
    halide = halides[0]
    inorganic_a_sites = [
        atom for atom in atoms if atom in PEROVSKITE_INORGANIC_A_SITE_CATIONS
    ]
    organic_a_sites = [
        label
        for label in molecules
        if label in PEROVSKITE_ORGANIC_A_SITE_CATIONS
    ]
    neutral_solvents = [
        label
        for label in molecules
        if label in PEROVSKITE_NEUTRAL_SOLVENTS
        or label not in PEROVSKITE_ORGANIC_A_SITE_CATIONS
    ]
    a_site_label = (
        organic_a_sites[0]
        if organic_a_sites
        else (inorganic_a_sites[0] if inorganic_a_sites else "")
    )
    has_solvent = bool(neutral_solvents)
    ordered_motifs = _ordered_perovskite_motifs(
        has_solvent,
        has_a_site_cation=bool(a_site_label),
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for motif in ordered_motifs:
        for z_multiplier in (1, 2):
            record = _perovskite_motif_formula_record(
                motif,
                b_site=b_site,
                halide=halide,
                a_site_label=a_site_label,
                neutral_solvents=neutral_solvents,
                z_multiplier=z_multiplier,
            )
            if not record:
                continue
            formula = str(record["formula"])
            if formula in seen:
                continue
            seen.add(formula)
            records.append(record)
            if len(records) >= max(1, int(limit)):
                return records
    return records


def _ordered_perovskite_motifs(
    has_neutral_solvent: bool,
    *,
    has_a_site_cation: bool,
) -> list[dict[str, Any]]:
    motifs = [dict(item) for item in PEROVSKITE_MOTIF_LIBRARY]
    if has_neutral_solvent and not has_a_site_cation:
        priority = {
            "2d_binary_halide_sheet": 0,
            "3d_corner_abx3": 1,
            "mixed_condensed_m3x8_solvate": 2,
        }
    elif has_neutral_solvent:
        priority = {
            "mixed_condensed_m3x8_solvate": 0,
            "3d_corner_abx3": 1,
            "2d_binary_halide_sheet": 2,
        }
    else:
        priority = {
            "3d_corner_abx3": 0,
            "2d_corner_layer_n1": 1,
            "1d_edge_chain": 2,
        }
    motifs.sort(
        key=lambda item: (
            priority.get(str(item["motif_id"]), 10),
            float(item.get("motif_prior_penalty", 0.0)),
            str(item["motif_id"]),
        )
    )
    return motifs


def _perovskite_motif_formula_record(
    motif: dict[str, Any],
    *,
    b_site: str,
    halide: str,
    a_site_label: str,
    neutral_solvents: list[str],
    z_multiplier: int,
) -> dict[str, Any] | None:
    z = max(1, int(z_multiplier))
    b_count = int(motif["b_count"]) * z
    x_count = int(motif["x_count"]) * z
    required_cation_count = max(0, x_count - 2 * b_count)
    prefix_pieces: list[str] = []
    cation_count = 0
    if a_site_label and required_cation_count > 0:
        cation_count = required_cation_count
        prefix_pieces.append(_formula_piece(a_site_label, cation_count))
    solvent_count = _neutral_solvent_count_for_motif(
        neutral_solvents,
        required_cation_count=required_cation_count,
        z_multiplier=z,
        motif=motif,
    )
    for solvent in neutral_solvents[:1]:
        if solvent_count > 0:
            prefix_pieces.append(_formula_piece(solvent, solvent_count))
    inorganic_formula = _formula_piece(b_site, b_count) + _formula_piece(
        halide, x_count
    )
    formula = "".join(prefix_pieces) + inorganic_formula
    charge_balance = _formula_charge_balance(formula)
    return {
        "formula": formula,
        "inorganic_formula": inorganic_formula,
        "b_site": b_site,
        "halide": halide,
        "b_count": b_count,
        "x_count": x_count,
        "x_to_b_ratio": x_count / max(b_count, 1),
        "a_site_label": a_site_label or None,
        "a_site_count": cation_count,
        "neutral_solvent_labels": neutral_solvents[:1],
        "neutral_solvent_count": solvent_count,
        "required_monovalent_cation_count": required_cation_count,
        "z_multiplier": z,
        "charge_balance": charge_balance,
        **{
            key: motif[key]
            for key in (
                "motif_id",
                "label",
                "dimensionality",
                "connectivity",
                "motif_prior_penalty",
            )
            if key in motif
        },
    }


def _neutral_solvent_count_for_motif(
    neutral_solvents: list[str],
    *,
    required_cation_count: int,
    z_multiplier: int,
    motif: dict[str, Any],
) -> int:
    if not neutral_solvents:
        return 0
    motif_id = str(motif.get("motif_id") or "")
    if motif.get("prefer_with_neutral_solvent"):
        return max(z_multiplier, required_cation_count)
    if required_cation_count > 0:
        return min(max(required_cation_count, z_multiplier), 4 * z_multiplier)
    if motif_id == "2d_binary_halide_sheet":
        return z_multiplier
    return max(1, z_multiplier)


def _perovskite_motif_metadata_by_formula(
    inorganic_atoms: Iterable[str],
    organic_molecules: Iterable[str],
    *,
    limit: int,
) -> dict[str, dict[str, Any]]:
    return {
        str(record["formula"]): dict(record)
        for record in perovskite_scaffold_hypotheses(
            inorganic_atoms,
            organic_molecules,
            limit=limit,
        )
    }


def _expand_axis_scale_candidates(
    candidates: list[LatticeCandidate],
    cfg: ExperimentalRefinementConfig,
) -> list[LatticeCandidate]:
    """Add subcell/supercell variants for incomplete peak maps."""

    scales = tuple(dict.fromkeys(cfg.candidate_axis_scale_variants))
    expanded: list[LatticeCandidate] = []
    seen: set[tuple[str, int, int, int]] = set()
    for candidate in candidates:
        for scale_a, scale_b, scale_c in product(scales, repeat=3):
            if not expanded and (scale_a, scale_b, scale_c) != (1.0, 1.0, 1.0):
                continue
            a = float(candidate.a) * scale_a
            b = float(candidate.b) * scale_b
            c = float(candidate.c) * scale_c
            if min(a, b, c) < cfg.candidate_lattice_min:
                continue
            if max(a, b, c) > cfg.candidate_lattice_max:
                continue
            key = (
                candidate.crystal_system,
                int(round(a * 1000)),
                int(round(b * 1000)),
                int(round(c * 1000)),
            )
            if key in seen:
                continue
            seen.add(key)
            changed_axes = sum(
                1
                for value in (scale_a, scale_b, scale_c)
                if abs(value - 1.0) > 1.0e-9
            )
            suffix = (
                ""
                if changed_axes == 0
                else "_axis_scale_"
                + "_".join(
                    _scale_token(value)
                    for value in (scale_a, scale_b, scale_c)
                )
            )
            expanded.append(
                LatticeCandidate(
                    candidate_id=f"{candidate.candidate_id}{suffix}",
                    crystal_system=candidate.crystal_system,
                    a=a,
                    b=b,
                    c=c,
                    alpha=candidate.alpha,
                    beta=candidate.beta,
                    gamma=candidate.gamma,
                    score=float(candidate.score) + 0.001 * changed_axes,
                    rms_error=candidate.rms_error,
                    matched_count=candidate.matched_count,
                    outlier_count=candidate.outlier_count,
                    method=f"{candidate.method}+axis_scale",
                    assignments=list(candidate.assignments),
                    notes=(
                        f"{candidate.notes}; "
                        f"axis scales {scale_a:g},{scale_b:g},{scale_c:g}"
                    ).strip("; "),
                    orientation_quaternion=candidate.orientation_quaternion,
                    projection_mode=candidate.projection_mode,
                )
            )
    expanded.sort(
        key=lambda item: (item.score, item.outlier_count, -item.matched_count)
    )
    scale_variant_budget = max(1, len(scales) ** 3)
    return expanded[
        : max(
            cfg.candidate_max_candidates * scale_variant_budget,
            cfg.candidate_max_candidates,
        )
    ]


def _generate_chemistry_candidate_cifs(
    spec: BenchmarkStructureSpec,
    candidates: list[Any],
    stoichiometry_hypotheses: list[str],
    cfg: ExperimentalRefinementConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    molecules = _molecule_records(spec.organic_molecules)
    if not candidates:
        return []
    per_hypothesis = max(1, cfg.records_per_stoichiometry)
    reduced_records_by_candidate: dict[str, dict[str, Any]] = {}
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
            for record in generate_ranked_cif_records(
                candidate,
                atoms=spec.inorganic_atoms,
                molecules=molecules,
                stoichiometry=hypothesis,
                limit=per_hypothesis,
                allow_explicit_templates=False,
                assume_unit_cell_symmetry=cfg.assume_unit_cell_symmetry,
            ):
                payload = dict(record)
                payload["cif_id"] = _slug(
                    f"{payload.get('cif_id')}_{_formula_id(hypothesis)}"
                )
                payload["solve_stoichiometry_hypothesis"] = hypothesis
                payload["solve_stoichiometry_hypothesis_rank"] = (
                    hypothesis_rank
                )
                payload["template_coordinates_allowed"] = False
                charge_balance = _formula_charge_balance(hypothesis)
                payload["charge_balance"] = charge_balance
                payload["charge_balance_penalty"] = float(
                    charge_balance.get("penalty", 0.0)
                )
                motif = dict(motif_by_formula.get(hypothesis, {}))
                payload["perovskite_motif_hypothesis"] = motif or None
                payload["perovskite_motif_prior_penalty"] = float(
                    motif.get("motif_prior_penalty", 0.12) if motif else 0.12
                )
                estimated_density = _estimated_candidate_density_g_cm3(
                    payload.get("composition_elements", {}),
                    candidate,
                )
                payload["estimated_density_g_cm3"] = estimated_density
                payload["estimated_density_penalty"] = (
                    _estimated_density_plausibility_penalty(
                        payload.get("composition_elements", {}),
                        estimated_density,
                    )
                )
                payload["axis_scale_penalty"] = _axis_scale_prior_penalty(
                    str(candidate.candidate_id)
                )
                payload["perovskite_axis_penalty"] = (
                    _perovskite_axis_prior_penalty(
                        payload.get("composition_elements", {}),
                        candidate,
                    )
                )
                payload["lattice_prior_penalty"] = float(
                    payload["axis_scale_penalty"]
                    + payload["perovskite_axis_penalty"]
                )
                payload["score"] = float(payload.get("score", math.inf)) + (
                    0.015 * (hypothesis_rank - 1)
                    + float(payload["charge_balance_penalty"])
                    + float(payload["perovskite_motif_prior_penalty"])
                    + float(payload["estimated_density_penalty"])
                    + float(payload["lattice_prior_penalty"])
                )
                candidate_id = str(payload.get("candidate_id") or "")
                if hypothesis_rank == 1:
                    reduced_records_by_candidate[candidate_id] = dict(payload)
                elif (
                    hypothesis_rank == 2
                    and candidate_id in reduced_records_by_candidate
                ):
                    reduced = reduced_records_by_candidate[candidate_id]
                    payload["cif_text"] = _duplicate_cif_motif_text(
                        str(reduced.get("cif_text") or ""),
                        payload.get("composition_elements", {}),
                        translation=(0.5, 0.5, 0.5),
                    )
                    payload["coordinate_model"] = (
                        "translated_duplicate_of_reduced_motif"
                    )
                    payload["motif_source_cif_id"] = reduced.get("cif_id")
                records.append(payload)
    records.sort(key=lambda item: float(item.get("score", math.inf)))
    return _diverse_cif_record_subset(
        records,
        limit=max(1, cfg.max_generated_cifs_to_compare),
    )


def _diverse_cif_record_subset(
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

    def force_add(record: dict[str, Any]) -> None:
        cif_id = str(record.get("cif_id") or "")
        if cif_id in selected_ids:
            return
        if len(selected) >= limit and selected:
            removed = max(
                selected,
                key=lambda item: float(item.get("score", math.inf)),
            )
            selected.remove(removed)
            selected_ids.discard(str(removed.get("cif_id") or ""))
        add(record)

    seen_hypotheses: set[str] = set()
    for record in records:
        hypothesis = str(record.get("solve_stoichiometry_hypothesis") or "")
        if hypothesis in seen_hypotheses:
            continue
        add(record)
        seen_hypotheses.add(hypothesis)
        if len(seen_hypotheses) >= max(1, limit // 3):
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
    selected_candidate_ids = {
        str(record.get("candidate_id") or "")
        for record in selected
        if int(record.get("solve_stoichiometry_hypothesis_rank", 0) or 0) == 1
    }
    for candidate_id in selected_candidate_ids:
        for record in records:
            if str(record.get("candidate_id") or "") != candidate_id:
                continue
            if (
                int(record.get("solve_stoichiometry_hypothesis_rank", 0) or 0)
                != 2
            ):
                continue
            force_add(record)
            break
    selected.sort(key=lambda item: float(item.get("score", math.inf)))
    return selected


def _duplicate_cif_motif_text(
    cif_text: str,
    composition: Any,
    *,
    translation: tuple[float, float, float],
) -> str:
    lines = cif_text.splitlines()
    translation = _best_duplicate_translation(lines, translation)
    duplicated: list[str] = []
    atom_rows: list[str] = []
    in_atom_rows = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("_chemical_formula_sum"):
            duplicated.append(
                f"_chemical_formula_sum '{_format_formula_sum(composition)}'"
            )
            continue
        duplicated.append(line)
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            in_atom_rows
            and stripped
            and not stripped.startswith(("_", "#", "loop_"))
        ):
            atom_rows.append(line)
    for line in atom_rows:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            coords = [float(parts[index]) for index in (2, 3, 4)]
        except ValueError:
            continue
        shifted = [
            (coord + delta) % 1.0 for coord, delta in zip(coords, translation)
        ]
        parts[0] = f"{parts[0]}B"
        parts[2:5] = [f"{value:.6f}" for value in shifted]
        duplicated.append(" ".join(parts))
    duplicated = _relayout_duplicate_molecules(duplicated)
    return "\n".join(duplicated).rstrip() + "\n"


def _best_duplicate_translation(
    lines: list[str],
    preferred: tuple[float, float, float],
) -> tuple[float, float, float]:
    rows, lattice = _duplicate_source_rows_and_lattice(lines)
    if not rows or lattice is None:
        return preferred
    candidates = [
        (float(preferred[0]), float(preferred[1]), float(preferred[2]))
    ]
    candidates.extend(
        (x, y, z)
        for x in (0.0, 0.5)
        for y in (0.0, 0.5)
        for z in (0.0, 0.5)
        if (x, y, z) != (0.0, 0.0, 0.0)
    )
    deduped = list(dict.fromkeys(candidates))
    return min(
        deduped,
        key=lambda item: (
            _duplicate_translation_score(rows, lattice, item),
            sum(abs(item[index] - preferred[index]) for index in range(3)),
        ),
    )


def _duplicate_source_rows_and_lattice(
    lines: list[str],
) -> tuple[list[tuple[str, str, np.ndarray]], np.ndarray | None]:
    cell: dict[str, float] = {}
    rows: list[tuple[str, str, np.ndarray]] = []
    in_atom_rows = False
    for line in lines:
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
                cell[parts[0]] = float(parts[1])
            except (IndexError, ValueError):
                pass
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            not in_atom_rows
            or stripped.startswith(("_", "#", "loop_"))
            or len(parts) < 6
        ):
            continue
        try:
            frac = np.asarray([float(parts[index]) for index in (2, 3, 4)])
        except ValueError:
            continue
        rows.append((str(parts[0]), str(parts[1]), frac))
    return rows, _duplicate_lattice_matrix(cell)


def _relayout_duplicate_molecules(lines: list[str]) -> list[str]:
    rows, lattice = _duplicate_rows_with_indices_and_lattice(lines)
    if not rows or lattice is None:
        return lines
    bodies: dict[str, list[dict[str, Any]]] = {}
    fixed_sites: list[tuple[str, np.ndarray]] = []
    for row in rows:
        if row["element"] in {"C", "H", "N", "O", "S"}:
            token = _duplicate_molecular_body_token(str(row["label"]))
            if token:
                bodies.setdefault(token, []).append(row)
                continue
        fixed_sites.append((str(row["element"]), np.asarray(row["frac"])))
    if not bodies or not fixed_sites:
        return lines
    updated = list(lines)
    for _, body_rows in sorted(
        bodies.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        coords = np.asarray([row["frac"] for row in body_rows], dtype=float)
        elements = [str(row["element"]) for row in body_rows]
        center = _duplicate_fractional_center(coords)
        offsets = _duplicate_body_cart_offsets(coords, center, lattice)
        best_coords = coords
        best_score = -math.inf
        for candidate_center in _duplicate_molecule_seed_centers(center):
            candidate_coords = (
                np.asarray(candidate_center, dtype=float)
                + offsets @ np.linalg.inv(lattice)
            ) % 1.0
            score = _duplicate_molecule_seed_score(
                candidate_coords,
                elements,
                fixed_sites,
                lattice,
            )
            if score > best_score:
                best_score = score
                best_coords = candidate_coords
        for row, coord in zip(body_rows, best_coords, strict=True):
            parts = updated[int(row["line_index"])].split()
            parts[2:5] = [f"{value:.6f}" for value in coord % 1.0]
            updated[int(row["line_index"])] = " ".join(parts)
            fixed_sites.append((str(row["element"]), coord % 1.0))
    return updated


def _duplicate_rows_with_indices_and_lattice(
    lines: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    cell: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    in_atom_rows = False
    for line_index, line in enumerate(lines):
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
                cell[parts[0]] = float(parts[1])
            except (IndexError, ValueError):
                pass
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            not in_atom_rows
            or stripped.startswith(("_", "#", "loop_"))
            or len(parts) < 6
        ):
            continue
        try:
            frac = np.asarray([float(parts[index]) for index in (2, 3, 4)])
        except ValueError:
            continue
        rows.append(
            {
                "label": str(parts[0]),
                "element": str(parts[1]),
                "frac": frac,
                "line_index": line_index,
            }
        )
    return rows, _duplicate_lattice_matrix(cell)


def _duplicate_molecular_body_token(label: str) -> str:
    if "_" not in label:
        return ""
    body, atom_token = label.split("_", 1)
    suffix = ""
    for char in reversed(atom_token):
        if not char.isalpha():
            break
        suffix = char + suffix
    return f"{body}{suffix}" if suffix else body


def _duplicate_fractional_center(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros(3, dtype=float)
    reference = np.asarray(coords[0], dtype=float)
    unwrapped = []
    for coord in coords:
        delta = np.asarray(coord, dtype=float) - reference
        delta -= np.round(delta)
        unwrapped.append(reference + delta)
    return np.mean(np.asarray(unwrapped), axis=0) % 1.0


def _duplicate_body_cart_offsets(
    coords: np.ndarray,
    center: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(coords, dtype=float) - np.asarray(center, dtype=float)
    delta -= np.round(delta)
    return delta @ lattice


def _duplicate_molecule_seed_centers(center: np.ndarray) -> list[np.ndarray]:
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
    centers = [np.asarray(center, dtype=float), *map(np.asarray, grid)]
    centers.extend(np.asarray(site, dtype=float) for site in symmetry_sites)
    deduped: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate in centers:
        wrapped = np.asarray(candidate, dtype=float) % 1.0
        key = tuple(int(round(value * 1000.0)) for value in wrapped)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(wrapped)
    return deduped


def _duplicate_molecule_seed_score(
    coords: np.ndarray,
    elements: list[str],
    fixed_sites: list[tuple[str, np.ndarray]],
    lattice: np.ndarray,
) -> float:
    min_distance = math.inf
    clash_penalty = 0.0
    for coord, element in zip(coords, elements, strict=True):
        for fixed_element, fixed_coord in fixed_sites:
            distance = _duplicate_pbc_distance(coord, fixed_coord, lattice)
            minimum = _duplicate_minimum_contact_distance(
                element,
                fixed_element,
            )
            min_distance = min(min_distance, distance)
            if distance < minimum:
                weight = (
                    2.5
                    if _duplicate_organic_inorganic_pair(
                        element,
                        fixed_element,
                    )
                    else 1.0
                )
                clash_penalty += (
                    weight
                    * (minimum - distance)
                    / max(
                        minimum,
                        1.0e-9,
                    )
                )
    if not np.isfinite(min_distance):
        min_distance = 4.0
    return (
        min(min_distance, 4.0)
        - 4.0 * clash_penalty
        - 5.0 * _duplicate_unit_cell_boundary_penalty(coords)
    )


def _duplicate_unit_cell_boundary_penalty(
    coords: np.ndarray,
    *,
    span_limit: float = 0.72,
) -> float:
    if coords.size == 0:
        return 0.0
    wrapped = np.asarray(coords, dtype=float) % 1.0
    spans = np.max(wrapped, axis=0) - np.min(wrapped, axis=0)
    return float(np.sum(np.clip(spans - span_limit, 0.0, None)))


def _duplicate_lattice_matrix(
    cell: dict[str, float],
) -> np.ndarray | None:
    try:
        a = float(cell["_cell_length_a"])
        b = float(cell["_cell_length_b"])
        c = float(cell["_cell_length_c"])
        alpha = math.radians(float(cell["_cell_angle_alpha"]))
        beta = math.radians(float(cell["_cell_angle_beta"]))
        gamma = math.radians(float(cell["_cell_angle_gamma"]))
    except KeyError:
        return None
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) <= 1.0e-12:
        return None
    avec = np.asarray([a, 0.0, 0.0])
    bvec = np.asarray([b * math.cos(gamma), b * sin_gamma, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    return np.vstack([avec, bvec, np.asarray([cx, cy, math.sqrt(cz_sq)])])


def _duplicate_translation_score(
    rows: list[tuple[str, str, np.ndarray]],
    lattice: np.ndarray,
    translation: tuple[float, float, float],
) -> float:
    shift = np.asarray(translation, dtype=float)
    shifted = [
        (label, element, (frac + shift) % 1.0) for label, element, frac in rows
    ]
    clash_score = 0.0
    for _, left_element, left_frac in rows:
        for _, right_element, right_frac in shifted:
            distance = _duplicate_pbc_distance(left_frac, right_frac, lattice)
            minimum = _duplicate_minimum_contact_distance(
                left_element,
                right_element,
            )
            if distance < minimum:
                weight = (
                    2.5
                    if _duplicate_organic_inorganic_pair(
                        left_element,
                        right_element,
                    )
                    else 1.0
                )
                clash_score += (
                    weight
                    * (minimum - distance)
                    / max(
                        minimum,
                        1.0e-9,
                    )
                )
    return clash_score + 3.0 * _duplicate_free_ion_penalty(
        [*rows, *shifted],
        lattice,
    )


def _duplicate_free_ion_penalty(
    rows: list[tuple[str, str, np.ndarray]],
    lattice: np.ndarray,
) -> float:
    cations = [row for row in rows if row[1] in {"Pb", "Sn", "Ge"}]
    halides = [row for row in rows if row[1] in {"I", "Br", "Cl", "F"}]
    if not cations or not halides:
        return 0.0
    penalty = 0.0
    for _, element, frac in cations:
        neighbors = [
            _duplicate_pbc_distance(frac, halide_frac, lattice)
            for _, halide_element, halide_frac in halides
            if _duplicate_cation_halide_pair(element, halide_element)
        ]
        near = [distance for distance in neighbors if 2.55 <= distance <= 3.85]
        if not near:
            penalty += 1.0
        elif len(near) < 3:
            penalty += 0.4 * (3 - len(near))
    for _, element, frac in halides:
        neighbors = [
            _duplicate_pbc_distance(frac, cation_frac, lattice)
            for _, cation_element, cation_frac in cations
            if _duplicate_cation_halide_pair(cation_element, element)
        ]
        near = [distance for distance in neighbors if 2.55 <= distance <= 3.85]
        if not near:
            penalty += 0.5
    return penalty


def _duplicate_pbc_distance(
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


def _duplicate_minimum_contact_distance(
    left_element: str,
    right_element: str,
) -> float:
    if _duplicate_cation_halide_pair(left_element, right_element):
        return 2.25
    if _duplicate_organic_inorganic_pair(left_element, right_element):
        return 1.55 if "H" in {left_element, right_element} else 2.25
    if left_element in {"Pb", "Sn", "Ge"} and right_element in {
        "Pb",
        "Sn",
        "Ge",
    }:
        return 3.2
    if left_element in {"I", "Br", "Cl", "F"} and right_element in {
        "I",
        "Br",
        "Cl",
        "F",
    }:
        return 2.45
    if "H" in {left_element, right_element}:
        return 0.75
    return 1.45


def _duplicate_organic_inorganic_pair(
    left_element: str,
    right_element: str,
) -> bool:
    organic = {"C", "H", "N", "O", "S"}
    inorganic = {"Pb", "Sn", "Ge", "I", "Br", "Cl", "F"}
    return (left_element in organic and right_element in inorganic) or (
        right_element in organic and left_element in inorganic
    )


def _duplicate_cation_halide_pair(
    left_element: str,
    right_element: str,
) -> bool:
    return (
        left_element in {"Pb", "Sn", "Ge"}
        and right_element in {"I", "Br", "Cl", "F"}
    ) or (
        right_element in {"Pb", "Sn", "Ge"}
        and left_element in {"I", "Br", "Cl", "F"}
    )


def _map_detector_to_qspace(
    image: np.ndarray,
    poni_file: Path,
    mask: np.ndarray | None,
    cfg: ExperimentalRefinementConfig,
) -> xr.DataArray:
    return map_grazing_incidence_qspace(
        image,
        poni_file,
        config=GrazingIncidenceConfig(
            npt_ip=cfg.qspace_shape[1],
            npt_oop=cfg.qspace_shape[0],
            ip_range=cfg.qspace_ip_range,
            oop_range=cfg.qspace_oop_range,
            xray_energy_kev=cfg.xray_energy_kev,
            incident_angle_deg=cfg.incident_angle_deg,
            tilt_angle_deg=cfg.tilt_angle_deg,
            sample_orientation=cfg.sample_orientation,
            correct_solid_angle=cfg.correct_solid_angle,
            polarization_factor=cfg.polarization_factor,
            normalization_factor=cfg.normalization_factor,
        ),
        mask=mask,
    )


def _standardize_qspace(qspace: xr.DataArray) -> xr.DataArray:
    target = qspace.rename({"q_ip": "qxy", "q_oop": "qz"})
    values = np.nan_to_num(np.asarray(target.values, dtype=float), nan=0.0)
    values = np.clip(values, 0.0, None)
    return xr.DataArray(
        values,
        dims=("qz", "qxy"),
        coords={
            "qz": np.asarray(target.coords["qz"].values, dtype=float),
            "qxy": np.asarray(target.coords["qxy"].values, dtype=float),
        },
        name="experimental_qspace_intensity",
        attrs=dict(qspace.attrs),
    )


def _benchmark_config_for_experiment(
    cfg: ExperimentalRefinementConfig,
    target: xr.DataArray,
) -> BenchmarkRunConfig:
    qxy = np.asarray(target.coords["qxy"].values, dtype=float)
    qz = np.asarray(target.coords["qz"].values, dtype=float)
    return BenchmarkRunConfig(
        output_dir=cfg.output_dir,
        hkl_extent=cfg.hkl_extent,
        detector_shape=(target.shape[0], target.shape[1]),
        qxy_range=(float(np.nanmin(qxy)), float(np.nanmax(qxy))),
        qz_range=(float(np.nanmin(qz)), float(np.nanmax(qz))),
        sigma_theta=cfg.sigma_theta,
        sigma_phi=cfg.sigma_phi,
        sigma_r=cfg.sigma_r,
        fiber_tilt_center_deg=cfg.fiber_tilt_center_deg,
        peak_threshold_percentile=cfg.peak_threshold_percentile,
        peak_adaptive_threshold=cfg.peak_adaptive_threshold,
        peak_adaptive_floor_percentile=cfg.peak_adaptive_floor_percentile,
        peak_min_snr=cfg.peak_min_snr,
        peak_max_peaks=cfg.peak_max_peaks,
        peak_min_distance_px=cfg.peak_min_distance_px,
        peak_deduplicate_tolerance=cfg.peak_deduplicate_tolerance,
        candidate_hkl_max=cfg.candidate_hkl_max,
        candidate_q_tolerance=cfg.candidate_q_tolerance,
        candidate_relative_tolerance=cfg.candidate_relative_tolerance,
        candidate_lattice_min=cfg.candidate_lattice_min,
        candidate_lattice_max=cfg.candidate_lattice_max,
        candidate_grid_points=cfg.candidate_grid_points,
        candidate_max_candidates=cfg.candidate_max_candidates,
        max_generated_cifs_to_compare=cfg.max_generated_cifs_to_compare,
        comparison_theta_x_offsets=cfg.comparison_theta_x_offsets,
        comparison_theta_y_values=cfg.comparison_theta_y_values,
        comparison_plot_count=cfg.comparison_plot_count,
        staged_refinement=cfg.staged_refinement,
        refinement_coarse_detector_shape=cfg.refinement_coarse_detector_shape,
        refinement_coarse_hkl_extent=cfg.refinement_coarse_hkl_extent,
        refinement_fractional_step=cfg.refinement_fractional_step,
        refinement_scaffold_variant_count=cfg.refinement_scaffold_variant_count,
        bragg_intensity_weight=cfg.bragg_intensity_weight,
        bragg_intensity_tolerance=cfg.bragg_intensity_tolerance,
        bragg_intensity_max_peaks=cfg.bragg_intensity_max_peaks,
    )


def _simulation_params_for_target(
    target: xr.DataArray,
    cfg: BenchmarkRunConfig,
) -> GIWAXSSimulationParameters:
    qxy = np.asarray(target.coords["qxy"].values, dtype=float)
    qz = np.asarray(target.coords["qz"].values, dtype=float)
    return GIWAXSSimulationParameters(
        sigma_theta=cfg.sigma_theta,
        sigma_phi=cfg.sigma_phi,
        sigma_r=cfg.sigma_r,
        hkl_extent=cfg.hkl_extent,
        theta_x_deg=cfg.fiber_tilt_center_deg,
        theta_y_deg=0.0,
        qxy_min=float(np.nanmin(qxy)),
        qxy_max=float(np.nanmax(qxy)),
        qz_min=float(np.nanmin(qz)),
        qz_max=float(np.nanmax(qz)),
        resolution_z=target.shape[0],
        resolution_x=target.shape[1],
    )


def _experimental_project(
    spec: BenchmarkStructureSpec,
    image_path: Path,
    poni_path: Path,
    mask_path: Path | None,
    qspace_path: Path,
    cfg: ExperimentalRefinementConfig,
    target: xr.DataArray,
) -> ProjectState:
    project = ProjectState(name=f"Experimental refinement {spec.structure_id}")
    data_file = project.add_data_file(
        image_path,
        experimental_refinement_fileset_id=spec.structure_id,
        reference_cif=str(spec.cif_path),
        synthetic=False,
    )
    data_id = str(data_file.data_id)
    project.processed_products[data_id] = str(qspace_path)
    mask_asset_id = None
    if mask_path is not None:
        mask_asset = CorrectionAssetRef(
            kind="mask",
            name=mask_path.name,
            path=mask_path,
            target_ids=[data_id],
            metadata={"role": "experimental_refinement_mask"},
        )
        mask_asset_id = mask_asset.asset_id
        project.masks.append(mask_asset)
    calibrant_asset = CorrectionAssetRef(
        kind="poni",
        name=poni_path.name,
        path=poni_path,
        target_ids=[data_id],
        metadata={"role": "experimental_refinement_calibration"},
    )
    project.calibrants.append(calibrant_asset)
    xray_energy_kev = cfg.xray_energy_kev or _energy_from_qspace_attrs(target)
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask_asset_id,
            calibrant_asset_id=calibrant_asset.asset_id,
            xray_energy_kev=xray_energy_kev,
            pyfai_sample_orientation=cfg.sample_orientation,
            correct_solid_angle=cfg.correct_solid_angle,
            polarization_factor=cfg.polarization_factor,
            normalization_factor=cfg.normalization_factor,
            confirmed=True,
            metadata={
                "incident_angle_deg": cfg.incident_angle_deg,
                "tilt_angle_deg": cfg.tilt_angle_deg,
                "qspace_product": str(qspace_path),
            },
        )
    )
    return project


def _store_experimental_peak_records(
    project: ProjectState,
    data_id: str,
    peaks: list[Any],
    families: list[dict[str, Any]],
) -> None:
    from ewald.benchmark.structure_benchmark import _store_peak_records
    from ewald.data.models import (
        STRUCTURE_ANALYSIS_KEY,
        STRUCTURE_ANALYSIS_PEAKS_KEY,
    )

    _store_peak_records(project, data_id, peaks)
    analysis = project.analysis_results.setdefault(STRUCTURE_ANALYSIS_KEY, {})
    entry = analysis.setdefault(
        data_id,
        {STRUCTURE_ANALYSIS_PEAKS_KEY: [peak.as_dict() for peak in peaks]},
    )
    entry["families"] = families


def _write_experiment_qspace_npz(
    path: Path,
    image: xr.DataArray,
    spec: BenchmarkStructureSpec,
    detector_image: Path,
    poni_file: Path,
    mask_file: Path | None,
    cfg: ExperimentalRefinementConfig,
) -> None:
    np.savez(
        path,
        intensity=np.asarray(image.values, dtype=float),
        q_ip=np.asarray(image.coords["qxy"].values, dtype=float),
        q_oop=np.asarray(image.coords["qz"].values, dtype=float),
        metadata_json=json.dumps(
            _json_safe(
                {
                    "workflow": "ewald_experimental_refinement",
                    "reference_cif": str(spec.cif_path),
                    "detector_image": str(detector_image),
                    "poni_file": str(poni_file),
                    "mask_file": str(mask_file) if mask_file else None,
                    "solve_constraints": _solve_constraints(spec),
                    "config": cfg.as_dict(),
                }
            ),
            sort_keys=True,
        ),
    )


def _load_detector_image(path: Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(path), dtype=float)


def _load_mask_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path)).astype(bool)
    if suffix == ".npz":
        payload = np.load(path)
        first_key = sorted(payload.files)[0]
        return np.asarray(payload[first_key]).astype(bool)
    if suffix in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(path)).astype(bool)
    import fabio

    return np.asarray(fabio.open(str(path)).data).astype(bool)


def _append_experimental_header(
    logbook: Path,
    *,
    run_id: str,
    spec: BenchmarkStructureSpec,
    detector_image: Path,
    poni_file: Path,
    mask_file: Path | None,
    cfg: ExperimentalRefinementConfig,
) -> None:
    _append_logbook(
        logbook,
        [
            f"# EWALD Experimental Refinement {run_id}",
            "",
            f"- Created: {_timestamp()}",
            f"- Detector image: `{detector_image}`",
            f"- PONI: `{poni_file}`",
            f"- Mask: `{mask_file or 'none'}`",
            f"- Validation CIF: `{spec.cif_path}`",
            (
                "- Solver input policy: inorganic atom labels and organic "
                "molecule labels only; validation CIF stoichiometry, cell, "
                "and coordinates are excluded from candidate generation."
            ),
            "",
            "## Refinement Protocol",
            "",
            (
                "- Map detector counts into qIP/qOOP using the PONI and mask, "
                "then detect Bragg-like maxima and q-coordinate families."
            ),
            (
                "- Generate broad perovskite-derivative stoichiometry "
                "hypotheses from allowed chemistry, not from the reference "
                "file."
            ),
            (
                "- Fit lattice candidates from peak families, write full CIFs "
                "with inorganic atoms and rigid full-molecule motifs, then "
                "score simulated GIWAXS against the experiment."
            ),
            (
                "- Refine in SHELX/Rietveld spirit: heavy-atom scaffold first, "
                "damped secondary-anion shifts next, rigid-body molecule "
                "translation last, with coarse-to-finer simulation checks."
            ),
            (
                "- Accept a solution only after comparing the generated CIF "
                "against the held-out validation CIF."
            ),
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(_json_safe(cfg.as_dict()), indent=2, sort_keys=True),
            "```",
            "",
        ],
    )


def _append_experimental_logbook(
    logbook: Path,
    summary: dict[str, Any],
    params: GIWAXSSimulationParameters,
    peaks: list[Any],
    families: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    hypotheses: list[str],
    comparisons: list[dict[str, Any]],
) -> None:
    _append_fileset_logbook(
        logbook,
        {
            **summary,
            "mock_tiff": summary["detector_image"],
            "solve_order": 1,
        },
        params,
        peaks,
        candidates,
        comparisons,
    )
    family_lines = [
        "",
        "Peak families:",
        "",
        "| family | kind | members | coordinate |",
        "| --- | --- | ---: | ---: |",
    ]
    for family in families[:12]:
        family_lines.append(
            "| {family} | {kind} | {count} | {value:.4g} |".format(
                family=family.get("family_id", ""),
                kind=family.get("kind", ""),
                count=len(family.get("peak_ids", [])),
                value=float(
                    family.get(
                        "reference",
                        math.nan,
                    )
                ),
            )
        )
    family_lines.extend(
        [
            "",
            "Chemistry hypotheses searched:",
            "",
            ", ".join(f"`{item}`" for item in hypotheses) or "none",
            "",
        ]
    )
    _append_logbook(logbook, family_lines)


def _molecule_count_patterns(
    molecules: list[str],
) -> list[tuple[tuple[str, int], ...]]:
    if not molecules:
        return [()]
    if len(molecules) == 1:
        label = molecules[0]
        return [((label, 1),), ((label, 2),), ((label, 3),)]
    primary = molecules[0]
    secondary = molecules[1:]
    patterns: list[tuple[tuple[str, int], ...]] = [
        ((primary, 2), (secondary[0], 2)),
        ((primary, 2), (secondary[0], 1)),
        ((primary, 3), (secondary[0], 1)),
        ((primary, 1), (secondary[0], 1)),
        ((primary, 1), (secondary[0], 2)),
        ((primary, 2),),
        ((primary, 1),),
        ((primary, 3),),
    ]
    for secondary_label in secondary[1:]:
        patterns.extend(
            [
                ((primary, 2), (secondary_label, 2)),
                ((primary, 1), (secondary_label, 1)),
            ]
        )
    return list(dict.fromkeys(patterns))


def _formula_piece(label: str, count: int) -> str:
    token = _molecule_token(label) if not _is_element_symbol(label) else label
    if _is_element_symbol(token):
        return token if count == 1 else f"{token}{count}"
    return f"({token}){count}" if count != 1 else f"({token})"


def _formula_id(value: str) -> str:
    return _slug(value.replace("(", "").replace(")", ""))


def _estimated_candidate_density_g_cm3(
    composition: Any,
    candidate: Any,
) -> float | None:
    if not isinstance(composition, dict):
        return None
    mass_amu = 0.0
    for element, count in composition.items():
        mass = ATOMIC_MASSES_AMU.get(str(element))
        if mass is None:
            continue
        try:
            mass_amu += mass * float(count)
        except (TypeError, ValueError):
            continue
    if mass_amu <= 0.0:
        return None
    try:
        a = float(candidate.a)
        b = float(candidate.b)
        c = float(candidate.c)
        alpha = math.radians(float(getattr(candidate, "alpha", 90.0)))
        beta = math.radians(float(getattr(candidate, "beta", 90.0)))
        gamma = math.radians(float(getattr(candidate, "gamma", 90.0)))
    except (AttributeError, TypeError, ValueError):
        return None
    volume_factor = math.sqrt(
        max(
            0.0,
            1.0
            + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
            - math.cos(alpha) ** 2
            - math.cos(beta) ** 2
            - math.cos(gamma) ** 2,
        )
    )
    volume_a3 = a * b * c * volume_factor
    if volume_a3 <= 1.0e-12:
        return None
    return mass_amu * 1.66053906660 / volume_a3


def _estimated_density_plausibility_penalty(
    composition: Any,
    density_g_cm3: float | None,
) -> float:
    if density_g_cm3 is None or not np.isfinite(density_g_cm3):
        return 0.0
    if not isinstance(composition, dict):
        return 0.0
    elements = {str(element) for element in composition}
    if not (elements & {"Pb", "Sn"}) or not (elements & {"I", "Br", "Cl"}):
        return 0.0
    density = float(density_g_cm3)
    if density < 3.0:
        return min(1.25, (3.0 - density) * 0.55)
    if density > 7.5:
        return min(1.0, (density - 7.5) * 0.20)
    return 0.0


def _axis_scale_prior_penalty(candidate_id: str) -> float:
    marker = "_axis_scale_"
    if marker not in candidate_id:
        return 0.0
    raw = candidate_id.rsplit(marker, 1)[-1]
    scales = []
    for token in raw.split("_")[:3]:
        try:
            scales.append(float(token.replace("p", ".")))
        except ValueError:
            return 0.0
    if len(scales) != 3:
        return 0.0
    changed_axes = sum(abs(value - 1.0) > 1.0e-9 for value in scales)
    penalty = 0.035 * changed_axes
    if changed_axes == 3 and max(scales) < 1.0:
        penalty += 0.18
    return penalty


def _perovskite_axis_prior_penalty(composition: Any, candidate: Any) -> float:
    if not isinstance(composition, dict):
        return 0.0
    elements = {str(element) for element in composition}
    if not (elements & {"Pb", "Sn"}) or not (elements & {"I", "Br", "Cl"}):
        return 0.0
    try:
        lengths = [
            float(candidate.a),
            float(candidate.b),
            float(candidate.c),
        ]
    except (AttributeError, TypeError, ValueError):
        return 0.0
    shortest = min(lengths)
    if shortest < 3.2:
        return min(0.35, (3.2 - shortest) * 0.20)
    if shortest > 7.5:
        return min(0.85, 0.35 + (shortest - 7.5) * 0.12)
    return 0.0


def _formula_charge_balance(formula: str) -> dict[str, Any]:
    charges = {
        "MA": 1.0,
        "FA": 1.0,
        "BA": 1.0,
        "DMF": 0.0,
        "DMSO": 0.0,
        "NMP": 0.0,
        "Cs": 1.0,
        "Rb": 1.0,
        "K": 1.0,
        "Na": 1.0,
        "Pb": 2.0,
        "Sn": 2.0,
        "Ge": 2.0,
        "Bi": 3.0,
        "Sb": 3.0,
        "I": -1.0,
        "Br": -1.0,
        "Cl": -1.0,
        "F": -1.0,
    }
    counts = _formula_species_counts(formula)
    net_charge = sum(
        float(count) * charges.get(species, 0.0)
        for species, count in counts.items()
    )
    charge_scale = sum(
        abs(float(count) * charges.get(species, 0.0))
        for species, count in counts.items()
    )
    relative = abs(net_charge) / max(charge_scale, 1.0)
    return {
        "net_charge": net_charge,
        "relative_imbalance": relative,
        "penalty": min(0.3, relative * 0.3),
        "species_counts": counts,
    }


def _formula_species_counts(formula: str) -> dict[str, float]:
    compact = "".join(str(formula).split())
    counts: dict[str, float] = {}
    index = 0
    while index < len(compact):
        char = compact[index]
        if char == "(":
            close = compact.find(")", index + 1)
            if close < 0:
                index += 1
                continue
            token = compact[index + 1 : close]
            multiplier, index = _parse_formula_number(compact, close + 1)
            counts[token] = counts.get(token, 0.0) + multiplier
            continue
        if char.isupper():
            end = index + 1
            if end < len(compact) and compact[end].islower():
                end += 1
            token = compact[index:end]
            multiplier, index = _parse_formula_number(compact, end)
            counts[token] = counts.get(token, 0.0) + multiplier
            continue
        index += 1
    return counts


def _parse_formula_number(value: str, index: int) -> tuple[float, int]:
    start = index
    while index < len(value) and (
        value[index].isdigit() or value[index] == "."
    ):
        index += 1
    if start == index:
        return 1.0, index
    try:
        return float(value[start:index]), index
    except ValueError:
        return 1.0, index


def _format_formula_sum(composition: Any) -> str:
    if not isinstance(composition, dict):
        return "X1"
    pieces = []
    for element in sorted(composition):
        try:
            count = float(composition[element])
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        count_text = (
            str(int(round(count)))
            if abs(count - round(count)) <= 1.0e-9
            else f"{count:.4g}"
        )
        pieces.append(f"{element}{count_text}")
    return " ".join(pieces) or "X1"


def _scale_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _molecule_token(value: str) -> str:
    return "".join(char for char in str(value).strip() if char.isalnum())


def _element_symbol(symbol: str) -> str:
    letters = "".join(char for char in str(symbol) if char.isalpha())
    if not letters:
        return str(symbol)
    if len(letters) == 1:
        return letters.upper()
    return f"{letters[0].upper()}{letters[1:].lower()}"


def _is_element_symbol(value: str) -> bool:
    return (
        bool(value)
        and value[0].isupper()
        and (len(value) == 1 or value[1:].islower())
    )


def _energy_from_qspace_attrs(target: xr.DataArray) -> float | None:
    value = target.attrs.get("xray_energy_kev")
    if value:
        return float(value)
    wavelength = target.attrs.get("wavelength_m")
    if wavelength:
        return xray_energy_kev_from_wavelength_m(float(wavelength))
    return None
