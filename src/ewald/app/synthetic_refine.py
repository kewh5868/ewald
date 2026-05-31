"""Run synthetic EWALD refinement diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ewald.benchmark import (
    SyntheticRefinementConfig,
    default_structure_specs,
    load_structure_specs,
    run_synthetic_refinement_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ewald-synthetic-refine",
        description=(
            "Generate fiber-textured synthetic GIWAXS datasets, solve blind "
            "peak/family/lattice/scaffold guesses, and write truth diagnostics "
            "under the experimental refinement output folder."
        ),
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("example/structures"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example/projects/experimental_refinement"),
    )
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--simulations-per-structure", type=int, default=2)
    parser.add_argument("--structure-limit", type=int)
    parser.add_argument("--hkl-extent", type=int, default=5)
    parser.add_argument(
        "--detector-shape",
        default="128x192",
        help="Detector shape as rows x columns, for example 128x192.",
    )
    parser.add_argument(
        "--qxy-range",
        default="-4.5,4.5",
        help="qxy range as min,max.",
    )
    parser.add_argument(
        "--qz-range",
        default="0.0,4.5",
        help="qz range as min,max.",
    )
    parser.add_argument("--peak-max-peaks", type=int, default=120)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--max-generated-cifs", type=int, default=6)
    parser.add_argument("--max-scaffolds", type=int, default=10)
    parser.add_argument("--max-organic-proxies", type=int, default=8)
    parser.add_argument("--max-organic-replacements", type=int, default=6)
    parser.add_argument("--stage-simulation-max-cifs", type=int, default=6)
    parser.add_argument("--organic-rmc-steps", type=int, default=12)
    parser.add_argument(
        "--organic-rmc-translation-step",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--organic-rmc-rotation-step-deg",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--bragg-intensity-weight",
        type=float,
        default=0.35,
        help=(
            "Weight for relative Bragg peak intensity matching in scaffold "
            "and generated-CIF ranking. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--bragg-intensity-tolerance",
        type=float,
        default=0.08,
        help="q-space tolerance used when matching detected peaks to simulated Bragg peaks.",
    )
    parser.add_argument(
        "--bragg-intensity-max-peaks",
        type=int,
        default=80,
        help="Maximum strongest detected peaks used for Bragg intensity matching.",
    )
    parser.add_argument(
        "--no-organic-proxies",
        action="store_true",
        help="Disable charge-balanced organic electron proxy CIF generation.",
    )
    parser.add_argument(
        "--no-organic-replacements",
        action="store_true",
        help="Disable full organic replacement and organic RMC stages.",
    )
    parser.add_argument(
        "--no-unit-cell-symmetry",
        action="store_true",
        help=(
            "Use unconstrained deterministic atom/molecule placement instead "
            "of symmetry-constrained unit-cell placement."
        ),
    )
    parser.add_argument(
        "--texture-modes",
        default=(
            "out_of_plane_stack,in_plane_stack,"
            "tilted_out_of_plane_stack,tilted_in_plane_stack"
        ),
        help=(
            "Comma-separated texture schedule. Examples: "
            "out_of_plane_stack,in_plane_stack,tilted_in_plane_stack."
        ),
    )
    parser.add_argument(
        "--texture-azimuth-jitter-deg",
        type=float,
        default=6.0,
        help=(
            "Maximum random azimuth jitter around each simple fibril "
            "texture orientation center, in degrees."
        ),
    )
    parser.add_argument(
        "--image-rerank",
        action="store_true",
        help="Also run low-budget simulated-image reranking for generated CIFs.",
    )
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help=(
            "Use exact reference lattice, synthetic truth peak positions/hkl "
            "families, and reference stoichiometry constraints for an "
            "upper-bound solver diagnostic. Atom coordinates remain hidden."
        ),
    )
    parser.add_argument(
        "--oracle-lattice",
        action="store_true",
        help="Use the exact reference unit-cell parameters as a candidate.",
    )
    parser.add_argument(
        "--oracle-peaks",
        action="store_true",
        help="Use synthetic truth peak positions and hkl labels as solver inputs.",
    )
    parser.add_argument(
        "--oracle-stoichiometry",
        action="store_true",
        help="Prepend exact reference stoichiometry hypotheses to the search.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = (
        load_structure_specs(args.manifest)
        if args.manifest
        else default_structure_specs(args.structures_dir)
    )
    if args.structure_limit is not None:
        specs = specs[: max(0, args.structure_limit)]
    cfg = SyntheticRefinementConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        simulations_per_structure=max(1, args.simulations_per_structure),
        hkl_extent=args.hkl_extent,
        detector_shape=_parse_shape(args.detector_shape),
        qxy_range=_parse_range(args.qxy_range),
        qz_range=_parse_range(args.qz_range),
        peak_max_peaks=args.peak_max_peaks,
        candidate_max_candidates=args.candidate_max,
        max_generated_cifs_to_compare=args.max_generated_cifs,
        max_scaffolds_to_validate=args.max_scaffolds,
        bragg_intensity_weight=args.bragg_intensity_weight,
        bragg_intensity_tolerance=args.bragg_intensity_tolerance,
        bragg_intensity_max_peaks=args.bragg_intensity_max_peaks,
        generate_organic_electron_proxies=not args.no_organic_proxies,
        max_organic_proxy_cifs_to_compare=args.max_organic_proxies,
        generate_organic_replacement_structures=(
            not args.no_organic_replacements
        ),
        max_organic_replacement_cifs_to_compare=(
            args.max_organic_replacements
        ),
        stage_simulation_max_cifs=args.stage_simulation_max_cifs,
        organic_rmc_steps=args.organic_rmc_steps,
        organic_rmc_translation_step=args.organic_rmc_translation_step,
        organic_rmc_rotation_step_deg=args.organic_rmc_rotation_step_deg,
        assume_unit_cell_symmetry=not args.no_unit_cell_symmetry,
        rank_generated_cifs_with_image_fit=args.image_rerank,
        texture_modes=_parse_texture_modes(args.texture_modes),
        texture_azimuth_jitter_deg=args.texture_azimuth_jitter_deg,
        oracle_lattice_parameters=(
            args.oracle_diagnostic or args.oracle_lattice
        ),
        oracle_peak_positions=args.oracle_diagnostic or args.oracle_peaks,
        oracle_stoichiometry_constraints=(
            args.oracle_diagnostic or args.oracle_stoichiometry
        ),
    )
    result = run_synthetic_refinement_pipeline(specs, cfg)
    print(f"Synthetic refinement run: {result.run_id}")
    print(f"Output directory: {result.output_dir}")
    print(f"Filesets solved: {len(result.filesets)}")
    return 0


def _parse_shape(value: str) -> tuple[int, int]:
    pieces = value.lower().replace(",", "x").split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Shape must look like 128x192.")
    return int(pieces[0]), int(pieces[1])


def _parse_range(value: str) -> tuple[float, float]:
    pieces = value.replace(":", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Range must look like min,max.")
    return float(pieces[0]), float(pieces[1])


def _parse_texture_modes(value: str) -> tuple[str, ...]:
    modes = tuple(
        piece.strip()
        for piece in str(value).replace(";", ",").split(",")
        if piece.strip()
    )
    return modes or ("out_of_plane_stack", "in_plane_stack")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
