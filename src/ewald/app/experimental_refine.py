"""Run headless EWALD refinement against a real detector image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ewald.benchmark import (
    ExperimentalRefinementConfig,
    run_experimental_refinement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ewald-experimental-refine",
        description=(
            "Map a TIFF/PONI/mask dataset to q-space, solve chemically "
            "constrained structure candidates, refine scaffold and molecule "
            "positions, and validate against a held-out CIF."
        ),
    )
    parser.add_argument("--reference-cif", type=Path, required=True)
    parser.add_argument("--detector-image", type=Path, required=True)
    parser.add_argument("--poni", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument(
        "--inorganic",
        required=True,
        help="Comma-separated inorganic atom labels, for example Pb,I.",
    )
    parser.add_argument(
        "--organic",
        default="",
        help="Comma-separated organic molecule labels, for example MA,DMF.",
    )
    parser.add_argument(
        "--label",
        help="Optional fileset label for output folders and project names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example/projects/experimental_refinement"),
    )
    parser.add_argument(
        "--qspace-shape",
        default="180x220",
        help="Mapped q-space shape as rows x columns.",
    )
    parser.add_argument(
        "--qspace-ip-range",
        help="Optional qIP range as min,max in A^-1.",
    )
    parser.add_argument(
        "--qspace-oop-range",
        help="Optional qOOP range as min,max in A^-1.",
    )
    parser.add_argument("--xray-energy-kev", type=float)
    parser.add_argument("--incident-angle-deg", type=float, default=0.3)
    parser.add_argument("--sample-orientation", type=int, default=4)
    parser.add_argument("--hkl-extent", type=int, default=7)
    parser.add_argument("--max-generated-cifs", type=int, default=12)
    parser.add_argument("--max-hypotheses", type=int, default=12)
    parser.add_argument(
        "--bragg-intensity-weight",
        type=float,
        default=0.35,
        help=(
            "Weight for relative Bragg peak intensity matching in generated "
            "CIF ranking. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--bragg-intensity-tolerance",
        type=float,
        default=0.08,
        help="q-space tolerance for matching detected peaks to simulated Bragg peaks.",
    )
    parser.add_argument(
        "--bragg-intensity-max-peaks",
        type=int,
        default=80,
        help="Maximum strongest detected peaks used for Bragg intensity matching.",
    )
    parser.add_argument(
        "--no-unit-cell-symmetry",
        action="store_true",
        help=(
            "Use unconstrained deterministic atom/molecule placement instead "
            "of symmetry-constrained unit-cell placement."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ExperimentalRefinementConfig(
        output_dir=args.output_dir,
        qspace_shape=_parse_shape(args.qspace_shape),
        qspace_ip_range=(
            _parse_range(args.qspace_ip_range)
            if args.qspace_ip_range
            else None
        ),
        qspace_oop_range=(
            _parse_range(args.qspace_oop_range)
            if args.qspace_oop_range
            else None
        ),
        xray_energy_kev=args.xray_energy_kev,
        incident_angle_deg=args.incident_angle_deg,
        sample_orientation=args.sample_orientation,
        hkl_extent=args.hkl_extent,
        max_generated_cifs_to_compare=args.max_generated_cifs,
        max_stoichiometry_hypotheses=args.max_hypotheses,
        bragg_intensity_weight=args.bragg_intensity_weight,
        bragg_intensity_tolerance=args.bragg_intensity_tolerance,
        bragg_intensity_max_peaks=args.bragg_intensity_max_peaks,
        assume_unit_cell_symmetry=not args.no_unit_cell_symmetry,
    )
    result = run_experimental_refinement(
        reference_cif=args.reference_cif,
        detector_image=args.detector_image,
        poni_file=args.poni,
        mask_file=args.mask,
        inorganic_atoms=_parse_list(args.inorganic),
        organic_molecules=_parse_list(args.organic),
        label=args.label,
        config=cfg,
    )
    print(f"Experimental refinement run: {result.run_id}")
    print(f"Output directory: {result.output_dir}")
    print(f"Project: {result.fileset.get('project')}")
    print(f"Best generated CIF: {result.fileset.get('best_generated_cif')}")
    return 0


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_shape(value: str) -> tuple[int, int]:
    pieces = value.lower().replace(",", "x").split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Shape must look like 180x220.")
    return int(pieces[0]), int(pieces[1])


def _parse_range(value: str) -> tuple[float, float]:
    pieces = value.replace(":", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Range must look like min,max.")
    return float(pieces[0]), float(pieces[1])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
