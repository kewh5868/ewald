"""Run headless EWALD structure benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ewald.benchmark import (
    BenchmarkRunConfig,
    default_structure_specs,
    load_structure_specs,
    run_structure_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ewald-benchmark",
        description=(
            "Generate mock GIWAXS measurements, run EWALD's headless "
            "structure workflow, and write projects, CIFs, plots, and a "
            "logbook."
        ),
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("example/structures"),
        help="Directory containing the default benchmark CIF files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON manifest overriding the default structure list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example/projects/structure_benchmark"),
        help="Directory for benchmark run outputs.",
    )
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--simulations-per-structure", type=int, default=1)
    parser.add_argument("--hkl-extent", type=int, default=7)
    parser.add_argument(
        "--detector-shape",
        default="224x320",
        help="Detector shape as rows x columns, for example 224x320.",
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
    parser.add_argument(
        "--structure-limit",
        type=int,
        help="Optional number of structures to run, useful for smoke tests.",
    )
    parser.add_argument(
        "--lattice-disorder-fraction",
        type=float,
        default=0.0,
        help="Gaussian fractional cell-length disorder for synthetic data.",
    )
    parser.add_argument(
        "--peak-dropout-fraction",
        type=float,
        default=0.0,
        help="Fraction of simulated truth peaks to suppress before solving.",
    )
    parser.add_argument(
        "--detector-gap-qxy-width",
        type=float,
        default=0.0,
        help="Width of a synthetic qxy detector gap centered at qxy=0.",
    )
    parser.add_argument(
        "--detector-gap-qz-width",
        type=float,
        default=0.0,
        help="Width of a synthetic qz detector gap centered in the mapped range.",
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
    config = BenchmarkRunConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        simulations_per_structure=max(1, args.simulations_per_structure),
        hkl_extent=args.hkl_extent,
        detector_shape=_parse_shape(args.detector_shape),
        qxy_range=_parse_range(args.qxy_range),
        qz_range=_parse_range(args.qz_range),
        synthetic_lattice_disorder_fraction=args.lattice_disorder_fraction,
        synthetic_peak_dropout_fraction=args.peak_dropout_fraction,
        synthetic_detector_gap_qxy_width=args.detector_gap_qxy_width,
        synthetic_detector_gap_qz_width=args.detector_gap_qz_width,
    )
    result = run_structure_benchmark(specs, config)
    print(f"Benchmark run: {result.run_id}")
    print(f"Output directory: {result.output_dir}")
    print(f"Filesets solved: {len(result.filesets)}")
    return 0


def _parse_shape(value: str) -> tuple[int, int]:
    pieces = value.lower().replace(",", "x").split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Shape must look like 224x320.")
    return int(pieces[0]), int(pieces[1])


def _parse_range(value: str) -> tuple[float, float]:
    pieces = value.replace(":", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Range must look like min,max.")
    return float(pieces[0]), float(pieces[1])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
