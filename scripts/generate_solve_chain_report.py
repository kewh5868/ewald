#!/usr/bin/env python
# flake8: noqa: E402
"""Generate the synthetic-refinement solve-chain roadmap report.

The report is intentionally regenerable: the same run writes a PDF, Markdown,
JSON manifest, rendered figures, and a small HybriD3/MatD3 cache record under
``example/projects/synthetic_refinement_history/reports``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import textwrap
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/ewald_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ewald_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from pymatgen.core import Lattice, Structure

HALIDES = {"F", "Cl", "Br", "I"}
B_SITE = {"Pb", "Sn"}
A_SITE = {"Cs", "Rb", "K", "Na"}
INORGANIC = HALIDES | B_SITE | A_SITE
THEME = {
    "ink": "#182635",
    "muted": "#5f6f7d",
    "navy": "#18344a",
    "blue": "#2c6f8f",
    "blue_light": "#e7f2f6",
    "gold": "#c49135",
    "gold_light": "#f7efe0",
    "green": "#47705a",
    "green_light": "#eaf3ed",
    "paper": "#f6f8fa",
    "white": "#ffffff",
    "rule": "#ccd6dd",
}
REPORT_FOOTER = "EWALD synthetic refinement | complete solve-chain plan"
PAGE_COUNTER = 0

HYBRID3_SOURCE_URLS = {
    "HybriD3 materials database": (
        "https://hybrid3.duke.edu/research/materials-database"
    ),
    "HybriD3 database website documentation": (
        "https://hybrid3-database.readthedocs.io/en/latest/website.html"
    ),
    "HybriD3 database webinar": (
        "https://hybrid3.duke.edu/workshops/2026-hybrid3-database-webinar"
    ),
    "MatD3 live REST API base": "https://materials.hybrid3.duke.edu/",
    "Bi(III) polyhalide motif review": (
        "https://www.sciencedirect.com/science/article/abs/pii/S0010854515300254"
    ),
    "Pb(II) 6s2 lone-pair coordination review": (
        "https://www.sciencedirect.com/science/article/pii/S0010854525000049"
    ),
    "Metastable low-dimensional halometallates": (
        "https://par.nsf.gov/servlets/purl/10329300"
    ),
}


@dataclass(frozen=True)
class MotifRecord:
    label: str
    formula: str
    dimensionality: str
    sharing: str
    graph_hint: str
    charge_logic: str
    solver_use: str


@dataclass(frozen=True)
class MotifExampleRecord:
    motif_id: str
    label: str
    formula: str
    dimensionality: str
    sharing: str
    source: str
    source_path: str
    generated_cif: str
    notes: str


MOTIF_LIBRARY = [
    MotifRecord(
        label="PbI2 layer",
        formula="PbI2",
        dimensionality="2D",
        sharing="edge-sharing sheet or layered parent",
        graph_hint="sheet",
        charge_logic="Pb2+ balanced by two I- per reduced B site",
        solver_use="Layered inorganic baseline and solvate-expanded parent.",
    ),
    MotifRecord(
        label="hexagonal PbI2",
        formula="2H-PbI2",
        dimensionality="2D",
        sharing="CdI2-type edge-sharing octahedral layer",
        graph_hint="hex_sheet",
        charge_logic="neutral PbI2 slab; organic/solvate mostly spacing",
        solver_use="Reference for edge-sharing sheets and strong in-plane peaks.",
    ),
    MotifRecord(
        label="Pb2I6 ribbon",
        formula="Pb2I6",
        dimensionality="1D",
        sharing="two-Pb-wide edge-sharing octahedral ribbon",
        graph_hint="dimer",
        charge_logic="2 Pb2+ and 6 I- leaves 2- before organic cations",
        solver_use="Tarasov Pb2I6 reduced ribbon reference.",
    ),
    MotifRecord(
        label="Pb3I8 strip",
        formula="Pb3I8",
        dimensionality="1D/2D",
        sharing="mixed edge/corner-sharing trimers or strips",
        graph_hint="trimer_strip",
        charge_logic="3 Pb2+ and 8 I- leaves 2- before organic cations",
        solver_use="Primary Tarasov reduced scaffold target.",
    ),
    MotifRecord(
        label="Pb3I10 ribbon",
        formula="Pb3I10",
        dimensionality="0D/1D",
        sharing="lower-connectivity octahedral fragment",
        graph_hint="open_trimer",
        charge_logic="3 Pb2+ and 10 I- leaves 4- before cations",
        solver_use="Candidate for cation-rich low-dimensional derivatives.",
    ),
    MotifRecord(
        label="Pb3I11 cluster",
        formula="Pb3I11",
        dimensionality="0D",
        sharing="weakly connected terminal-rich cluster",
        graph_hint="cluster",
        charge_logic="3 Pb2+ and 11 I- leaves 5- before cations",
        solver_use="Stress-test motif for terminal halide-rich salts.",
    ),
    MotifRecord(
        label="PbI5 pyramid",
        formula="PbI5",
        dimensionality="1D",
        sharing="corner-sharing octahedral pair/chain stoichiometry",
        graph_hint="pyramid",
        charge_logic="Pb2+ and 5 I- leaves 3- before cations",
        solver_use="Tarasov PbI5 reference for corner-sharing low-dimensional motifs.",
    ),
    MotifRecord(
        label="PbI6 to PbI3",
        formula="PbI6 / PbI3",
        dimensionality="0D/3D",
        sharing="isolated octahedron or all-corner-sharing perovskite net",
        graph_hint="octahedron",
        charge_logic="isolated PbI6 is 4-; full corner sharing gives PbI3-",
        solver_use="Primitive octahedral motif for 0D clusters and 3D perovskites.",
    ),
]


REPORT_SECTIONS = {
    "summary": [
        "This report defines a deterministic, physics-first solve chain for "
        "hybrid lead/tin halide perovskites, perovskitoids, and solvate "
        "derivatives. The main change is to solve a reduced inorganic motif "
        "first, estimate the Z multiplier separately, and place organic "
        "matter initially as charge-balanced electron-density proxies.",
        "The worked example is (MA)2(DMF)2Pb3I8 from the local Tarasov CIF. "
        "The reduced target is Pb3I8, not the full organic cell. This gives "
        "the solver a chemically plausible construction basis before full "
        "organic replacement and RMC refinement.",
    ],
    "program_description": [
        "Program purpose: EWALD should convert calibrated GIWAXS/WAXS evidence "
        "and known chemistry into testable CIF hypotheses. The solver is not "
        "a black-box structure predictor; it is a staged inference program "
        "that records what information was used, what assumptions were made, "
        "and where the reconstruction failed.",
        "Primary inputs are calibrated 2D diffraction images or synthetic "
        "images, reference CIFs for benchmarks, detector/q-space metadata, "
        "texture assumptions, and a motif knowledge base. Primary outputs are "
        "peak tables with integrated intensities, hkl/family assignments, "
        "lattice candidates, reduced-motif hypotheses, scaffold CIFs, organic "
        "proxy CIFs, organic replacement CIFs, simulated comparison images, "
        "and JSON/Markdown diagnostics.",
        "Program modules should remain separable: peak/ROI analysis, family "
        "and hkl inference, evidence auditing, motif/Z selection, Wyckoff and "
        "site generation, chemistry/steric filtering, GIWAXS simulation, "
        "ranking/minimization, organic replacement, and benchmark reporting. "
        "This separation is what lets oracle runs identify whether the "
        "failure is information recovery or independent coordinate generation.",
        "The report artifacts under synthetic_refinement_history are part of "
        "the program design: they document solver assumptions, record motif "
        "examples, expose HybriD3/MatD3 reference metadata, and give future "
        "solver variants a concrete checklist for improvement.",
    ],
    "peak_chain": [
        "The peak chain starts with ROI detection, sub-pixel center fitting, "
        "integrated intensity, and uncertainty estimates. Benchmark mode must "
        "compare each detected ROI against synthetic truth peaks, including "
        "precision, recall, center error, intensity-weighted recall, and "
        "bias in qxy/qz.",
        "Family grouping should use q-ratios, candidate hkl consistency, "
        "symmetry support, relative Bragg intensity, texture mode, and "
        "uncertainty. Families that agree in d-spacing but disagree in "
        "texture or intensity should remain splittable.",
    ],
    "information_evaluation": [
        "The solver should separate evidence channels: lattice evidence, hkl "
        "evidence, motif evidence, stoichiometry evidence, intensity evidence, "
        "texture evidence, and steric/void evidence. Each stage should report "
        "a confidence and a failure attribution instead of only a final score.",
        "Oracle runs are the upper-bound diagnostic. If oracle lattice, hkl, "
        "peak families, and stoichiometry are perfect but the scaffold RMS is "
        "still poor, the failure is coordinate construction, motif selection, "
        "Wyckoff assignment, or refinement scoring.",
    ],
    "reduced_unit": [
        "Reduced-unit construction avoids atom-count runaway. The solver first "
        "selects a plausible inorganic motif graph such as Pb3I8, then "
        "separately infers Z from lattice volume, family multiplicity, charge "
        "balance, systematic absences, and intensity scale.",
        "Full-cell stoichiometry is used as validation and replication target, "
        "not as the first coordinate-generation target. This prevents a large "
        "organic-expanded cell from tricking the inorganic scaffold builder "
        "into huge Pb/I atom counts.",
    ],
    "motifs": [
        "The motif library should encode B-X graph templates for Pb/Sn halide "
        "chemistry across 0D, 1D, 2D, and 3D structures. Templates include "
        "isolated octahedra, dimers, strips, ribbons, sheets, corner-sharing "
        "networks, edge-sharing networks, face-sharing chains, and mixed "
        "perovskite-solvate derivatives.",
        "Ranking should combine B:X ratio, charge balance, known "
        "perovskite/perovskitoid chemistry, lattice anisotropy, Wyckoff "
        "multiplicity, texture-compatible families, Bragg intensity agreement, "
        "and steric compatibility with organic/solvent cavities.",
        "Stoichiometry follows sharing topology. An isolated PbI6 octahedron "
        "is a 0D building block, but a 3D all-corner-sharing PbI6 network is "
        "stoichiometric PbI3 because each iodide is shared by two octahedra. "
        "Corner-sharing chains tend toward PbI5, single-layer corner-sharing "
        "slabs toward PbI4, edge-sharing ribbons can give Pb2I6 or Pb3I8, "
        "and hexagonally packed CdI2-type sheets give neutral PbI2 layers.",
        "Bi(III) and Pb(II) both present a 6s2 lone-pair electronic motif, so "
        "bismuth and antimony polyhalide literature is valuable as an analog "
        "library for corner-, edge-, and face-sharing halometallate graphs. "
        "The solver should treat those examples as topology priors, then "
        "re-score them with Pb/Sn charge balance and radius/steric filters.",
    ],
    "wyckoff": [
        "Wyckoff/site guessing should map the reduced motif onto candidate "
        "space groups and multiplicities. The reduced motif defines how many "
        "unique B and X sites are needed; Z determines replication; symmetry "
        "then constrains where equivalent atoms and organic proxy centers can "
        "sit.",
        "The solver should prefer special positions only when they satisfy "
        "coordination and intensity evidence. General positions should be "
        "allowed when lower symmetry is the physically reasonable explanation.",
    ],
    "organic": [
        "Organic matter enters in three stages. First, infer the number and "
        "charge of organic cations or neutral solvents from charge balance and "
        "known formulas. Second, place electron-count proxy centers in voids. "
        "For example, a roughly 20-electron molecule can be represented by a "
        "Ca-like proxy for initial scattering-center placement, while its "
        "steric radius remains molecule-like rather than calcium-like.",
        "Third, replace proxies with molecular bodies using symmetry, sterics, "
        "hydrogen-bond donor/acceptor geometry, cavity fit, and contacts to "
        "terminal halides. Final organic refinement is bounded RMC: rotations "
        "and subtle translations only, with molecule integrity preserved.",
    ],
    "constraints": [
        "Physical filters should enforce Pauling-style sanity checks: Pb2+ or "
        "Sn2+ should prefer halide-rich coordination near octahedral geometry "
        "unless a motif template explicitly encodes a lower-coordination "
        "fragment. Penalize impossible B-X, X-X, B-B, organic-inorganic, and "
        "organic-organic contacts.",
        "Charge balance should be hard or near-hard at the reduced motif level. "
        "Steric checks should use covalent/ionic radii plus molecule envelopes; "
        "hydrogen bonding should reward plausible N-H...I/Br/Cl and O-H...X "
        "geometries when donor/acceptor labels are available.",
    ],
    "simulation_schedule": [
        "Resolution should increase by stage. Coarse runs screen lattice and "
        "motif cheaply. Medium runs rank scaffold and proxy placements using "
        "peak-level residuals and relative intensities. High-resolution runs "
        "are reserved for final organic/RMC image residuals.",
        "Every stage should retain peak-level metrics: matched fraction, "
        "center residual, integrated intensity residual, family residual, "
        "image correlation, weighted RMSE, and residual-map localization.",
    ],
    "training": [
        "The base solver should remain deterministic and physics-first. "
        "Synthetic runs and HybriD3/MatD3 references can produce labeled "
        "datasets for peak precision/recall, family purity, dimensionality, "
        "corner/edge/face-sharing motif class, reduced motif, Z, and scaffold "
        "ranking versus final RMS.",
        "Start with classical ranking models and calibration curves over "
        "engineered features. Only after ablations show value should neural "
        "models propose priors. Learned priors should never silently override "
        "charge balance, symmetry, or steric constraints.",
    ],
    "example_case": [
        "Example test case: start from the Tarasov (MA)2(DMF)2Pb3I8 CIF and "
        "generate a textured synthetic target. In oracle mode, pass exact "
        "lattice parameters, exact peak positions, exact hkl labels/families, "
        "and the reduced Pb3I8 stoichiometry to the construction engine while "
        "withholding atom coordinates.",
        "Expected diagnostic path: peak and family metrics should be perfect; "
        "lattice error should be zero; the independent scaffold builder should "
        "construct a Pb3I8 motif with plausible Pb-I coordination; Z should "
        "replicate that motif to the full cell; organic proxies should satisfy "
        "the remaining charge and void evidence; final organic replacement "
        "should reduce the image residual without introducing steric clashes.",
        "Failure attribution: if the oracle run still has high scaffold RMS, "
        "the problem is not peak detection. Improve motif graphs, Wyckoff "
        "multiplicity choices, charge/steric filters, and Bragg-intensity "
        "placement before retuning the peak finder.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the EWALD solve-chain roadmap report."
    )
    parser.add_argument(
        "--example-cif",
        type=Path,
        default=Path("example/structures/(MA)2(DMF)2Pb3I8_2017Tarasov.cif"),
        help="Worked-example CIF for report figures.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("example/projects/synthetic_refinement_history/reports"),
        help="Directory where report artifacts are written.",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().strftime("%Y%m%d"),
        help="Report date stamp used in filenames.",
    )
    parser.add_argument(
        "--hybrid3-timeout",
        type=float,
        default=5.0,
        help="Seconds per HybriD3/MatD3 API probe.",
    )
    parser.add_argument(
        "--skip-hybrid3-api",
        action="store_true",
        help="Write seeded HybriD3 references without API probing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    example_cif = (repo_root / args.example_cif).resolve()
    output_root = (repo_root / args.output_root).resolve()
    report_dir = output_root
    assets_dir = report_dir / "assets" / args.date
    cache_dir = report_dir / "hybrid3_cache" / args.date
    motif_dir = report_dir / "motif_subunits" / args.date
    assets_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    motif_dir.mkdir(parents=True, exist_ok=True)

    structure = Structure.from_file(str(example_cif))
    structure_summary = summarize_structure(structure, example_cif)
    motif_examples = build_motif_subunit_library(repo_root, motif_dir)
    hybrid3_cache = build_hybrid3_cache(
        cache_dir,
        timeout=args.hybrid3_timeout,
        skip_api=args.skip_hybrid3_api,
    )
    recent_runs = inspect_recent_runs(repo_root)

    figures = generate_figures(
        structure=structure,
        structure_summary=structure_summary,
        example_cif=example_cif,
        assets_dir=assets_dir,
        motif_examples=motif_examples,
    )
    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": args.date,
        "example_cif": str(example_cif),
        "outputs": {},
        "structure_summary": structure_summary,
        "motif_library": [asdict(record) for record in MOTIF_LIBRARY],
        "motif_examples": [asdict(record) for record in motif_examples],
        "hybrid3_cache": hybrid3_cache,
        "recent_runs": recent_runs,
        "figures": {key: str(path) for key, path in figures.items()},
        "algorithm_sections": REPORT_SECTIONS,
        "source_urls": HYBRID3_SOURCE_URLS,
    }

    md_path = report_dir / f"solve_chain_plan_{args.date}.md"
    json_path = report_dir / f"solve_chain_plan_{args.date}.json"
    pdf_path = report_dir / f"solve_chain_plan_{args.date}.pdf"
    write_markdown(md_path, manifest, figures)
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_pdf(pdf_path, manifest, figures)
    manifest["outputs"] = {
        "pdf": str(pdf_path),
        "markdown": str(md_path),
        "json": str(json_path),
        "assets_dir": str(assets_dir),
        "hybrid3_cache_dir": str(cache_dir),
        "motif_subunits_dir": str(motif_dir),
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote PDF: {pdf_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote assets: {assets_dir}")


def summarize_structure(
    structure: Structure, cif_path: Path
) -> dict[str, Any]:
    composition = structure.composition
    inorganic_counts: dict[str, float] = {}
    organic_counts: dict[str, float] = {}
    for element, amount in composition.element_composition.items():
        symbol = element.symbol
        target = inorganic_counts if symbol in INORGANIC else organic_counts
        target[symbol] = float(amount)

    b_count = sum(inorganic_counts.get(symbol, 0.0) for symbol in B_SITE)
    halide_count = sum(inorganic_counts.get(symbol, 0.0) for symbol in HALIDES)
    reduced = reduce_formula_counts(
        {
            symbol: int(round(amount))
            for symbol, amount in inorganic_counts.items()
            if symbol in B_SITE | HALIDES and amount > 0
        }
    )
    charge = 2.0 * b_count - halide_count
    abc = tuple(float(value) for value in structure.lattice.abc)
    angles = tuple(float(value) for value in structure.lattice.angles)
    density_proxy = float(
        len(structure) / max(structure.lattice.volume, 1.0e-9)
    )
    return {
        "path": str(cif_path),
        "formula": composition.reduced_formula,
        "full_formula": composition.formula,
        "site_count": len(structure),
        "lattice_abc": abc,
        "lattice_angles": angles,
        "lattice_volume": float(structure.lattice.volume),
        "inorganic_counts": inorganic_counts,
        "organic_or_light_counts": organic_counts,
        "inorganic_reduced_formula": formula_from_counts(reduced),
        "inorganic_charge_before_organic": charge,
        "estimated_organic_countercharge": -charge,
        "site_density_per_angstrom3": density_proxy,
        "worked_example_target": "Pb3I8",
        "worked_example_logic": (
            "Build Pb3I8 as the reduced scaffold, infer Z separately, then "
            "explain residual charge and scattering with MA cation and DMF "
            "proxy centers before full organic replacement."
        ),
    }


def build_motif_subunit_library(
    repo_root: Path,
    motif_dir: Path,
) -> list[MotifExampleRecord]:
    """Create known-reference scaffold CIFs and idealized motif
    templates."""

    motif_dir.mkdir(parents=True, exist_ok=True)
    records: list[MotifExampleRecord] = []
    known_sources = [
        {
            "motif_id": "known_pb3i8_edge_strip_petrov",
            "label": "FA2Pb3I8-4DMF Petrov reference",
            "formula": "Pb3I8",
            "dimensionality": "1D/2D",
            "sharing": "three-Pb-wide edge-sharing octahedral subunits",
            "path": repo_root
            / "example/structures/FA2Pb3I8-4DMF-2022-Petrov.cif",
            "notes": "User-specified Pb3I8 example with three-Pb-wide edge-sharing units.",
        },
        {
            "motif_id": "known_pbi5_corner_pair_tarasov",
            "label": "(MA)3(DMF)PbI5 Tarasov reference",
            "formula": "PbI5",
            "dimensionality": "1D",
            "sharing": "pairs/chains of corner-sharing octahedra",
            "path": repo_root
            / "example/structures/(MA)3(DMF)PbI5_2017Tarasov.cif",
            "notes": "User-specified PbI5 stoichiometry example.",
        },
        {
            "motif_id": "known_pb2i6_edge_ribbon_tarasov",
            "label": "(MA)2(DMF)2Pb2I6 Tarasov reference",
            "formula": "Pb2I6",
            "dimensionality": "1D",
            "sharing": "two-Pb-wide edge-sharing octahedral ribbons",
            "path": repo_root
            / "example/structures/(MA)2(DMF)2Pb2I6_2017Tarasov.cif",
            "notes": "User-specified Pb2I6 ribbon example.",
        },
        {
            "motif_id": "known_2h_pbi2_hex_sheet_desktop",
            "label": "2H-PbI2 hexagonal sheet reference",
            "formula": "PbI2",
            "dimensionality": "2D",
            "sharing": "hexagonally packed edge-sharing PbI2 sheets",
            "path": Path(
                "/Users/keithwhite/Desktop/CIFs/cif_perovskite-precursors/PbI2.cif"
            ),
            "notes": "User-specified 2H-PbI2 precursor CIF.",
        },
    ]

    for item in known_sources:
        source_path = Path(item["path"])
        output = motif_dir / f"{item['motif_id']}_inorganic_scaffold.cif"
        if source_path.exists():
            try:
                write_inorganic_scaffold(source_path, output)
                generated = str(output)
                notes = str(item["notes"])
            except Exception as exc:
                generated = ""
                notes = f"{item['notes']} Extraction failed: {exc!r}"
        else:
            generated = ""
            notes = f"{item['notes']} Source path was not found."
        records.append(
            MotifExampleRecord(
                motif_id=str(item["motif_id"]),
                label=str(item["label"]),
                formula=str(item["formula"]),
                dimensionality=str(item["dimensionality"]),
                sharing=str(item["sharing"]),
                source="local_reference",
                source_path=str(source_path),
                generated_cif=generated,
                notes=notes,
            )
        )

    records.extend(write_idealized_motif_cifs(motif_dir))
    manifest_path = motif_dir / "motif_subunits_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )
    return records


def write_inorganic_scaffold(source_path: Path, output_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = Structure.from_file(str(source_path))
    species = []
    frac_coords = []
    for site in structure:
        symbol = site.specie.symbol
        if symbol in B_SITE | HALIDES | A_SITE:
            species.append(symbol)
            frac_coords.append(site.frac_coords)
    scaffold = Structure(structure.lattice, species, frac_coords)
    scaffold.to(filename=str(output_path))


def write_idealized_motif_cifs(motif_dir: Path) -> list[MotifExampleRecord]:
    examples: list[tuple[str, str, str, str, str, str, Structure]] = []
    examples.append(
        (
            "ideal_pb_i6_isolated_octahedron",
            "Ideal isolated PbI6 octahedron",
            "PbI6",
            "0D",
            "no shared ligands",
            "0D coordination sanity-check template.",
            Structure(
                Lattice.cubic(12.0),
                ["Pb", "I", "I", "I", "I", "I", "I"],
                [
                    (0.5, 0.5, 0.5),
                    (0.68, 0.5, 0.5),
                    (0.32, 0.5, 0.5),
                    (0.5, 0.68, 0.5),
                    (0.5, 0.32, 0.5),
                    (0.5, 0.5, 0.68),
                    (0.5, 0.5, 0.32),
                ],
            ),
        )
    )
    examples.append(
        (
            "ideal_pbi3_corner_sharing_3d",
            "Ideal PbI3 3D corner-sharing perovskite net",
            "PbI3",
            "3D",
            "all six octahedral corners shared",
            "ABX3-like perovskite scaffold; A site intentionally omitted.",
            Structure(
                Lattice.cubic(6.35),
                ["Pb", "I", "I", "I"],
                [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)],
            ),
        )
    )
    examples.append(
        (
            "ideal_pbi5_corner_chain",
            "Ideal PbI5 corner-sharing chain motif",
            "PbI5",
            "1D",
            "two axial corners shared along chain",
            "Corner-chain stoichiometry anchor for PbI5 salts.",
            Structure(
                Lattice.orthorhombic(6.2, 6.2, 6.4),
                ["Pb", "I", "I", "I", "I", "I"],
                [
                    (0.5, 0.5, 0.0),
                    (0.5, 0.5, 0.5),
                    (0.68, 0.5, 0.0),
                    (0.32, 0.5, 0.0),
                    (0.5, 0.68, 0.0),
                    (0.5, 0.32, 0.0),
                ],
            ),
        )
    )
    examples.append(
        (
            "ideal_pb2i6_edge_ribbon",
            "Ideal Pb2I6 edge-sharing ribbon motif",
            "Pb2I6",
            "1D",
            "two-Pb-wide edge-sharing ribbon",
            "Reduced ribbon basis matching the Tarasov Pb2I6 reference class.",
            Structure(
                Lattice.orthorhombic(7.2, 6.8, 6.5),
                ["Pb", "Pb", "I", "I", "I", "I", "I", "I"],
                [
                    (0.34, 0.5, 0.0),
                    (0.66, 0.5, 0.0),
                    (0.50, 0.38, 0.0),
                    (0.50, 0.62, 0.0),
                    (0.18, 0.5, 0.0),
                    (0.82, 0.5, 0.0),
                    (0.34, 0.5, 0.5),
                    (0.66, 0.5, 0.5),
                ],
            ),
        )
    )
    examples.append(
        (
            "ideal_pb3i8_edge_strip",
            "Ideal Pb3I8 edge-sharing strip motif",
            "Pb3I8",
            "1D/2D",
            "three-Pb-wide edge-sharing strip",
            "Reduced strip basis for the Petrov and Tarasov Pb3I8 class.",
            Structure(
                Lattice.orthorhombic(9.2, 7.0, 6.5),
                ["Pb", "Pb", "Pb", "I", "I", "I", "I", "I", "I", "I", "I"],
                [
                    (0.25, 0.5, 0.0),
                    (0.50, 0.5, 0.0),
                    (0.75, 0.5, 0.0),
                    (0.375, 0.38, 0.0),
                    (0.375, 0.62, 0.0),
                    (0.625, 0.38, 0.0),
                    (0.625, 0.62, 0.0),
                    (0.12, 0.5, 0.0),
                    (0.88, 0.5, 0.0),
                    (0.33, 0.5, 0.5),
                    (0.67, 0.5, 0.5),
                ],
            ),
        )
    )
    examples.append(
        (
            "ideal_2h_pbi2_hex_sheet",
            "Ideal 2H-PbI2 hexagonal sheet motif",
            "PbI2",
            "2D",
            "CdI2-type edge-sharing hexagonal sheet",
            "Neutral precursor layer useful for PbI2 solvate derivatives.",
            Structure(
                Lattice.hexagonal(4.56, 8.0),
                ["Pb", "I", "I"],
                [(0, 0, 0.5), (1 / 3, 2 / 3, 0.62), (2 / 3, 1 / 3, 0.38)],
            ),
        )
    )
    examples.append(
        (
            "ideal_pbi3_face_sharing_chain",
            "Ideal PbI3 face-sharing chain motif",
            "PbI3",
            "1D",
            "opposite triangular faces shared along chain",
            "High-connectivity chain prior; should be strongly distance-filtered.",
            Structure(
                Lattice.hexagonal(6.4, 4.1),
                ["Pb", "I", "I", "I"],
                [
                    (0, 0, 0),
                    (0.34, 0, 0.25),
                    (0, 0.34, 0.25),
                    (0.66, 0.66, 0.25),
                ],
            ),
        )
    )
    examples.append(
        (
            "ideal_pb3i10_terminal_ribbon",
            "Ideal Pb3I10 terminal-rich ribbon",
            "Pb3I10",
            "0D/1D",
            "lower-connectivity terminal-rich trimer",
            "Cation-rich motif prior for low-dimensional salts.",
            Structure(
                Lattice.orthorhombic(11.0, 8.0, 7.0),
                ["Pb", "Pb", "Pb"] + ["I"] * 10,
                [
                    (0.25, 0.5, 0.5),
                    (0.50, 0.5, 0.5),
                    (0.75, 0.5, 0.5),
                    (0.18, 0.5, 0.5),
                    (0.88, 0.5, 0.5),
                    (0.38, 0.40, 0.5),
                    (0.38, 0.60, 0.5),
                    (0.62, 0.40, 0.5),
                    (0.62, 0.60, 0.5),
                    (0.25, 0.5, 0.25),
                    (0.50, 0.5, 0.75),
                    (0.75, 0.5, 0.25),
                    (0.50, 0.25, 0.5),
                ],
            ),
        )
    )
    records: list[MotifExampleRecord] = []
    for motif_id, label, formula, dim, sharing, notes, structure in examples:
        output = motif_dir / f"{motif_id}.cif"
        structure.to(filename=str(output))
        records.append(
            MotifExampleRecord(
                motif_id=motif_id,
                label=label,
                formula=formula,
                dimensionality=dim,
                sharing=sharing,
                source="idealized_template",
                source_path="generated",
                generated_cif=str(output),
                notes=notes,
            )
        )
    return records


def reduce_formula_counts(counts: dict[str, int]) -> dict[str, int]:
    values = [abs(value) for value in counts.values() if value]
    if not values:
        return {}
    divisor = values[0]
    for value in values[1:]:
        divisor = math.gcd(divisor, value)
    divisor = max(divisor, 1)
    return {key: value // divisor for key, value in counts.items()}


def formula_from_counts(counts: dict[str, int]) -> str:
    order = ["Pb", "Sn", "I", "Br", "Cl", "F", "Cs", "Rb", "K", "Na"]
    parts: list[str] = []
    for symbol in order:
        count = counts.get(symbol)
        if not count:
            continue
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    for symbol in sorted(set(counts) - set(order)):
        count = counts[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def build_hybrid3_cache(
    cache_dir: Path,
    *,
    timeout: float,
    skip_api: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_records = [
        {
            "record_kind": "seed_reference",
            "dataset_id": "HybriD3-public-reference-seed",
            "system_id": "lead-halide-perovskite-derivatives",
            "reference_id": "hybrid3-materials-database-overview",
            "formula": "Pb/Sn-halide hybrids",
            "dimensionality": "0D/1D/2D/3D",
            "motif_tags": [
                "perovskite",
                "perovskitoid",
                "organic-inorganic-halide",
                "corner-edge-face-sharing",
            ],
            "source_label": "HybriD3 Materials Database",
            "url": HYBRID3_SOURCE_URLS["HybriD3 materials database"],
            "notes": (
                "Seed record used when live MatD3 API probing is unavailable. "
                "It records where reference structures should be pulled from "
                "for benchmark expansion."
            ),
        },
        {
            "record_kind": "seed_reference",
            "dataset_id": "MatD3-public-api-seed",
            "system_id": "hybrid-perovskite-systems",
            "reference_id": "hybrid3-database-docs",
            "formula": "(organic) Pb/Sn X_n",
            "dimensionality": "mixed",
            "motif_tags": ["HybriD3", "MatD3", "CIF", "reference-cache"],
            "source_label": "HybriD3 database website documentation",
            "url": HYBRID3_SOURCE_URLS[
                "HybriD3 database website documentation"
            ],
            "notes": (
                "Documentation-backed reference for database navigation and "
                "future API-backed CIF metadata ingestion."
            ),
        },
    ]
    api_attempts: list[dict[str, Any]] = []
    discovered_records: list[dict[str, Any]] = []
    if not skip_api:
        api_attempts, discovered_records = probe_matd3_api(timeout=timeout)
    cache = {
        "source_urls": HYBRID3_SOURCE_URLS,
        "api_attempts": api_attempts,
        "records": seed_records + discovered_records,
        "record_count": len(seed_records) + len(discovered_records),
        "cache_policy": (
            "Cache selected metadata and links only; do not mirror the full "
            "database. Each reference record should keep dataset/system/"
            "reference identifiers, formula, dimensionality, motif tags, "
            "source label, URL, and CIF/download link when available."
        ),
    }
    (cache_dir / "hybrid3_cache.json").write_text(
        json.dumps(cache, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return cache


def probe_matd3_api(
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - dependency guard.
        return (
            [
                {
                    "url": "requests-import",
                    "status": "unavailable",
                    "error": repr(exc),
                }
            ],
            [],
        )

    base_urls = ["https://materials.hybrid3.duke.edu"]
    endpoint_paths = [
        "/materials/datasets/?page=1&page_size=3",
        "/materials/systems/?page=1&page_size=3",
        "/materials/references/?page=1&page_size=3",
        "/materials/properties/?page=1&page_size=3",
        "/materials/units/?page=1&page_size=3",
    ]
    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for base in base_urls:
        for path in endpoint_paths:
            url = f"{base}{path}"
            attempt: dict[str, Any] = {"url": url}
            try:
                response = requests.get(url, timeout=timeout)
                attempt["status_code"] = response.status_code
                attempt["content_type"] = response.headers.get(
                    "content-type", ""
                )
                response.raise_for_status()
                try:
                    payload: Any = response.json()
                except ValueError:
                    payload = {"text_sample": response.text[:500]}
                attempt["payload_kind"] = type(payload).__name__
                attempt["record_count"] = len(extract_sequence(payload))
                records.extend(records_from_api_payload(url, payload))
            except Exception as exc:
                attempt["error"] = repr(exc)
            attempts.append(attempt)
    return attempts, records


def extract_sequence(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "objects"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def records_from_api_payload(url: str, payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(extract_sequence(payload)):
        if not isinstance(item, dict):
            continue
        system = (
            item.get("system") if isinstance(item.get("system"), dict) else {}
        )
        reference = (
            item.get("reference")
            if isinstance(item.get("reference"), dict)
            else {}
        )
        dataset_id = first_present(item, ("dataset_id", "dataset", "pk", "id"))
        system_id = first_present(system, ("system_id", "id"))
        reference_id = first_present(reference, ("reference_id", "id"))
        formula = first_known(
            first_present(item, ("formula", "pretty_formula", "name")),
            first_present(system, ("formula", "group", "compound_name")),
        )
        dimensionality = first_known(
            first_present(item, ("dimensionality", "dimension", "category")),
            first_present(system, ("dimensionality", "dimension", "category")),
        )
        tags = listify(first_present(item, ("tags", "keywords")))
        tags.extend(listify(first_present(system, ("tags", "keywords"))))
        inorganic = first_present(system, ("inorganic",))
        if inorganic != "unknown":
            tags.append(f"inorganic:{inorganic}")
        record = {
            "record_kind": "api_probe",
            "dataset_id": str(dataset_id),
            "system_id": str(system_id),
            "reference_id": str(reference_id),
            "formula": str(formula),
            "dimensionality": str(dimensionality),
            "motif_tags": sorted(set(tags)),
            "source_label": "MatD3 API probe",
            "url": url,
            "dataset_url": (
                f"https://materials.hybrid3.duke.edu/materials/dataset/{dataset_id}"
                if dataset_id != "unknown"
                else "unknown"
            ),
            "system_url": (
                f"https://materials.hybrid3.duke.edu/materials/{system_id}"
                if system_id != "unknown"
                else "unknown"
            ),
            "reference_title": str(first_present(reference, ("title",))),
            "cif_url": str(
                first_present(
                    item,
                    ("cif_url", "download_url", "structure_url", "url"),
                )
            ),
            "api_index": index,
        }
        records.append(record)
    return records


def first_known(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", "unknown"):
            return value
    return "unknown"


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return "unknown"


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value != "unknown":
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def inspect_recent_runs(repo_root: Path) -> dict[str, Any]:
    history = (
        repo_root
        / "example/projects/synthetic_refinement_history/20260519/runs"
    )
    runs: dict[str, Any] = {}
    for kind in ("staged", "oracle_diagnostic", "oracle_smoke"):
        parent = history / kind
        summaries = sorted(parent.glob("run_*/summary.json"))
        if not summaries:
            continue
        latest = summaries[-1]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except Exception as exc:
            runs[kind] = {"path": str(latest), "error": repr(exc)}
            continue
        aggregate = payload.get("aggregate", {})
        runs[kind] = {
            "path": str(latest.parent),
            "aggregate": aggregate,
            "fileset_count": len(payload.get("filesets", [])),
            "interpretation": interpret_run_metrics(kind, aggregate),
        }
    return runs


def interpret_run_metrics(kind: str, aggregate: dict[str, Any]) -> str:
    if kind == "oracle_diagnostic":
        return (
            "Oracle diagnostics separate construction failures from peak/hkl "
            "information loss. Perfect oracle peak metrics with high scaffold "
            "RMS means the coordinate builder needs motif/Wyckoff/refinement "
            "work."
        )
    if not aggregate:
        return "No aggregate metrics were available."
    recall = aggregate.get("mean_peak_recall")
    family = aggregate.get("mean_peak_family_weighted_purity")
    return (
        f"Staged run snapshot: peak recall={recall}, family purity={family}. "
        "Use this to decide whether the next failure is detection, family "
        "grouping, or construction."
    )


def generate_figures(
    *,
    structure: Structure,
    structure_summary: dict[str, Any],
    example_cif: Path,
    assets_dir: Path,
    motif_examples: list[MotifExampleRecord],
) -> dict[str, Path]:
    figures = {
        "solve_chain_flow": assets_dir / "solve_chain_flow.png",
        "peak_family_schematic": assets_dir / "peak_family_schematic.png",
        "reduced_motif_z": assets_dir / "reduced_motif_z.png",
        "motif_library": assets_dir / "motif_library.png",
        "motif_stoichiometry": assets_dir / "motif_stoichiometry.png",
        "motif_reference_examples": assets_dir
        / "motif_reference_examples.png",
        "dimensional_motifs": assets_dir / "dimensional_motifs.png",
        "tarasov_structure_projection": assets_dir
        / "tarasov_structure_projection.png",
        "tarasov_giwaxs_comparison": assets_dir
        / "tarasov_giwaxs_comparison.png",
        "ranking_failure_table": assets_dir / "ranking_failure_table.png",
    }
    draw_solve_chain_flow(figures["solve_chain_flow"])
    draw_peak_family_schematic(figures["peak_family_schematic"])
    draw_reduced_motif_z(figures["reduced_motif_z"], structure_summary)
    draw_motif_library(figures["motif_library"])
    draw_motif_stoichiometry(figures["motif_stoichiometry"])
    draw_motif_reference_examples(
        figures["motif_reference_examples"],
        motif_examples,
    )
    draw_dimensional_motifs(figures["dimensional_motifs"])
    draw_structure_projection(
        figures["tarasov_structure_projection"], structure
    )
    draw_giwaxs_panels(
        figures["tarasov_giwaxs_comparison"], example_cif, assets_dir
    )
    draw_ranking_failure_table(figures["ranking_failure_table"])
    return figures


def draw_solve_chain_flow(path: Path) -> None:
    stages = [
        ("Input CIF or image", "calibration, q-grid, texture mode"),
        ("Peak ROIs", "centers, integrated intensities, uncertainty"),
        ("Families", "q ratios, hkl, symmetry, intensity support"),
        ("Information audit", "lattice, hkl, motif, Z, stoichiometry"),
        ("Reduced motif", "Pb/Sn-halide graph + charge balance"),
        ("Wyckoff sites", "multiplicity, symmetry, steric voids"),
        ("Scaffold refine", "peak/image/intensity residuals"),
        ("Organic proxies", "charge, electron count, cavity envelope"),
        ("Organic bodies", "sterics, H-bonds, symmetry"),
        ("Bounded RMC", "rotations, small translations, final ranking"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 11.0))
    fig.patch.set_facecolor("#f7fafc")
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.08,
        0.945,
        "Complete solve chain",
        fontsize=18,
        weight="bold",
        color=THEME["ink"],
        va="top",
    )
    ax.text(
        0.08,
        0.907,
        "A staged diagnostic pipeline with every construction decision tied to evidence.",
        fontsize=9.5,
        color=THEME["muted"],
        va="top",
    )
    rail_x = 0.12
    box_x = 0.20
    box_w = 0.70
    box_h = 0.065
    y_top = 0.835
    gap = 0.016
    centers_y: list[float] = []
    for index, (title, body) in enumerate(stages):
        y = y_top - index * (box_h + gap)
        center_y = y + box_h / 2.0
        centers_y.append(center_y)
        face = "#ffffff" if index < 4 else "#f3faf5"
        ax.add_patch(
            Rectangle(
                (box_x, y),
                box_w,
                box_h,
                facecolor=face,
                edgecolor=THEME["rule"],
                linewidth=0.9,
            )
        )
        ax.add_patch(
            Circle(
                (rail_x, center_y),
                0.022,
                facecolor=THEME["navy"] if index < 4 else THEME["green"],
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(
            rail_x,
            center_y,
            f"{index + 1}",
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color="white",
        )
        ax.text(
            box_x + 0.025,
            y + box_h - 0.020,
            title,
            fontsize=9.4,
            weight="bold",
            color=THEME["ink"],
            va="top",
        )
        ax.text(
            box_x + 0.255,
            y + box_h - 0.020,
            "\n".join(textwrap.wrap(body, width=52)),
            fontsize=7.7,
            color=THEME["muted"],
            va="top",
        )
    for index in range(len(centers_y) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (rail_x, centers_y[index] - 0.024),
                (rail_x, centers_y[index + 1] + 0.024),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color=THEME["gold"],
            )
        )
    ax.add_patch(
        Rectangle(
            (0.20, 0.045),
            0.70,
            0.055,
            facecolor=THEME["gold_light"],
            edgecolor="none",
        )
    )
    ax.text(
        0.225,
        0.075,
        "Design rule: if a later stage fails while oracle evidence is perfect, fix construction and ranking before retuning peak detection.",
        fontsize=8.0,
        color=THEME["ink"],
        va="center",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_peak_family_schematic(path: Path) -> None:
    rng = np.random.default_rng(20260520)
    families = [
        ("00l", 0.15, "#2b6cb0"),
        ("h00/k00", 0.55, "#2f855a"),
        ("mixed hk/l", 0.95, "#b7791f"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.8), constrained_layout=True
    )
    ax = axes[0]
    ax.set_title("Synthetic ROI centers and integrated intensity")
    ax.set_xlabel("qxy (A^-1)")
    ax.set_ylabel("qz (A^-1)")
    for label, slope, color in families:
        qxy = np.linspace(-2.7, 2.7, 22)
        qz = 0.35 + slope * np.abs(qxy) + rng.normal(0.0, 0.035, qxy.size)
        intensity = (
            25
            + 650 * np.exp(-0.6 * np.abs(qxy))
            + rng.normal(0.0, 25.0, qxy.size)
        )
        ax.scatter(
            qxy,
            qz,
            s=np.clip(intensity / 5.0, 18, 170),
            c=color,
            alpha=0.72,
            label=label,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.plot(qxy, 0.35 + slope * np.abs(qxy), color=color, alpha=0.45)
    ax.legend(frameon=False, loc="upper center", ncols=3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3)
    ax.grid(alpha=0.18)

    ax = axes[1]
    ax.set_title("Family grouping score uses more than distance")
    metrics = ["d-spacing", "hkl fit", "symmetry", "texture", "intensity"]
    good = [0.92, 0.86, 0.78, 0.82, 0.74]
    bad = [0.88, 0.34, 0.28, 0.43, 0.22]
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, good, width=0.36, color="#2f855a", label="kept family")
    ax.bar(x + 0.18, bad, width=0.36, color="#c05621", label="split/penalize")
    ax.set_xticks(x, metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalized support")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_reduced_motif_z(path: Path, summary: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.set_axis_off()
    ax.text(
        0.03,
        0.93,
        "Reduced motif plus Z prevents inorganic atom-count runaway",
        fontsize=15,
        weight="bold",
    )
    blocks = [
        (
            "Full input formula",
            "(MA)2(DMF)2Pb3I8",
            "contains scaffold + cation + neutral solvent",
            "#fff5e6",
        ),
        (
            "Reduced scaffold target",
            summary["worked_example_target"],
            "build motif graph independently of atom coordinates",
            "#e8f4ff",
        ),
        (
            "Infer Z separately",
            "Z from volume, hkl multiplicity,\ncharge, intensity scale",
            "replicate motif only after it is plausible",
            "#edf7ed",
        ),
        (
            "Organic/proxy model",
            "MA+ + DMF proxies",
            "charge balance, electron count, and void constraints",
            "#f4edff",
        ),
    ]
    for index, (title, formula, body, color) in enumerate(blocks):
        x = 0.04 + index * 0.24
        ax.add_patch(
            Rectangle((x, 0.48), 0.2, 0.28, facecolor=color, edgecolor="#333")
        )
        ax.text(x + 0.015, 0.71, title, fontsize=10, weight="bold")
        ax.text(x + 0.015, 0.62, formula, fontsize=13, weight="bold")
        ax.text(
            x + 0.015,
            0.51,
            "\n".join(textwrap.wrap(body, width=25)),
            fontsize=8.5,
        )
        if index < len(blocks) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.205, 0.62),
                    (x + 0.235, 0.62),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=1.2,
                    color="#555",
                )
            )
    y = 0.22
    for i in range(5):
        x = 0.15 + i * 0.12
        draw_octahedron(ax, x, y, scale=0.035, color="#6b46c1")
        if i > 0:
            ax.plot([x - 0.075, x - 0.035], [y, y], color="#2d3748", lw=2)
    ax.text(0.1, 0.08, "motif graph", fontsize=9, ha="center")
    ax.text(0.45, 0.08, "replicate by Z", fontsize=9, ha="center")
    ax.add_patch(
        FancyArrowPatch(
            (0.35, 0.19),
            (0.48, 0.19),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2,
        )
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_motif_library(path: Path) -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(13.5, 7.5), constrained_layout=True
    )
    for ax, motif in zip(axes.ravel(), MOTIF_LIBRARY):
        ax.set_title(
            f"{motif.label}\n{motif.formula}", fontsize=10, weight="bold"
        )
        ax.set_axis_off()
        draw_motif_graph(ax, motif.graph_hint)
        ax.text(
            0.03,
            0.08,
            "\n".join(
                textwrap.wrap(
                    f"{motif.dimensionality}; {motif.sharing}", width=31
                )
            ),
            transform=ax.transAxes,
            fontsize=7.8,
            va="bottom",
        )
    fig.suptitle(
        "Reduced Pb/Sn-halide motif library for construction basis",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_motif_stoichiometry(path: Path) -> None:
    headers = [
        "Topology",
        "Dim.",
        "Representative stoich.",
        "Sharing interpretation",
        "Solver implication",
    ]
    rows = [
        (
            "isolated octahedron",
            "0D",
            "PbI6",
            "no iodide sharing",
            "cluster/coordination prior; high anion charge",
        ),
        (
            "corner chain",
            "1D",
            "PbI5",
            "two axial corners shared",
            "Tarasov PbI5 class; cation-rich low-D salts",
        ),
        (
            "corner layer",
            "2D",
            "PbI4",
            "four equatorial corners shared",
            "Ruddlesden-Popper-like single slabs",
        ),
        (
            "corner network",
            "3D",
            "PbI3",
            "all six octahedral corners shared",
            "ABX3 perovskite lattice, A-site required",
        ),
        (
            "edge ribbon",
            "1D",
            "Pb2I6 / Pb3I8",
            "shared edges build 2- or 3-Pb-wide motifs",
            "reduced ribbon/strip basis before full-cell Z",
        ),
        (
            "hex edge sheet",
            "2D",
            "PbI2",
            "CdI2-type packed edge-sharing sheets",
            "neutral precursor layer and solvate parent",
        ),
        (
            "face chain",
            "1D",
            "PbI3",
            "opposite triangular faces shared",
            "allowed as analog prior, strongly distance-filter",
        ),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 5.9))
    ax.set_axis_off()
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.16, 0.07, 0.15, 0.28, 0.34],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.75)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#24415c")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fafc" if row % 2 else "#edf2f7")
    ax.set_title(
        "Stoichiometry follows octahedral sharing topology",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_motif_reference_examples(
    path: Path,
    motif_examples: list[MotifExampleRecord],
) -> None:
    known = [
        record
        for record in motif_examples
        if record.source == "local_reference"
    ]
    headers = ["Reference", "Motif", "Dim.", "Sharing", "Generated scaffold"]
    rows = []
    for record in known:
        rows.append(
            (
                record.label,
                record.formula,
                record.dimensionality,
                record.sharing,
                (
                    Path(record.generated_cif).name
                    if record.generated_cif
                    else "not generated"
                ),
            )
        )
    fig, ax = plt.subplots(figsize=(13.5, 4.9))
    ax.set_axis_off()
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.26, 0.10, 0.08, 0.30, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1.0, 2.05)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e5f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#e6f1f4")
    ax.set_title(
        "Local reference structures used to expand the motif knowledge base",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_dimensional_motifs(path: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.0), constrained_layout=True
    )
    cases = [
        ("0D isolated octahedra", "octahedron", "molecular salt / cluster"),
        ("1D chains", "chain", "edge- or face-sharing ribbons"),
        ("2D slabs", "sheet", "corner/edge-sharing perovskite layers"),
        ("3D networks", "network", "corner-sharing ABX3-like framework"),
    ]
    for ax, (title, mode, caption) in zip(axes.ravel(), cases):
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, weight="bold")
        draw_dimensional_graph(ax, mode)
        ax.text(
            0.04,
            0.04,
            caption,
            transform=ax.transAxes,
            fontsize=9,
            color="#333",
        )
    fig.suptitle(
        "Dimensionality and octahedral-sharing modes considered by the solver",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_structure_projection(path: Path, structure: Structure) -> None:
    species = [site.specie.symbol for site in structure]
    coords = np.asarray(
        [site.frac_coords % 1.0 for site in structure], dtype=float
    )
    groups = {
        "Pb/Sn": [i for i, symbol in enumerate(species) if symbol in B_SITE],
        "Halide": [i for i, symbol in enumerate(species) if symbol in HALIDES],
        "A-site/inorganic": [
            i for i, symbol in enumerate(species) if symbol in A_SITE
        ],
        "Organic/solvent": [
            i for i, symbol in enumerate(species) if symbol not in INORGANIC
        ],
    }
    colors = {
        "Pb/Sn": "#4c51bf",
        "Halide": "#805ad5",
        "A-site/inorganic": "#2f855a",
        "Organic/solvent": "#718096",
    }
    sizes = {
        "Pb/Sn": 85,
        "Halide": 42,
        "A-site/inorganic": 48,
        "Organic/solvent": 12,
    }
    projections = [("a", "b", 0, 1), ("a", "c", 0, 2), ("b", "c", 1, 2)]
    fig, axes = plt.subplots(
        1, 3, figsize=(13.0, 4.6), constrained_layout=True
    )
    for ax, (xlabel, ylabel, ix, iy) in zip(axes, projections):
        for label, indices in groups.items():
            if not indices:
                continue
            subset = coords[indices]
            ax.scatter(
                subset[:, ix],
                subset[:, iy],
                s=sizes[label],
                c=colors[label],
                label=label,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlabel(f"fractional {xlabel}")
        ax.set_ylabel(f"fractional {ylabel}")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Tarasov worked example: reference scaffold and organic/solvent envelope",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_giwaxs_panels(
    path: Path, example_cif: Path, assets_dir: Path
) -> None:
    try:
        from ewald.simulation.giwaxs import (
            GIWAXSSimulationParameters,
            compare_giwaxs_images,
            simulate_giwaxs_image,
        )

        params = GIWAXSSimulationParameters(
            hkl_extent=3,
            qxy_min=-3.0,
            qxy_max=3.0,
            qz_min=0.0,
            qz_max=3.0,
            resolution_x=160,
            resolution_z=112,
            sigma_theta=0.035,
            sigma_phi=0.25,
            sigma_r=0.04,
        )
        full = simulate_giwaxs_image(example_cif, params)
        scaffold_cif = write_inorganic_only_cif(example_cif, assets_dir)
        scaffold = simulate_giwaxs_image(scaffold_cif, params)
        comparison = compare_giwaxs_images(
            full,
            scaffold,
            target_label="Full Tarasov synthetic target",
            simulated_label="Pb/I scaffold-only simulation",
        )
        arrays = [
            (comparison.target.values, "Full target"),
            (comparison.fitted_simulated.values, "Fitted scaffold"),
            (comparison.difference.values, "Residual"),
        ]
        fig, axes = plt.subplots(
            1, 3, figsize=(13.2, 4.4), constrained_layout=True
        )
        for ax, (values, title) in zip(axes, arrays):
            cmap = "coolwarm" if title == "Residual" else "viridis"
            vmax = float(np.nanquantile(np.abs(values), 0.995))
            if title == "Residual":
                vmin = -vmax
            else:
                vmin = 0.0
            im = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=(-3, 3, 0, 3),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax if vmax > 0 else None,
            )
            ax.set_title(title)
            ax.set_xlabel("qxy (A^-1)")
            ax.set_ylabel("qz (A^-1)")
            fig.colorbar(im, ax=ax, shrink=0.78)
        metrics = comparison.metrics
        fig.suptitle(
            "Worked example GIWAXS comparison: full structure versus inorganic-only scaffold "
            f"(corr={metrics.get('correlation', 0.0):.3f}, "
            f"RMSE={metrics.get('difference_rmse', 0.0):.3f})",
            fontsize=12.5,
            weight="bold",
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        draw_placeholder(path, "GIWAXS comparison unavailable", repr(exc))


def write_inorganic_only_cif(example_cif: Path, assets_dir: Path) -> Path:
    structure = Structure.from_file(str(example_cif))
    species = []
    coords = []
    for site in structure:
        symbol = site.specie.symbol
        if symbol in B_SITE | HALIDES | A_SITE:
            species.append(site.specie)
            coords.append(site.frac_coords)
    scaffold = Structure(structure.lattice, species, coords)
    output = assets_dir / f"{example_cif.stem}_inorganic_only.cif"
    scaffold.to(filename=str(output))
    return output


def draw_ranking_failure_table(path: Path) -> None:
    headers = [
        "Stage",
        "Metric",
        "Failure signal",
        "Code response",
    ]
    rows = [
        (
            "Peak finder",
            "center error, recall, intensity recall",
            "strong peaks missed or biased",
            "ROI threshold schedule, sub-pixel fits, texture-aware masks",
        ),
        (
            "Family grouping",
            "weighted purity, hkl consistency",
            "families merge unrelated arcs",
            "split by symmetry, texture, intensity, uncertainty",
        ),
        (
            "Reduced motif",
            "B:X ratio, charge, anisotropy",
            "atom-count runaway",
            "construct Pb/Sn-halide motif first, infer Z later",
        ),
        (
            "Scaffold",
            "RMS, Bragg intensity corr.",
            "right cell but wrong sites",
            "Wyckoff search, motif graph constraints, intensity refinement",
        ),
        (
            "Organic/RMC",
            "sterics, H-bonds, residual map",
            "voids unexplained or clashes",
            "proxy centers, molecule replacement, bounded rotations",
        ),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_axis_off()
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.14, 0.24, 0.27, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.8)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#24415c")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fafc" if row % 2 else "#edf2f7")
    ax.set_title(
        "Ranking metrics and failure attribution for future solver improvements",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.set_axis_off()
    ax.text(0.5, 0.62, title, ha="center", fontsize=16, weight="bold")
    ax.text(
        0.5,
        0.38,
        "\n".join(textwrap.wrap(message, width=100)),
        ha="center",
        fontsize=9,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_motif_graph(ax: plt.Axes, hint: str) -> None:
    graph = nx.Graph()
    if hint == "octahedron":
        graph.add_node("Pb", pos=(0.5, 0.5), kind="B")
        for i, angle in enumerate(
            np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ):
            name = f"I{i}"
            graph.add_node(
                name,
                pos=(0.5 + 0.28 * np.cos(angle), 0.5 + 0.28 * np.sin(angle)),
                kind="X",
            )
            graph.add_edge("Pb", name)
    elif hint == "dimer":
        add_octahedral_center(graph, "Pb1", (0.38, 0.5))
        add_octahedral_center(graph, "Pb2", (0.62, 0.5))
        graph.add_edge("Pb1", "Pb2")
    elif hint in {"trimer_strip", "open_trimer", "cluster"}:
        centers = [(0.28, 0.52), (0.5, 0.48), (0.72, 0.52)]
        for idx, center in enumerate(centers):
            add_octahedral_center(graph, f"Pb{idx+1}", center)
        graph.add_edges_from([("Pb1", "Pb2"), ("Pb2", "Pb3")])
        if hint == "cluster":
            graph.add_edge("Pb1", "Pb3")
    elif hint in {"sheet", "hex_sheet"}:
        for ix in range(3):
            for iy in range(2):
                add_octahedral_center(
                    graph,
                    f"Pb{ix}{iy}",
                    (
                        0.28 + ix * 0.22 + (0.04 if iy else 0.0),
                        0.38 + iy * 0.23,
                    ),
                    radius=0.08,
                )
        for a, b in zip(list(graph.nodes)[:-1], list(graph.nodes)[1:]):
            if a.startswith("Pb") and b.startswith("Pb"):
                graph.add_edge(a, b)
    elif hint == "pyramid":
        graph.add_node("Pb", pos=(0.5, 0.5), kind="B")
        points = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.5), (0.5, 0.3), (0.62, 0.62)]
        for i, point in enumerate(points):
            name = f"I{i}"
            graph.add_node(name, pos=point, kind="X")
            graph.add_edge("Pb", name)
    pos = nx.get_node_attributes(graph, "pos")
    kinds = nx.get_node_attributes(graph, "kind")
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#4a5568", width=1.5)
    for kind, color, size in (("B", "#4c51bf", 330), ("X", "#805ad5", 120)):
        nodes = [
            node for node, node_kind in kinds.items() if node_kind == kind
        ]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=color,
            node_size=size,
            ax=ax,
            edgecolors="white",
            linewidths=0.8,
        )
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)


def add_octahedral_center(
    graph: nx.Graph,
    name: str,
    center: tuple[float, float],
    *,
    radius: float = 0.12,
) -> None:
    graph.add_node(name, pos=center, kind="B")
    for i, angle in enumerate((0, np.pi / 2, np.pi, 3 * np.pi / 2)):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        ligand = f"{name}_I{i}"
        graph.add_node(ligand, pos=(x, y), kind="X")
        graph.add_edge(name, ligand)


def draw_dimensional_graph(ax: plt.Axes, mode: str) -> None:
    if mode == "octahedron":
        draw_octahedron(ax, 0.5, 0.55, scale=0.18)
    elif mode == "chain":
        for i in range(5):
            draw_octahedron(ax, 0.18 + i * 0.16, 0.55, scale=0.07)
            if i:
                ax.plot(
                    [0.18 + i * 0.16 - 0.11, 0.18 + i * 0.16 - 0.04],
                    [0.55, 0.55],
                    color="#333",
                    lw=2,
                )
    elif mode == "sheet":
        for ix in range(4):
            for iy in range(3):
                draw_octahedron(
                    ax,
                    0.18 + ix * 0.18 + (0.04 if iy % 2 else 0.0),
                    0.28 + iy * 0.18,
                    scale=0.045,
                )
    elif mode == "network":
        for ix in range(3):
            for iy in range(3):
                x = 0.25 + ix * 0.22
                y = 0.3 + iy * 0.2
                draw_octahedron(ax, x, y, scale=0.04)
                if ix:
                    ax.plot([x - 0.18, x - 0.04], [y, y], color="#333", lw=1.4)
                if iy:
                    ax.plot([x, x], [y - 0.16, y - 0.04], color="#333", lw=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_octahedron(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    scale: float = 0.08,
    color: str = "#4c51bf",
) -> None:
    ax.add_patch(
        Circle(
            (x, y), scale * 0.45, facecolor=color, edgecolor="white", lw=0.7
        )
    )
    ligand_color = "#805ad5"
    for dx, dy in (
        (scale, 0),
        (-scale, 0),
        (0, scale),
        (0, -scale),
        (scale * 0.65, scale * 0.65),
        (-scale * 0.65, -scale * 0.65),
    ):
        ax.plot([x, x + dx], [y, y + dy], color="#4a5568", lw=1.0)
        ax.add_patch(
            Circle(
                (x + dx, y + dy),
                scale * 0.26,
                facecolor=ligand_color,
                edgecolor="white",
                lw=0.5,
            )
        )


def write_markdown(
    path: Path,
    manifest: dict[str, Any],
    figures: dict[str, Path],
) -> None:
    summary = manifest["structure_summary"]
    lines = [
        "# Complete Solve Chain Plan",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        f"Worked example CIF: `{manifest['example_cif']}`",
        "",
        "## Executive Summary",
        "",
        *REPORT_SECTIONS["summary"],
        "",
        "## Worked Example Summary",
        "",
        f"- Formula: `{summary['formula']}`",
        f"- Full formula: `{summary['full_formula']}`",
        f"- Site count: `{summary['site_count']}`",
        f"- Lattice abc: `{format_tuple(summary['lattice_abc'])}`",
        f"- Lattice angles: `{format_tuple(summary['lattice_angles'])}`",
        f"- Inorganic reduced formula from CIF counts: `{summary['inorganic_reduced_formula']}`",
        f"- Solve target: `{summary['worked_example_target']}`",
        "",
    ]
    for key, title in [
        ("solve_chain_flow", "Solve Chain Flow"),
        ("peak_family_schematic", "Peak Detection And Families"),
        ("reduced_motif_z", "Reduced Motif Plus Z"),
        ("motif_library", "Motif Library"),
        ("motif_stoichiometry", "Motif Stoichiometry"),
        ("motif_reference_examples", "Reference Motif Examples"),
        ("dimensional_motifs", "Dimensional Motifs"),
        ("tarasov_structure_projection", "Tarasov Structure Projection"),
        ("tarasov_giwaxs_comparison", "GIWAXS Target Simulation Residual"),
        ("ranking_failure_table", "Ranking And Failure Attribution"),
    ]:
        rel = figures[key].relative_to(path.parent)
        lines.extend([f"## {title}", "", f"![{title}]({rel})", ""])
    for key, title in [
        ("program_description", "Program Description"),
        ("peak_chain", "Peak Chain"),
        ("information_evaluation", "Information Evaluation"),
        ("reduced_unit", "Reduced Unit Scaffold Construction"),
        ("motifs", "Motif Library And Perovskite Chemistry"),
        ("wyckoff", "Wyckoff And Site Guessing"),
        ("organic", "Organic And Proxy Placement"),
        ("constraints", "Physical Constraints"),
        ("simulation_schedule", "Simulation And Refinement Schedule"),
        ("training", "Training-Ready Improvement Strategy"),
    ]:
        lines.extend([f"## {title}", ""])
        for paragraph in REPORT_SECTIONS[key]:
            lines.extend([paragraph, ""])
    lines.extend(["## HybriD3 / MatD3 Reference Cache", ""])
    for label, url in HYBRID3_SOURCE_URLS.items():
        lines.append(f"- {label}: {url}")
    lines.extend(
        [
            "",
            "Cache file: `hybrid3_cache/hybrid3_cache.json`",
            "",
            "Seed reference IDs are included so future online runs can replace "
            "the cache entries with live dataset/system/reference metadata.",
            "",
            "## Generated Motif Subunit CIFs",
            "",
            "The report generator writes local reference scaffolds and "
            "idealized motif templates under the motif subunit directory.",
            "",
            "| Motif | Formula | Dim. | Source | CIF |",
            "| --- | --- | --- | --- | --- |",
            *[
                "| "
                f"{record['label']} | {record['formula']} | "
                f"{record['dimensionality']} | {record['source']} | "
                f"`{Path(record['generated_cif']).name if record['generated_cif'] else 'missing'}` |"
                for record in manifest["motif_examples"]
            ],
            "",
            "## HybriD3 Reference Ingestion",
            "",
            "The generated cache stores selected metadata and links only: "
            "dataset ID, system ID, reference ID, formula, dimensionality, "
            "motif tags, source URL, and CIF/download URL when exposed.",
            "",
            "## Thorough Example Test Case",
            "",
            *REPORT_SECTIONS["example_case"],
            "",
            "## Acceptance Criteria Checklist",
            "",
            "- PDF, Markdown, JSON, assets, and HybriD3 cache are generated together.",
            "- Tarasov Pb3I8 worked example is included.",
            "- Reduced motif plus Z, organic proxy placement, Pauling/steric constraints, simulation-resolution schedule, and training pathway are explicit.",
            "- Figures render without GUI dependencies.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(
    path: Path,
    manifest: dict[str, Any],
    figures: dict[str, Path],
) -> None:
    global PAGE_COUNTER
    PAGE_COUNTER = 0
    with PdfPages(path) as pdf:
        add_title_page(pdf, manifest)
        add_contents_page(pdf)
        add_section_divider(
            pdf,
            "I. Signal And Evidence",
            "Extract reliable peaks, families, intensities, and failure attribution before construction.",
        )
        add_text_page(
            pdf,
            "Executive Summary",
            REPORT_SECTIONS["summary"],
            footer="Scope: deterministic baseline now, training-ready interfaces later.",
        )
        add_text_page(
            pdf,
            "Program Description",
            REPORT_SECTIONS["program_description"],
        )
        add_figure_page(
            pdf,
            "Solve Chain Flow",
            figures["solve_chain_flow"],
            "End-to-end diagnostic flow from peaks through organic RMC.",
        )
        add_text_page(
            pdf,
            "Peak Detection And Family Grouping",
            REPORT_SECTIONS["peak_chain"],
        )
        add_figure_page(
            pdf,
            "Peak Finder Diagnostics",
            figures["peak_family_schematic"],
            "Peak center, integrated intensity, family purity, and hkl consistency must be evaluated together.",
        )
        add_text_page(
            pdf,
            "Information Evaluation",
            REPORT_SECTIONS["information_evaluation"],
        )
        add_text_page(
            pdf,
            "Reduced Unit Scaffold Construction",
            REPORT_SECTIONS["reduced_unit"],
        )
        add_figure_page(
            pdf,
            "Reduced Motif Plus Z",
            figures["reduced_motif_z"],
            "The Tarasov example should construct Pb3I8 first and infer replication separately.",
        )
        add_section_divider(
            pdf,
            "II. Motif Chemistry",
            "Use known perovskite, perovskitoid, solvate, and halometallate topology before full-cell atom guessing.",
        )
        add_text_page(pdf, "Motif Library", REPORT_SECTIONS["motifs"])
        add_figure_page(
            pdf,
            "Motif Stoichiometry",
            figures["motif_stoichiometry"],
            "The same PbI6 coordination unit maps to different reduced formulas as corners, edges, or faces are shared.",
        )
        add_motif_examples_page(pdf, manifest)
        add_figure_page(
            pdf,
            "Reference Motif Examples",
            figures["motif_reference_examples"],
            "Local reference CIFs are converted into inorganic scaffold examples for the motif knowledge base.",
        )
        add_figure_page(
            pdf,
            "Motif Library Sheet",
            figures["motif_library"],
            "Reduced Pb/Sn-halide graph basis for perovskite and perovskitoid candidates.",
        )
        add_figure_page(
            pdf,
            "Dimensional Motifs",
            figures["dimensional_motifs"],
            "0D, 1D, 2D, and 3D motif families with corner-, edge-, and face-sharing variants.",
        )
        add_section_divider(
            pdf,
            "III. Construction And Refinement",
            "Choose Wyckoff-compatible sites, construct the scaffold, add organic proxies, and refine by images and peaks.",
        )
        add_text_page(
            pdf, "Wyckoff And Site Guessing", REPORT_SECTIONS["wyckoff"]
        )
        add_text_page(
            pdf,
            "Organic Proxy And Molecule Placement",
            REPORT_SECTIONS["organic"],
        )
        add_text_page(
            pdf, "Physical Constraints", REPORT_SECTIONS["constraints"]
        )
        add_figure_page(
            pdf,
            "Tarasov Structure Example",
            figures["tarasov_structure_projection"],
            "Reference projections show why organic/proxy placement should constrain voids early.",
        )
        add_text_page(
            pdf,
            "Thorough Example Test Case",
            REPORT_SECTIONS["example_case"],
        )
        add_text_page(
            pdf,
            "Simulation And Refinement Schedule",
            REPORT_SECTIONS["simulation_schedule"],
        )
        add_figure_page(
            pdf,
            "GIWAXS Target Simulation Residual",
            figures["tarasov_giwaxs_comparison"],
            "Low-resolution example comparing the full Tarasov structure to an inorganic-only scaffold simulation.",
        )
        add_figure_page(
            pdf,
            "Ranking And Failure Attribution",
            figures["ranking_failure_table"],
            "Each metric should map to a concrete next code improvement.",
        )
        add_section_divider(
            pdf,
            "IV. Training And Reference Expansion",
            "Keep the baseline deterministic while using synthetic and HybriD3 labels to calibrate ranking.",
        )
        add_text_page(
            pdf,
            "Training-Ready Improvement Strategy",
            REPORT_SECTIONS["training"],
        )
        add_hybrid3_reference_page(pdf, manifest)
        add_text_page(
            pdf,
            "Implementation Roadmap",
            [
                "Phase 1 adds peak integrated intensity and truth matching to every synthetic run. Phase 2 replaces full-cell atom guessing with reduced motif plus Z construction. Phase 3 adds Wyckoff-aware scaffold generation and Bragg-intensity-driven ranking. Phase 4 places organic proxies, then molecular bodies, then bounded RMC.",
                "The first acceptance test is the oracle diagnostic: exact lattice, peak positions, hkl families, and stoichiometry should reconstruct the Pb3I8 scaffold within a low cartesian RMS before real peak-finder uncertainty is allowed back into the problem.",
            ],
        )
        add_sources_page(pdf, manifest)


def add_canvas(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=-10)
    ax.set_axis_off()
    return ax


def save_pdf_page(
    pdf: PdfPages, fig: plt.Figure, *, number: bool = True
) -> None:
    global PAGE_COUNTER
    PAGE_COUNTER += 1
    if number:
        fig.text(
            0.92,
            0.034,
            f"{PAGE_COUNTER}",
            fontsize=8,
            color=THEME["gold"],
            ha="right",
        )
    pdf.savefig(fig)


def add_report_header(
    fig: plt.Figure,
    title: str,
    *,
    eyebrow: str = "EWALD Synthetic Refinement",
) -> None:
    ax = add_canvas(fig)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), 1.0, 1.0, facecolor=THEME["paper"], edgecolor="none"
        )
    )
    top = 0.865
    ax.add_patch(
        Rectangle(
            (0.0, top), 1.0, 0.135, facecolor=THEME["white"], edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, top), 0.035, 0.135, facecolor=THEME["navy"], edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.035, top),
            0.008,
            0.135,
            facecolor=THEME["gold"],
            edgecolor="none",
        )
    )
    fig.text(
        0.08,
        0.964,
        eyebrow.upper(),
        fontsize=7.6,
        color=THEME["gold"],
        weight="bold",
    )
    title_lines = textwrap.wrap(title, width=52)
    title_text = "\n".join(title_lines[:2])
    font_size = 15.2 if len(title_lines) <= 1 else 13.0
    fig.text(
        0.08,
        0.932,
        title_text,
        fontsize=font_size,
        weight="bold",
        color=THEME["ink"],
        va="top",
        linespacing=1.0,
    )
    ax.plot(
        [0.08, 0.92], [top - 0.006, top - 0.006], color=THEME["rule"], lw=0.8
    )


def add_report_footer(fig: plt.Figure) -> None:
    y = 0.034
    fig.text(0.08, y, REPORT_FOOTER, fontsize=7.8, color=THEME["muted"])


def add_title_page(pdf: PdfPages, manifest: dict[str, Any]) -> None:
    summary = manifest["structure_summary"]
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.patch.set_facecolor(THEME["paper"])
    ax = add_canvas(fig)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), 0.28, 1.0, facecolor=THEME["navy"], edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.28, 0.0), 0.018, 1.0, facecolor=THEME["gold"], edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.34, 0.15),
            0.56,
            0.62,
            facecolor=THEME["white"],
            edgecolor=THEME["rule"],
            linewidth=0.8,
        )
    )
    fig.text(
        0.08, 0.91, "EWALD", fontsize=17, weight="bold", color=THEME["white"]
    )
    fig.text(
        0.08,
        0.86,
        "synthetic\nrefinement",
        fontsize=10,
        color="#dce8ee",
        linespacing=1.3,
    )
    fig.text(
        0.38,
        0.73,
        "Complete\nSolve-Chain Plan",
        fontsize=29,
        weight="bold",
        color=THEME["ink"],
        va="top",
        linespacing=0.95,
    )
    fig.text(
        0.38,
        0.56,
        "Reduced motifs, perovskite/perovskitoid scaffold construction,\n"
        "organic proxy placement, Bragg intensity refinement,\n"
        "and training-ready diagnostics.",
        fontsize=11.5,
        color=THEME["muted"],
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.38,
        0.43,
        "Worked Example",
        fontsize=12,
        weight="bold",
        color=THEME["blue"],
    )
    facts = [
        f"CIF: {Path(manifest['example_cif']).name}",
        f"Formula: {summary['formula']}",
        f"Reduced scaffold target: {summary['worked_example_target']}",
        f"Generated: {manifest['generated_at']}",
    ]
    facts_text = "\n".join(textwrap.fill(item, width=52) for item in facts)
    fig.text(
        0.38,
        0.325,
        facts_text,
        fontsize=8.3,
        color=THEME["ink"],
        linespacing=1.25,
    )
    for i, label in enumerate(
        (
            "Pb/Sn halides",
            "reduced motif + Z",
            "Bragg intensity",
            "organic proxies",
        )
    ):
        y = 0.255 - i * 0.026
        ax.add_patch(
            Rectangle(
                (0.38, y - 0.011),
                0.19,
                0.022,
                facecolor=THEME["blue_light"],
                edgecolor="none",
            )
        )
        fig.text(
            0.39,
            y - 0.005,
            label,
            fontsize=7.6,
            color=THEME["blue"],
            weight="bold",
        )
    fig.text(
        0.08,
        0.09,
        "Generated from:\n"
        "generate_solve_chain_report.py\n\n"
        "Outputs:\n"
        "PDF / Markdown / JSON\n"
        "figures / motifs / HybriD3 cache",
        fontsize=7.6,
        color="#dce8ee",
        linespacing=1.35,
    )
    save_pdf_page(pdf, fig)
    plt.close(fig)


def add_contents_page(pdf: PdfPages) -> None:
    groups = [
        (
            "Signal And Evidence",
            [
                "Executive summary",
                "Program description",
                "Solve-chain flow",
                "Peak detection and family grouping",
                "Information evaluation",
                "Reduced motif plus Z",
            ],
        ),
        (
            "Motif Chemistry",
            [
                "Motif stoichiometry and reference examples",
                "Motif library and dimensional motifs",
            ],
        ),
        (
            "Construction And Refinement",
            [
                "Wyckoff/site guessing",
                "Organic proxy and molecule placement",
                "Physical constraints",
                "Tarasov Pb3I8 worked example",
                "Simulation and refinement schedule",
                "Ranking/failure attribution",
            ],
        ),
        (
            "Training And References",
            [
                "Training-ready improvement strategy",
                "HybriD3/MatD3 reference ingestion",
                "Implementation roadmap and sources",
            ],
        ),
    ]
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.patch.set_facecolor(THEME["paper"])
    add_report_header(fig, "Contents", eyebrow="Report Navigation")
    ax = add_canvas(fig)
    item_index = 1
    column_x = [0.08, 0.52]
    column_w = 0.40
    for group_index, (group, entries) in enumerate(groups, start=1):
        col = 0 if group_index <= 2 else 1
        row = group_index - 1 if col == 0 else group_index - 3
        x0 = column_x[col]
        y = 0.775 - row * 0.335
        ax.add_patch(
            Rectangle(
                (x0, y - 0.015),
                column_w,
                0.048,
                facecolor=(
                    THEME["navy"]
                    if group_index in {1, 3}
                    else THEME["blue_light"]
                ),
                edgecolor="none",
            )
        )
        fig.text(
            x0 + 0.02,
            y,
            group,
            fontsize=10.4,
            weight="bold",
            color=THEME["white"] if group_index in {1, 3} else THEME["blue"],
            va="center",
        )
        y -= 0.060
        for entry in entries:
            fig.text(
                x0 + 0.025,
                y,
                f"{item_index:02d}",
                fontsize=8.8,
                color=THEME["gold"],
                weight="bold",
            )
            fig.text(
                x0 + 0.075,
                y,
                "\n".join(textwrap.wrap(entry, width=30)),
                fontsize=8.9,
                color=THEME["ink"],
                va="top",
            )
            y -= 0.048
            item_index += 1
    fig.text(
        0.08,
        0.10,
        "The JSON manifest beside this PDF records exact paths, figures, cache records, and algorithm sections.",
        fontsize=9,
        color=THEME["muted"],
    )
    add_report_footer(fig)
    save_pdf_page(pdf, fig)
    plt.close(fig)


def add_section_divider(pdf: PdfPages, title: str, subtitle: str) -> None:
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.patch.set_facecolor(THEME["navy"])
    ax = add_canvas(fig)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), 1.0, 1.0, facecolor=THEME["navy"], edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), 0.035, 1.0, facecolor=THEME["gold"], edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0.13, 0.31), 0.72, 0.36, facecolor="#22475f", edgecolor="none"
        )
    )
    fig.text(
        0.18, 0.58, title, fontsize=27, weight="bold", color=THEME["white"]
    )
    fig.text(
        0.18,
        0.49,
        textwrap.fill(subtitle, width=72),
        fontsize=12.2,
        color="#d8e7ee",
        linespacing=1.35,
    )
    fig.text(
        0.18,
        0.39,
        "method section",
        fontsize=9.2,
        color=THEME["gold"],
        weight="bold",
    )
    save_pdf_page(pdf, fig)
    plt.close(fig)


def add_text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: list[str],
    *,
    footer: str | None = None,
) -> None:
    body_top = 0.805
    body_bottom = 0.085

    def new_page(page_index: int) -> tuple[plt.Figure, plt.Axes, float]:
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.patch.set_facecolor(THEME["paper"])
        resolved_title = title if page_index == 1 else f"{title} (continued)"
        add_report_header(fig, resolved_title)
        ax = add_canvas(fig)
        ax.add_patch(
            Rectangle(
                (0.08, 0.105),
                0.84,
                0.735,
                facecolor=THEME["white"],
                edgecolor=THEME["rule"],
                linewidth=0.8,
            )
        )
        return fig, ax, body_top

    def close_page(
        fig: plt.Figure, *, include_footer_note: bool = False
    ) -> None:
        if include_footer_note and footer:
            fig.text(0.105, 0.087, footer, fontsize=8.5, color=THEME["muted"])
        add_report_footer(fig)
        save_pdf_page(pdf, fig)
        plt.close(fig)

    fig, ax, y = new_page(1)
    page_index = 1

    for paragraph_index, paragraph in enumerate(paragraphs):
        chunks = paragraph.splitlines() if "\n" in paragraph else [paragraph]
        is_block = len(chunks) > 1
        for chunk_index, chunk in enumerate(chunks):
            if not chunk.strip():
                y -= 0.018
                continue
            width = 88 if is_block else 84
            font_size = 8.5 if is_block else 9.2
            line_height = 0.022 if is_block else 0.025
            wrapped_lines = textwrap.wrap(chunk, width=width) or [chunk]
            required = line_height * len(wrapped_lines) + (
                0.012 if not is_block else 0.006
            )
            if y - required < body_bottom:
                close_page(fig)
                page_index += 1
                fig, ax, y = new_page(page_index)
            if not is_block and chunk_index == 0:
                ax.add_patch(
                    Rectangle(
                        (0.105, y - 0.009),
                        0.010,
                        0.010,
                        facecolor=THEME["gold"],
                        edgecolor="none",
                    )
                )
                x = 0.13
            else:
                x = 0.12
            fig.text(
                x,
                y,
                "\n".join(wrapped_lines),
                fontsize=font_size,
                va="top",
                color=THEME["ink"],
                linespacing=1.25,
            )
            y -= required
        if paragraph_index < len(paragraphs) - 1:
            y -= 0.012
    close_page(fig, include_footer_note=True)


def add_motif_examples_page(pdf: PdfPages, manifest: dict[str, Any]) -> None:
    records = manifest.get("motif_examples", [])
    known = [
        record
        for record in records
        if record.get("source") == "local_reference"
    ]
    idealized = [
        record
        for record in records
        if record.get("source") == "idealized_template"
    ]
    paragraphs = [
        "The report now carries a generated motif subunit library. Known local reference structures are converted to inorganic scaffold CIFs, while idealized template CIFs capture the topology and reduced formula used by the independent scaffold builder.",
        "Local references: "
        + "; ".join(
            f"{record['formula']} from {record['label']}" for record in known
        ),
        "Idealized templates: "
        + "; ".join(
            f"{record['formula']} {record['dimensionality']} {record['sharing']}"
            for record in idealized
        ),
        "These CIFs are not meant to replace crystallographic references. They are small, explicit construction priors that let the solver test whether a candidate lattice can host a chemically plausible reduced motif before it guesses the full atom count.",
    ]
    add_text_page(pdf, "Generated Motif Subunit CIF Library", paragraphs)


def add_figure_page(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    caption: str,
) -> None:
    image = plt.imread(image_path)
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.patch.set_facecolor(THEME["paper"])
    add_report_header(fig, title, eyebrow="Figure Plate")
    canvas = add_canvas(fig)
    canvas.add_patch(
        Rectangle(
            (0.08, 0.205),
            0.84,
            0.60,
            facecolor=THEME["white"],
            edgecolor=THEME["rule"],
            linewidth=0.8,
        )
    )
    ax = fig.add_axes([0.105, 0.245, 0.79, 0.52])
    ax.imshow(image)
    ax.set_axis_off()
    canvas.add_patch(
        Rectangle(
            (0.08, 0.105),
            0.84,
            0.075,
            facecolor=THEME["white"],
            edgecolor=THEME["rule"],
            linewidth=0.6,
        )
    )
    fig.text(
        0.105,
        0.158,
        textwrap.fill(caption, width=96),
        fontsize=8.7,
        color=THEME["ink"],
        va="top",
    )
    add_report_footer(fig)
    save_pdf_page(pdf, fig)
    plt.close(fig)


def add_hybrid3_reference_page(
    pdf: PdfPages, manifest: dict[str, Any]
) -> None:
    cache = manifest["hybrid3_cache"]
    records = cache.get("records", [])[:4]
    lines = [
        "HybriD3/MatD3 ingestion should remain a metadata cache, not a full database mirror. The solver needs enough information to choose references and download selected CIFs later: dataset ID, system ID, reference ID, source URL, formula, dimensionality, motif tags, and a CIF/download URL when exposed by the API.",
        "Current generated cache:",
    ]
    for record in records:
        lines.append(
            "- "
            f"dataset={record.get('dataset_id')}; "
            f"system={record.get('system_id')}; "
            f"reference={record.get('reference_id')}; "
            f"source={record.get('source_label')}; "
            f"url={record.get('url')}"
        )
    if cache.get("api_attempts"):
        first_attempt = cache["api_attempts"][0]
        lines.append(
            "API probe example: "
            f"{first_attempt.get('url')} "
            f"status={first_attempt.get('status_code', 'unavailable')}."
        )
    lines.append(
        "Future online runs should replace seed records with live HybriD3 reference structures, then tag each reference by reduced motif, Z, dimensionality, and sharing mode."
    )
    add_text_page(pdf, "HybriD3 Reference Ingestion", ["\n".join(lines)])


def add_sources_page(pdf: PdfPages, manifest: dict[str, Any]) -> None:
    hybrid3 = manifest["hybrid3_cache"]
    source_lines = [
        "Local source files and generated outputs",
        f"- Example CIF: {manifest['example_cif']}",
        "- Report generator: scripts/generate_solve_chain_report.py",
        "- HybriD3 cache: hybrid3_cache/hybrid3_cache.json",
        "",
        "HybriD3 / MatD3 references",
    ]
    for label, url in HYBRID3_SOURCE_URLS.items():
        source_lines.append(f"- {label}: {url}")
    source_lines.extend(
        [
            "",
            f"Cached reference records: {hybrid3.get('record_count', 0)}",
            "The cache stores selected metadata and links, not a full database mirror.",
        ]
    )
    add_text_page(pdf, "Sources And Cache Policy", ["\n".join(source_lines)])


def format_tuple(values: Any) -> str:
    return ", ".join(f"{float(value):.4g}" for value in values)


if __name__ == "__main__":
    main()
