"""Headless structure benchmark harness tests."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ewald.analysis.structure import (
    LatticeCandidate,
    StructurePeak,
    generate_ranked_cif_records,
)
from ewald.analysis.structure import REFERENCE_MOLECULES
from ewald.benchmark import (
    BenchmarkRunConfig,
    BenchmarkStructureSpec,
    SyntheticRefinementConfig,
    chemistry_stoichiometry_hypotheses,
    default_structure_specs,
    perovskite_scaffold_hypotheses,
    run_structure_benchmark,
)
from ewald.benchmark.synthetic_refinement import (
    assess_candidate_assignments_against_truth,
    assess_peak_families_against_truth,
    electron_proxy_element_for_count,
    formula_electron_count,
    generate_organic_electron_proxy_cifs,
    generate_organic_replacement_cifs,
    generate_organic_rmc_variant_cifs,
    generate_scaffold_candidate_cifs,
    match_detected_peaks_to_truth,
    organic_electron_proxy_plan,
    _scheduled_texture_mode,
    _synthetic_texture_parameters,
)
from ewald.benchmark.structure_benchmark import _score_bragg_peak_intensity_match
from ewald.io.project import load_project


def test_default_structure_specs_include_requested_molecules(repo_root):
    specs = default_structure_specs(repo_root / "example" / "structures")
    by_name = {spec.cif_path.name: spec for spec in specs}

    assert len(specs) == 15
    assert by_name["BA_PbI3_1DMF_Dahlman_2019.cif"].organic_molecules == (
        "BA",
        "DMF",
    )
    assert by_name["PbI2_NMP_2019_NanfengZheng.cif"].organic_molecules == (
        "MA",
        "NMP",
    )
    assert REFERENCE_MOLECULES["BA"]["formula"] == "C4H12N"
    assert REFERENCE_MOLECULES["NMP"]["formula"] == "C5H9NO"


def test_chemistry_hypotheses_prioritize_layered_perovskite_derivatives():
    hypotheses = chemistry_stoichiometry_hypotheses(
        ("Pb", "I"),
        ("MA", "DMF"),
        limit=12,
    )

    assert hypotheses[0] == "(MA)2(DMF)2Pb3I8"
    assert hypotheses[1] == "(MA)4(DMF)4Pb6I16"
    assert "(MA)2(DMF)2PbI2" in hypotheses
    assert "(MA)2(DMF)2PbI3" in hypotheses


def test_perovskite_motif_hypotheses_cover_dimensionality_and_connectivity():
    records = perovskite_scaffold_hypotheses(
        ("Pb", "I"),
        ("MA", "DMF"),
        limit=24,
    )
    by_formula = {record["formula"]: record for record in records}

    assert by_formula["(MA)2(DMF)2Pb3I8"]["motif_id"] == (
        "mixed_condensed_m3x8_solvate"
    )
    assert by_formula["(MA)2(DMF)2Pb3I8"]["charge_balance"][
        "net_charge"
    ] == pytest.approx(0.0)
    assert {"3D", "2D", "1D", "0D"} <= {
        str(record["dimensionality"]).split("/")[0] for record in records
    }
    assert {"corner_sharing", "edge_sharing", "face_sharing"} <= {
        record["connectivity"] for record in records
    }


def test_perovskite_motif_hypotheses_support_inorganic_a_site_cations():
    records = perovskite_scaffold_hypotheses(
        ("Cs", "Pb", "Br"),
        (),
        limit=4,
    )

    assert records[0]["formula"] == "CsPbBr3"
    assert records[0]["dimensionality"] == "3D"
    assert records[0]["charge_balance"]["net_charge"] == pytest.approx(0.0)


def test_chemistry_hypotheses_prioritize_neutral_solvates_without_cation():
    hypotheses = chemistry_stoichiometry_hypotheses(
        ("Pb", "I"),
        ("DMF",),
        limit=4,
    )

    assert hypotheses[0] == "(DMF)PbI2"
    assert hypotheses[1] == "(DMF)2Pb2I4"
    assert hypotheses[2] == "(DMF)PbI3"


def test_filtered_inorganic_scaffold_header_records_planned_organics(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    source = tmp_path / "full.cif"
    source.write_text(
        "\n".join(
            [
                "data_full",
                "_chemical_formula_sum 'C H6 I Pb N'",
                "_cell_length_a 6.0",
                "_cell_length_b 6.0",
                "_cell_length_c 6.0",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "# molecular species: MA, DMF",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_occupancy",
                "Pb1 Pb 0.0 0.0 0.0 1",
                "I1 I 0.5 0.5 0.5 1",
                "MA1_C1 C 0.2 0.2 0.2 1",
                "MA1_N1 N 0.3 0.3 0.3 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scaffold = tmp_path / "scaffold.cif"

    benchmark_module._write_filtered_cif(
        source,
        scaffold,
        keep_elements={"Pb", "I"},
        organic_note="MA, DMF",
    )

    text = scaffold.read_text(encoding="utf-8")
    assert "# structure variant: inorganic scaffold only" in text
    assert "# organic molecules to add: MA, DMF" in text
    assert (
        "# molecular species: inorganic scaffold only; planned organics: "
        "MA, DMF"
    ) in text
    assert "_chemical_formula_sum 'I1 Pb1'" in text
    assert "Pb1 Pb" in text
    assert "I1 I" in text
    assert "MA1_C1" not in text
    assert "MA1_N1" not in text


def test_estimated_density_prior_prefers_full_heavy_halide_cell():
    from ewald.benchmark.experimental_refinement import (
        _estimated_candidate_density_g_cm3,
        _estimated_density_plausibility_penalty,
        _perovskite_axis_prior_penalty,
    )

    candidate = LatticeCandidate(
        candidate_id="candidate_001",
        crystal_system="Orthorhombic",
        a=17.1,
        b=22.0,
        c=4.55,
    )
    reduced = {"Pb": 3, "I": 8, "C": 8, "H": 26, "N": 4, "O": 2}
    full_cell = {element: count * 2 for element, count in reduced.items()}

    reduced_density = _estimated_candidate_density_g_cm3(reduced, candidate)
    full_density = _estimated_candidate_density_g_cm3(full_cell, candidate)

    assert reduced_density < 3.0
    assert full_density > 3.0
    assert (
        _estimated_density_plausibility_penalty(reduced, reduced_density) > 0.4
    )
    assert (
        _estimated_density_plausibility_penalty(full_cell, full_density) == 0.0
    )
    wrong_axis_cell = LatticeCandidate(
        candidate_id="candidate_002",
        crystal_system="Orthorhombic",
        a=17.1,
        b=10.9,
        c=9.5,
    )
    assert _perovskite_axis_prior_penalty(full_cell, candidate) == 0.0
    assert _perovskite_axis_prior_penalty(full_cell, wrong_axis_cell) > 0.5


def test_axis_scale_expansion_retains_two_axis_half_cell_variant():
    from ewald.benchmark.experimental_refinement import (
        ExperimentalRefinementConfig,
        _axis_scale_prior_penalty,
        _expand_axis_scale_candidates,
        _perovskite_axis_prior_penalty,
    )

    candidate = LatticeCandidate(
        candidate_id="candidate_001",
        crystal_system="Orthorhombic",
        a=34.0,
        b=22.0,
        c=9.0,
        score=0.1,
    )
    cfg = ExperimentalRefinementConfig(candidate_max_candidates=1)

    expanded = _expand_axis_scale_candidates([candidate], cfg)
    ids = {item.candidate_id for item in expanded}

    assert "candidate_001_axis_scale_0p5_1_0p5" in ids
    assert _axis_scale_prior_penalty(
        "candidate_001_axis_scale_0p5_0p5_0p5"
    ) > _axis_scale_prior_penalty("candidate_001_axis_scale_0p5_1_0p5")
    assert _perovskite_axis_prior_penalty(
        {"Pb": 6, "I": 16},
        LatticeCandidate(
            candidate_id="candidate_002",
            crystal_system="Orthorhombic",
            a=17.0,
            b=21.0,
            c=10.0,
        ),
    ) > _perovskite_axis_prior_penalty(
        {"Pb": 6, "I": 16},
        LatticeCandidate(
            candidate_id="candidate_003",
            crystal_system="Orthorhombic",
            a=17.0,
            b=21.0,
            c=5.0,
        ),
    )


def test_synthetic_refinement_truth_diagnostics_match_peak_families():
    from ewald.analysis.structure import StructurePeak

    peaks = [
        StructurePeak("p1", "P1", 1.0, 0.2),
        StructurePeak("p2", "P2", 2.0, 0.4),
        StructurePeak("p3", "P3", 3.0, 0.6),
    ]
    truth = [
        {"h": 1, "k": 0, "l": 0, "qxy": 1.01, "qz": 0.21},
        {"h": 2, "k": 0, "l": 0, "qxy": 2.01, "qz": 0.39},
        {"h": 0, "k": 1, "l": 0, "qxy": 3.00, "qz": 0.59},
    ]

    matched = match_detected_peaks_to_truth(peaks, truth, tolerance=0.04)
    families = [
        {
            "family_id": "family_001",
            "kind": "qxy multiples",
            "peak_ids": ["p1", "p2"],
        }
    ]
    family_assessment = assess_peak_families_against_truth(
        families,
        matched["matches"],
    )
    candidate_metrics = assess_candidate_assignments_against_truth(
        [
            {
                "candidate_id": "candidate_001",
                "assignments": [
                    {"peak_id": "p1", "hkl": "(1 0 0)"},
                    {"peak_id": "p2", "hkl": "(2 0 0)"},
                    {"peak_id": "p3", "hkl": "(0 0 1)"},
                ],
            }
        ],
        matched["matches"],
    )

    assert matched["summary"]["matched_detected_peak_count"] == 3
    assert matched["summary"]["top25_truth_recall"] == pytest.approx(1.0)
    assert matched["summary"]["intensity_weighted_recall"] == pytest.approx(1.0)
    assert matched["truth_rank_diagnostics"]["top_truth_recall"]["top_25"][
        "matched_truth_peak_count"
    ] == 3
    assert matched["summary"]["mean_radial_position_error"] == pytest.approx(
        matched["position_error_summary"]["mean_radial_error"]
    )
    assert matched["position_error_summary"]["mean_delta_qz"] == pytest.approx(
        0.01 / 3.0
    )
    assert matched["matches"][0]["position_error"][
        "delta_abs_qxy"
    ] == pytest.approx(-0.01)
    assert family_assessment["summary"]["high_purity_family_count"] == 1
    assert family_assessment["summary"][
        "weighted_mean_truth_family_purity"
    ] == pytest.approx(1.0)
    assert candidate_metrics["candidate_001"][
        "hkl_family_accuracy"
    ] == pytest.approx(1.0)


def test_bragg_peak_intensity_matching_prefers_relative_intensity_agreement():
    from ewald.analysis.structure import StructurePeak
    from ewald.benchmark.structure_benchmark import (
        _score_bragg_peak_intensity_match,
    )

    peaks = [
        StructurePeak(
            "p1",
            "P1",
            1.0,
            0.2,
            metadata={"integrated_intensity": 100.0},
        ),
        StructurePeak(
            "p2",
            "P2",
            1.5,
            0.4,
            metadata={"integrated_intensity": 50.0},
        ),
        StructurePeak(
            "p3",
            "P3",
            2.0,
            0.8,
            metadata={"integrated_intensity": 10.0},
        ),
    ]
    good_rows = [
        {"h": 1, "k": 0, "l": 0, "qxy": 1.0, "qz": 0.2, "amplitude": 100.0},
        {"h": 1, "k": 1, "l": 0, "qxy": 1.5, "qz": 0.4, "amplitude": 50.0},
        {"h": 2, "k": 0, "l": 0, "qxy": 2.0, "qz": 0.8, "amplitude": 10.0},
    ]
    bad_rows = [
        {"h": 1, "k": 0, "l": 0, "qxy": 1.0, "qz": 0.2, "amplitude": 10.0},
        {"h": 1, "k": 1, "l": 0, "qxy": 1.5, "qz": 0.4, "amplitude": 50.0},
        {"h": 2, "k": 0, "l": 0, "qxy": 2.0, "qz": 0.8, "amplitude": 100.0},
    ]

    good = _score_bragg_peak_intensity_match(
        peaks,
        good_rows,
        tolerance=0.04,
        max_peaks=10,
    )
    bad = _score_bragg_peak_intensity_match(
        peaks,
        bad_rows,
        tolerance=0.04,
        max_peaks=10,
    )

    assert good["status"] == "computed"
    assert good["matched_peak_fraction"] == pytest.approx(1.0)
    assert good["intensity_match_penalty"] < bad["intensity_match_penalty"]
    assert good["log_intensity_correlation"] > bad["log_intensity_correlation"]


def test_synthetic_refinement_config_defaults_to_experimental_output():
    cfg = SyntheticRefinementConfig()

    assert cfg.output_dir == Path("example/projects/experimental_refinement")
    assert cfg.rank_generated_cifs_with_image_fit is False
    assert cfg.texture_modes[0] == "out_of_plane_stack"
    assert cfg.assume_unit_cell_symmetry is True
    assert cfg.bragg_intensity_weight > 0.0


def test_synthetic_refinement_texture_schedule_is_controlled():
    cfg = SyntheticRefinementConfig(
        texture_modes=("out_of_plane_stack", "in_plane_stack"),
        fiber_tilt_jitter_deg=0.0,
        texture_azimuth_jitter_deg=0.0,
    )
    rng = np.random.default_rng(123)

    assert _scheduled_texture_mode(cfg, 1) == "out_of_plane_stack"
    assert _scheduled_texture_mode(cfg, 2) == "in_plane_stack"
    assert _scheduled_texture_mode(cfg, 3) == "out_of_plane_stack"

    out_of_plane = _synthetic_texture_parameters(
        rng,
        cfg,
        "out_of_plane_stack",
    )
    in_plane = _synthetic_texture_parameters(rng, cfg, "in_plane_stack")

    assert out_of_plane.theta_x_deg == pytest.approx(0.0)
    assert in_plane.theta_x_deg == pytest.approx(90.0)
    assert out_of_plane.theta_y_deg in {0.0, 90.0, 180.0, 270.0}
    assert in_plane.theta_y_deg in {0.0, 90.0, 180.0, 270.0}


def test_synthetic_scaffold_generation_strips_organic_formula_labels(tmp_path):
    candidate = LatticeCandidate(
        "candidate_001",
        "Orthorhombic",
        8.0,
        12.0,
        16.0,
    )
    spec = BenchmarkStructureSpec(
        cif_path=tmp_path / "reference.cif",
        inorganic_atoms=("Pb", "I"),
        organic_molecules=("MA", "DMF"),
    )
    cfg = SyntheticRefinementConfig(max_scaffolds_to_validate=1)

    records = generate_scaffold_candidate_cifs(
        spec,
        [candidate],
        ["(MA)2(DMF)2Pb3I8"],
        cfg,
    )

    assert records[0]["inorganic_stoichiometry"] == "Pb3I8"
    assert records[0]["composition_elements"] == {"Pb": 3.0, "I": 8.0}
    assert "MA" not in records[0]["cif_text"]
    assert "DMF" not in records[0]["cif_text"]


def test_synthetic_scaffold_generation_keeps_stoichiometry_diversity(tmp_path):
    candidate = LatticeCandidate(
        "candidate_001",
        "Orthorhombic",
        8.0,
        12.0,
        16.0,
    )
    spec = BenchmarkStructureSpec(
        cif_path=tmp_path / "reference.cif",
        inorganic_atoms=("Pb", "I"),
        organic_molecules=("MA", "DMF"),
    )
    cfg = SyntheticRefinementConfig(max_scaffolds_to_validate=4)

    records = generate_scaffold_candidate_cifs(
        spec,
        [candidate],
        [
            "(MA)2(DMF)2Pb3I8",
            "(MA)2(DMF)2PbI3",
            "(MA)2(DMF)2PbI2",
        ],
        cfg,
    )

    assert {"Pb3I8", "PbI3"} <= {
        record["inorganic_stoichiometry"] for record in records
    }
    by_stoichiometry = {
        record["inorganic_stoichiometry"]: record for record in records
    }
    assert by_stoichiometry["Pb3I8"]["stoichiometry_prior_penalty"] == 0.0
    assert by_stoichiometry["PbI3"][
        "stoichiometry_prior_penalty"
    ] == pytest.approx(0.30)


def test_organic_electron_proxy_maps_electron_count_to_nearest_element():
    assert electron_proxy_element_for_count(18)["element"] == "K"
    assert electron_proxy_element_for_count(20)["element"] == "Ca"
    assert formula_electron_count("C3H7NO") == 40


def test_organic_electron_proxy_generation_uses_charge_balanced_counts(
    tmp_path,
):
    candidate = LatticeCandidate(
        "candidate_001",
        "Orthorhombic",
        8.0,
        12.0,
        16.0,
    )
    spec = BenchmarkStructureSpec(
        cif_path=tmp_path / "reference.cif",
        inorganic_atoms=("Pb", "I"),
        organic_molecules=("MA", "DMF"),
    )
    cfg = SyntheticRefinementConfig(max_organic_proxy_cifs_to_compare=2)

    plan = organic_electron_proxy_plan(
        "(MA)2(DMF)2Pb3I8",
        spec.organic_molecules,
    )
    records = generate_organic_electron_proxy_cifs(
        spec,
        [candidate],
        ["(MA)2(DMF)2Pb3I8", "(MA)2(DMF)2PbI2"],
        cfg,
    )

    assert plan["molecule_counts"] == {"DMF": 2.0, "MA": 2.0}
    assert plan["organic_proxy_formula"] == "K2Zr2"
    assert records[0]["coordinate_model"] == "organic_electron_proxy"
    assert records[0]["organic_proxy_stoichiometry"] == "Pb3I8K2Zr2"
    assert records[0]["charge_balance"]["net_charge"] == pytest.approx(0.0)
    assert records[0]["composition_elements"] == {
        "Pb": 3.0,
        "I": 8.0,
        "K": 2.0,
        "Zr": 2.0,
    }
    assert "MA" not in records[0]["cif_text"]
    assert "DMF" not in records[0]["cif_text"]


def test_bragg_intensity_match_rewards_relative_intensity_agreement():
    peaks = [
        StructurePeak(
            peak_id="p1",
            label="p1",
            qxy=1.0,
            qz=0.5,
            metadata={"integrated_intensity": {"integrated_intensity": 100.0}},
        ),
        StructurePeak(
            peak_id="p2",
            label="p2",
            qxy=1.5,
            qz=0.7,
            metadata={"integrated_intensity": {"integrated_intensity": 20.0}},
        ),
    ]
    good = _score_bragg_peak_intensity_match(
        peaks,
        [
            {"qxy": 1.0, "qz": 0.5, "amplitude": 100.0, "h": 1, "k": 0, "l": 0},
            {"qxy": 1.5, "qz": 0.7, "amplitude": 20.0, "h": 0, "k": 1, "l": 0},
        ],
        tolerance=0.05,
        max_peaks=10,
    )
    swapped = _score_bragg_peak_intensity_match(
        peaks,
        [
            {"qxy": 1.0, "qz": 0.5, "amplitude": 20.0, "h": 1, "k": 0, "l": 0},
            {"qxy": 1.5, "qz": 0.7, "amplitude": 100.0, "h": 0, "k": 1, "l": 0},
        ],
        tolerance=0.05,
        max_peaks=10,
    )

    assert good["matched_peak_fraction"] == pytest.approx(1.0)
    assert good["intensity_match_penalty"] < swapped["intensity_match_penalty"]


def test_organic_replacement_and_rmc_stages_write_full_molecules(tmp_path):
    candidate = LatticeCandidate(
        "candidate_001",
        "Orthorhombic",
        8.0,
        8.0,
        12.0,
    )
    spec = BenchmarkStructureSpec(
        cif_path=tmp_path / "reference.cif",
        inorganic_atoms=("Pb", "I"),
        organic_molecules=("MA",),
    )
    cfg = SyntheticRefinementConfig(
        max_organic_replacement_cifs_to_compare=2,
        organic_rmc_steps=2,
    )

    records = generate_organic_replacement_cifs(
        spec,
        [candidate],
        ["(MA)2PbI4"],
        cfg,
    )
    assert records
    assert records[0]["coordinate_model"] == "full_organic_replacement"
    assert "# molecular species: MA" in records[0]["cif_text"]

    source = tmp_path / "organic_replacement.cif"
    source.write_text(records[0]["cif_text"], encoding="utf-8")
    rmc_records = generate_organic_rmc_variant_cifs(
        source,
        fileset_id="test",
        rng=np.random.default_rng(7),
        cfg=cfg,
        output_dir=tmp_path / "rmc",
    )

    assert len(rmc_records) == 2
    assert Path(rmc_records[0]["path"]).exists()
    assert rmc_records[0]["organic_rmc_transform"]["body_count"] > 0
    assert rmc_records[0]["chemistry_metrics"]["physical_penalty"] >= 0.0


def test_blind_cif_generation_disables_exact_reference_template():
    candidate = LatticeCandidate(
        candidate_id="candidate_001",
        crystal_system="Orthorhombic",
        a=8.772,
        b=9.081,
        c=12.672,
        score=0.2,
    )
    molecules = [
        {"label": "MA", **REFERENCE_MOLECULES["MA"]},
        {"label": "DMF", **REFERENCE_MOLECULES["DMF"]},
    ]

    record = generate_ranked_cif_records(
        candidate,
        atoms=("Pb", "I"),
        molecules=molecules,
        stoichiometry="(MA)2(DMF)2Pb3I8",
        limit=1,
        allow_explicit_templates=False,
    )[0]

    assert record["coordinate_model"] == "deterministic_fractional_grid"
    assert "explicit full unit cell generated" not in record["cif_text"]
    assert "MA1_C1" in record["cif_text"]
    assert "MA2_N" in record["cif_text"]
    assert "DMF1_O1" in record["cif_text"]
    assert "DMF2_C" in record["cif_text"]


def test_symmetry_constrained_cif_generation_pairs_atoms_and_molecules():
    candidate = LatticeCandidate(
        candidate_id="candidate_001",
        crystal_system="Cubic",
        a=12.0,
        b=12.0,
        c=12.0,
        score=0.2,
    )
    molecules = [{"label": "MA", **REFERENCE_MOLECULES["MA"]}]

    record = generate_ranked_cif_records(
        candidate,
        atoms=("Pb", "I"),
        molecules=molecules,
        stoichiometry="(MA)2PbI3",
        limit=1,
        allow_explicit_templates=False,
        assume_unit_cell_symmetry=True,
    )[0]

    assert record["unit_cell_symmetry_assumed"] is True
    assert record["coordinate_model"] == "symmetry_constrained_fractional_grid"
    assert "# unit-cell placement: inversion-symmetric" in record["cif_text"]
    atom_rows = _atom_rows_from_cif_text(record["cif_text"])
    _assert_inversion_symmetric_by_element(atom_rows)
    _assert_no_cross_element_coordinate_collapses(atom_rows)


def test_full_molecule_atoms_preserved_in_generated_cifs():
    candidate = LatticeCandidate(
        candidate_id="candidate_001",
        crystal_system="Orthorhombic",
        a=16.5,
        b=21.8,
        c=4.6,
        score=0.2,
    )
    molecules = [
        {"label": "MA", **REFERENCE_MOLECULES["MA"]},
        {"label": "DMF", **REFERENCE_MOLECULES["DMF"]},
    ]

    record = generate_ranked_cif_records(
        candidate,
        atoms=("Pb", "I"),
        molecules=molecules,
        stoichiometry="(MA)2(DMF)2Pb3I8",
        limit=1,
        allow_explicit_templates=False,
    )[0]

    assert _element_counts_from_cif_text(record["cif_text"]) == {
        "Pb": 3,
        "I": 8,
        "C": 8,
        "H": 26,
        "N": 4,
        "O": 2,
    }


def test_organic_rigid_body_orientation_avoids_close_contacts(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    lattice_length = 18.0
    rows = [
        ("Pb1", "Pb", 0.5, 0.5, 0.5),
        ("I1", "I", 0.5, 0.5, 0.33),
        ("I2", "I", 0.5, 0.5, 0.67),
        ("I3", "I", 0.5, 0.33, 0.5),
        ("I4", "I", 0.5, 0.67, 0.5),
        ("I5", "I", 0.33, 0.5, 0.5),
        ("I6", "I", 0.67, 0.5, 0.5),
    ]
    rows.extend(
        _molecule_rows(
            "MA",
            1,
            np.asarray([0.505, 0.5, 0.5]),
            lattice_length,
        )
    )
    source = _write_raw_cif(tmp_path / "clashing.cif", lattice_length, rows)
    output = tmp_path / "physicalized.cif"

    benchmark_module._write_physicalized_cif(source, output)
    metrics = benchmark_module._cif_physical_chemistry_metrics(output)

    assert (
        _element_counts_from_cif_text(output.read_text(encoding="utf-8"))["H"]
        == 6
    )
    assert metrics["organic_inorganic_clash_count"] == 0
    assert metrics["organic_restraints"]["penalty"] < 0.05


def test_donor_acceptor_orientation_improves_hydrogen_bond_plausibility():
    import ewald.benchmark.structure_benchmark as benchmark_module

    lattice = np.eye(3) * 12.0
    center = np.asarray([0.5, 0.5, 0.5])
    bad_rotation = benchmark_module._axis_angle_rotation_matrix(
        np.asarray([0.0, 0.0, 1.0]),
        np.pi,
    )
    coords, elements = _molecule_coords(
        "MA",
        center,
        12.0,
        rotation=bad_rotation,
    )
    halide = np.asarray([0.75, 0.5, 0.5])
    inorganic_rows = [
        {
            "label": "I1",
            "element": "I",
            "frac": halide,
            "row_index": 0,
        }
    ]
    adjusted = {0: halide}
    before = benchmark_module._pose_hydrogen_bond_score(
        coords,
        elements,
        "MA",
        inorganic_rows,
        [],
        adjusted,
        lattice,
    )
    candidates = benchmark_module._rigid_body_pose_candidates(
        coords,
        elements,
        "MA",
        center,
        inorganic_rows,
        [],
        adjusted,
        lattice,
    )
    best = max(
        candidates,
        key=lambda item: benchmark_module._molecule_pose_score(
            item,
            elements,
            "MA",
            center,
            inorganic_rows,
            [],
            [],
            adjusted,
            lattice,
        ),
    )
    after = benchmark_module._pose_hydrogen_bond_score(
        best,
        elements,
        "MA",
        inorganic_rows,
        [],
        adjusted,
        lattice,
    )

    assert after > before + 0.2
    assert after > 0.2


def test_physical_metrics_penalize_free_floating_inorganic_ions(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    path = _write_raw_cif(
        tmp_path / "free_ions.cif",
        30.0,
        [
            ("Pb1", "Pb", 0.05, 0.05, 0.05),
            ("I1", "I", 0.55, 0.55, 0.55),
        ],
    )

    metrics = benchmark_module._cif_physical_chemistry_metrics(path)

    assert metrics["coordination"]["free_cation_count"] == 1
    assert metrics["coordination"]["uncoordinated_halide_count"] == 1
    assert metrics["physical_penalty"] > 0.0


def test_duplicate_motif_suffixes_are_separate_molecular_bodies():
    import ewald.benchmark.structure_benchmark as benchmark_module

    assert benchmark_module._molecular_body_token("DMF1_O1") == "DMF1"
    assert benchmark_module._molecular_body_token("DMF1_O1B") == "DMF1B"
    assert benchmark_module._molecular_body_token("MA2_H6C") == "MA2C"
    assert benchmark_module._molecular_species_token("DMF1B") == "DMF"


def test_zero_fractional_shift_grid_deduplicates_noop_shift():
    import ewald.benchmark.structure_benchmark as benchmark_module

    assert benchmark_module._fractional_shift_grid(0.0) == [(0.0, 0.0, 0.0)]


def test_generated_short_axis_organics_seed_into_open_voids(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    candidate = LatticeCandidate(
        candidate_id="candidate_003_axis_scale_0p5_0p5_0p5",
        crystal_system="Orthorhombic",
        a=16.281967,
        b=10.830950,
        c=4.405309,
        score=0.2,
    )
    molecules = [
        {"label": "MA", **REFERENCE_MOLECULES["MA"]},
        {"label": "DMF", **REFERENCE_MOLECULES["DMF"]},
    ]

    record = generate_ranked_cif_records(
        candidate,
        atoms=("Pb", "I"),
        molecules=molecules,
        stoichiometry="(MA)2(DMF)2Pb3I8",
        limit=1,
        allow_explicit_templates=False,
    )[0]
    path = tmp_path / "short_axis_seed.cif"
    path.write_text(record["cif_text"], encoding="utf-8")

    metrics = benchmark_module._cif_physical_chemistry_metrics(path)

    assert metrics["organic_inorganic_clash_count"] < 10
    assert metrics["organic_restraints"]["body_count"] == 4


def test_duplicate_motif_translation_preserves_coordination_and_reduces_clashes(
    tmp_path,
):
    import ewald.benchmark.experimental_refinement as refinement_module
    import ewald.benchmark.structure_benchmark as benchmark_module

    candidate = LatticeCandidate(
        candidate_id="candidate_003_axis_scale_0p5_0p5_0p5",
        crystal_system="Orthorhombic",
        a=16.281967,
        b=10.830950,
        c=4.405309,
        score=0.2,
    )
    molecules = [
        {"label": "MA", **REFERENCE_MOLECULES["MA"]},
        {"label": "DMF", **REFERENCE_MOLECULES["DMF"]},
    ]
    record = generate_ranked_cif_records(
        candidate,
        atoms=("Pb", "I"),
        molecules=molecules,
        stoichiometry="(MA)2(DMF)2Pb3I8",
        limit=1,
        allow_explicit_templates=False,
    )[0]
    duplicated = refinement_module._duplicate_cif_motif_text(
        record["cif_text"],
        {"Pb": 6, "I": 16, "C": 16, "H": 52, "N": 8, "O": 4},
        translation=(0.5, 0.5, 0.5),
    )
    path = tmp_path / "duplicated_seed.cif"
    path.write_text(duplicated, encoding="utf-8")

    metrics = benchmark_module._cif_physical_chemistry_metrics(path)

    assert metrics["organic_inorganic_clash_count"] < 40
    assert metrics["coordination"]["free_cation_count"] == 0
    assert metrics["coordination"]["uncoordinated_halide_count"] == 0


def test_body_orientation_score_rejects_short_axis_wrapping():
    import ewald.benchmark.structure_benchmark as benchmark_module

    lattice = np.diag([16.3, 21.7, 4.4])
    center = np.asarray([0.5, 0.5, 0.5])
    template = np.asarray(
        [
            [float(atom[1]), float(atom[2]), float(atom[3])]
            for atom in REFERENCE_MOLECULES["DMF"]["atoms"]
        ],
        dtype=float,
    )
    elements = [str(atom[0]) for atom in REFERENCE_MOLECULES["DMF"]["atoms"]]
    good = (center + template @ np.linalg.inv(lattice)) % 1.0
    rotation = benchmark_module._axis_angle_rotation_matrix(
        np.asarray([0.0, 1.0, 0.0]),
        np.pi / 2.0,
    )
    wrapped = (center + (template @ rotation.T) @ np.linalg.inv(lattice)) % 1.0

    assert (
        benchmark_module._body_internal_restraint_penalty(
            good,
            elements,
            "DMF",
            lattice,
        )
        < 0.05
    )
    assert (
        benchmark_module._body_internal_restraint_penalty(
            wrapped,
            elements,
            "DMF",
            lattice,
        )
        > 0.5
    )


def test_physicalization_rebuilds_known_molecule_body_from_template(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    rows = [
        ("Pb1", "Pb", 0.5, 0.5, 0.5),
        ("I1", "I", 0.5, 0.5, 0.25),
    ]
    for label, element, x, y, z in _molecule_rows(
        "DMF",
        1,
        np.asarray([0.98, 0.5, 0.5]),
        12.0,
    ):
        rows.append((label, element, x % 1.0, y % 1.0, z % 1.0))
    source = _write_raw_cif(tmp_path / "split_dmf.cif", 12.0, rows)
    output = tmp_path / "physicalized.cif"

    benchmark_module._write_physicalized_cif(source, output)
    metrics = benchmark_module._cif_physical_chemistry_metrics(output)
    dmf = next(
        record
        for record in metrics["organic_restraints"]["body_records"]
        if record["body"] == "DMF1"
    )

    assert dmf["unit_cell_boundary_penalty"] == 0.0
    assert dmf["max_pair_distance_deviation"] < 0.05


def test_organic_restraints_penalize_unit_cell_boundary_sharing(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    path = _write_raw_cif(
        tmp_path / "boundary_split.cif",
        12.0,
        [
            ("DMF1_C1", "C", 0.02, 0.5, 0.5),
            ("DMF1_N1", "N", 0.98, 0.5, 0.5),
            ("DMF1_O1", "O", 0.04, 0.55, 0.5),
        ],
    )

    metrics = benchmark_module._cif_physical_chemistry_metrics(path)
    record = metrics["organic_restraints"]["body_records"][0]

    assert record["max_fractional_span"] > 0.9
    assert record["unit_cell_boundary_penalty"] > 0.0
    assert metrics["organic_restraints"]["penalty"] > 0.0


def test_halide_assignment_improves_undercoordinated_pb():
    import ewald.benchmark.structure_benchmark as benchmark_module

    lattice = np.diag([16.0, 16.0, 16.0])
    cation_center = np.asarray([0.5, 0.5, 0.5])
    inv_lattice = np.linalg.inv(lattice)
    directions = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
    ]
    candidates = [
        (cation_center + (direction * 3.05) @ inv_lattice) % 1.0
        for direction in directions
    ]
    halides = [
        {
            "label": f"I{index}",
            "element": "I",
            "frac": np.asarray([0.05 * index, 0.05, 0.05]),
            "row_index": index,
        }
        for index in range(4)
    ]
    adjusted = {
        int(row["row_index"]): np.asarray(row["frac"], dtype=float)
        for row in halides
    }

    benchmark_module._improve_halide_coordination_assignment(
        halides,
        candidates,
        [cation_center],
        [(2.65, 3.65)],
        adjusted,
        lattice,
    )
    near = [
        benchmark_module._pbc_distance(adjusted[index], cation_center, lattice)
        for index in adjusted
    ]

    assert sum(2.65 <= distance <= 3.65 for distance in near) == 4


def test_pair_distribution_validation_reports_partial_pair_mismatch(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module

    reference = _write_raw_cif(
        tmp_path / "reference.cif",
        12.0,
        [("Pb1", "Pb", 0.5, 0.5, 0.5), ("I1", "I", 0.5, 0.5, 0.75)],
    )
    generated = _write_raw_cif(
        tmp_path / "generated.cif",
        12.0,
        [("Pb1", "Pb", 0.5, 0.5, 0.5), ("I1", "I", 0.5, 0.5, 0.95)],
    )

    metrics = benchmark_module._pair_distribution_validation_metrics(
        generated,
        reference,
        max_distance=8.0,
        bin_width=0.1,
    )

    assert metrics["status"] == "computed"
    assert metrics["weighted_l1_distance"] > 0.0
    assert metrics["worst_partial_pairs"][0]["pair"] == "Pb-I"


def test_molecule_refinement_preserves_chemical_geometry_source(
    tmp_path,
    monkeypatch,
):
    import ewald.benchmark.structure_benchmark as benchmark_module
    from ewald.simulation.giwaxs import GIWAXSSimulationParameters

    source = tmp_path / "candidate.cif"
    source.write_text(
        "\n".join(
            [
                "data_candidate",
                "_cell_length_a 8",
                "_cell_length_b 8",
                "_cell_length_c 8",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_occupancy",
                "Pb1 Pb 0.1 0.1 0.1 1",
                "MA1_C C 0.2 0.2 0.2 1",
                "MA1_N N 0.3 0.3 0.3 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured_source_paths = []

    def fake_evaluate(target, path, parameter_grid, *, label):
        physical_penalty = (
            0.0 if str(path).endswith("_chemical_geometry.cif") else 1.0
        )
        return {
            "path": str(path),
            "metrics": {"peak_focus_score": 0.0, "difference_rmse": 0.0},
            "physical_penalty": physical_penalty,
        }

    def fake_physicalize(input_path, output_path):
        output_path.write_text(
            input_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    def fake_best_shifted(
        target,
        source_path,
        selector,
        params,
        cfg,
        output_dir,
        *,
        stage_name,
        selector_mode="element",
    ):
        captured_source_paths.append(Path(source_path))
        return Path(source_path), fake_evaluate(
            target, source_path, [params], label=stage_name
        )

    monkeypatch.setattr(benchmark_module, "_evaluate_cif_path", fake_evaluate)
    monkeypatch.setattr(
        benchmark_module, "_write_physicalized_cif", fake_physicalize
    )
    monkeypatch.setattr(
        benchmark_module, "_best_shifted_cif", fake_best_shifted
    )

    target = xr.DataArray(np.zeros((2, 2)), dims=("qz", "qxy"))
    params = GIWAXSSimulationParameters(
        resolution_z=2,
        resolution_x=2,
        hkl_extent=1,
    )
    cfg = BenchmarkRunConfig(
        output_dir=tmp_path,
        detector_shape=(2, 2),
        qxy_range=(-1.0, 1.0),
        qz_range=(0.0, 1.0),
        refinement_coarse_detector_shape=(2, 2),
        refinement_coarse_hkl_extent=1,
    )

    result = benchmark_module._staged_refinement_path(
        target,
        {
            "cif_id": "candidate",
            "atoms": ["Pb"],
            "molecules": [{"label": "MA"}],
        },
        source,
        params,
        cfg,
        tmp_path / "staged",
    )

    assert captured_source_paths
    assert captured_source_paths[0].name.endswith("_chemical_geometry.cif")
    assert result["path"].endswith("_chemical_geometry.cif")


def test_halide_site_candidates_reject_short_axis_pbc_contacts():
    import ewald.benchmark.structure_benchmark as benchmark_module

    lattice = benchmark_module._lattice_matrix_from_cell(
        {
            "_cell_length_a": 16.3,
            "_cell_length_b": 21.7,
            "_cell_length_c": 4.4,
            "_cell_angle_alpha": 90.0,
            "_cell_angle_beta": 90.0,
            "_cell_angle_gamma": 90.0,
        }
    )
    cation = {
        "label": "Pb1",
        "element": "Pb",
        "frac": np.asarray([0.5, 0.5, 0.0]),
        "row_index": 0,
    }

    candidates = benchmark_module._halide_site_candidates(
        [cation],
        lattice,
        {0: cation["frac"]},
    )

    assert candidates
    assert all(
        benchmark_module._pbc_distance(cation["frac"], candidate, lattice)
        >= 2.65
        for candidate in candidates
    )


def test_structure_benchmark_writes_reproducible_artifacts(tmp_path):
    reference_cif = _write_simple_cif(tmp_path / "Si_reference.cif")
    spec = BenchmarkStructureSpec(
        cif_path=reference_cif,
        inorganic_atoms=("Si",),
        organic_molecules=(),
    )
    config = BenchmarkRunConfig(
        output_dir=tmp_path / "benchmark",
        seed=42,
        simulations_per_structure=1,
        hkl_extent=1,
        detector_shape=(30, 42),
        qxy_range=(-3.0, 3.0),
        qz_range=(0.0, 3.0),
        peak_threshold_percentile=98.0,
        peak_max_peaks=20,
        candidate_hkl_max=1,
        candidate_grid_points=4,
        candidate_max_candidates=3,
        cif_records_per_candidate=1,
        max_generated_cifs_to_compare=2,
        comparison_theta_x_offsets=(0.0,),
        comparison_theta_y_values=(0.0,),
        comparison_plot_count=1,
    )

    result = run_structure_benchmark([spec], config)
    fileset = result.filesets[0]
    project = load_project(fileset["project"])
    benchmark = project.analysis_results["benchmark"][fileset["fileset_id"]]
    qspace = np.load(fileset["mock_qspace"])

    assert Path(fileset["mock_tiff"]).exists()
    assert Path(fileset["project"]).exists()
    assert Path(fileset["readable_project"]).exists()
    assert Path(fileset["peak_detection_plot"]).exists()
    assert Path(fileset["best_generated_cif"]).exists()
    assert Path(result.output_dir / "LOGBOOK.md").exists()
    assert qspace["intensity"].shape == config.detector_shape
    assert benchmark["constraints"]["inorganic_atoms"] == ["Si"]
    assert (
        "reference_lattice_constants"
        in benchmark["constraints"]["excluded_from_solver"]
    )
    assert project.reference_cifs["generated"]


def test_validation_rejects_wrong_lattice_and_composition(tmp_path):
    import ewald.benchmark.structure_benchmark as benchmark_module
    from pymatgen.core import Lattice, Structure

    reference_cif = _write_simple_cif(tmp_path / "Si_reference.cif")
    wrong = tmp_path / "wrong_lattice_composition.cif"
    structure = Structure(
        Lattice.cubic(7.2),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    wrong.write_text(structure.to(fmt="cif"), encoding="utf-8")

    validation = benchmark_module._validate_best_generated_structure(
        [{"path": str(wrong), "metrics": {}}],
        reference_cif,
        tmp_path / "local_best",
        tmp_path / "global_best",
    )

    assert validation["status"] == "validated"
    assert validation["attained_solution"] is False
    assert validation["lattice_metrics"]["sorted_abc_relative_error"] > 0.08
    assert (
        validation["composition_metrics"]["element_count_relative_error"]
        > 0.10
    )


def _write_simple_cif(path: Path) -> Path:
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.4),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )
    path.write_text(structure.to(fmt="cif"), encoding="utf-8")
    return path


def _element_counts_from_cif_text(cif_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    in_atom_rows = False
    for line in cif_text.splitlines():
        stripped = line.strip()
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            not in_atom_rows
            or not stripped
            or stripped.startswith(("_", "#", "loop_"))
        ):
            continue
        parts = stripped.split()
        if len(parts) < 6:
            continue
        counts[parts[1]] = counts.get(parts[1], 0) + 1
    return counts


def _atom_rows_from_cif_text(
    cif_text: str,
) -> list[tuple[str, str, np.ndarray]]:
    rows: list[tuple[str, str, np.ndarray]] = []
    in_atom_rows = False
    for line in cif_text.splitlines():
        stripped = line.strip()
        if stripped == "_atom_site_occupancy":
            in_atom_rows = True
            continue
        if (
            not in_atom_rows
            or not stripped
            or stripped.startswith(("_", "#", "loop_"))
        ):
            continue
        parts = stripped.split()
        if len(parts) < 6:
            continue
        rows.append(
            (
                parts[0],
                parts[1],
                np.asarray(
                    [float(parts[2]), float(parts[3]), float(parts[4])],
                    dtype=float,
                ),
            )
        )
    return rows


def _assert_inversion_symmetric_by_element(
    rows: list[tuple[str, str, np.ndarray]],
) -> None:
    grouped: dict[str, list[np.ndarray]] = {}
    for _, element, coord in rows:
        grouped.setdefault(element, []).append(np.asarray(coord, dtype=float))
    for element, coords in grouped.items():
        for coord in coords:
            target = _wrapped_fractional(-coord)
            assert any(
                np.allclose(_wrapped_fractional(candidate), target, atol=2e-6)
                for candidate in coords
            ), f"{element} site {coord} lacks inversion partner"


def _assert_no_cross_element_coordinate_collapses(
    rows: list[tuple[str, str, np.ndarray]],
) -> None:
    elements_by_site: dict[tuple[int, int, int], set[str]] = {}
    for _, element, coord in rows:
        key = tuple(int(round(value * 1_000_000)) for value in coord)
        elements_by_site.setdefault(key, set()).add(element)
    collapsed = {
        key: elements
        for key, elements in elements_by_site.items()
        if len(elements) > 1
    }
    assert not collapsed


def _wrapped_fractional(coord: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(coord, dtype=float) % 1.0
    wrapped[np.isclose(wrapped, 1.0, atol=1e-9)] = 0.0
    wrapped[np.isclose(wrapped, 0.0, atol=1e-9)] = 0.0
    return wrapped


def _write_raw_cif(
    path: Path,
    lattice_length: float,
    rows: list[tuple[str, str, float, float, float]],
) -> Path:
    lines = [
        "data_test",
        f"_cell_length_a {lattice_length}",
        f"_cell_length_b {lattice_length}",
        f"_cell_length_c {lattice_length}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    lines.extend(
        f"{label} {element} {x:.6f} {y:.6f} {z:.6f} 1"
        for label, element, x, y, z in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _molecule_rows(
    label: str,
    copy_index: int,
    center: np.ndarray,
    lattice_length: float,
) -> list[tuple[str, str, float, float, float]]:
    coords, elements = _molecule_coords(label, center, lattice_length)
    counters: dict[str, int] = {}
    rows = []
    for element, frac in zip(elements, coords, strict=True):
        counters[element] = counters.get(element, 0) + 1
        rows.append(
            (
                f"{label}{copy_index}_{element}{counters[element]}",
                element,
                float(frac[0]),
                float(frac[1]),
                float(frac[2]),
            )
        )
    return rows


def _molecule_coords(
    label: str,
    center: np.ndarray,
    lattice_length: float,
    *,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    template = REFERENCE_MOLECULES[label]["atoms"]
    cart = np.asarray(
        [
            [float(atom[1]), float(atom[2]), float(atom[3])]
            for atom in template
        ],
        dtype=float,
    )
    if rotation is not None:
        cart = cart @ rotation.T
    coords = (np.asarray(center, dtype=float) + cart / lattice_length) % 1.0
    elements = [str(atom[0]) for atom in template]
    return coords, elements
