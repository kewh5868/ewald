"""Tests for Structure Analysis data model and approximation helpers."""

import numpy as np
import pytest
from qtpy import QtCore, QtGui, QtWidgets

from ewald.analysis.structure import (
    CRYSTAL_SYSTEM_SPACE_GROUP_RANGES,
    DEFAULT_PHASE_TAG,
    PHASE_SECONDARY,
    CandidateSearchConfig,
    LatticeCandidate,
    StructurePeak,
    build_structure_peaks,
    generate_ranked_cif_records,
    group_peak_families,
    guess_lattice_candidates,
    refine_lattice_candidate,
    registered_wyckoff_possibilities,
    wyckoff_combination_count,
    wyckoff_registry_summary,
    wyckoff_site_combinations,
    wyckoff_space_group_option,
)
from ewald.crystallography.cif import compare_cif_atom_coordinates
from ewald.crystallography.overlay import quaternion_from_axis_angle
from ewald.data.models import (
    PEAK_POINT_KIND_FITTED_CENTER,
    PEAK_POINT_KIND_GAP_ESTIMATED,
    ProjectState,
    ROIRegion,
)
from ewald.ui.peak_identification import (
    HKL_LABEL_MODE_ALL,
    HKL_LABEL_MODE_NONE,
    HKL_LABEL_MODE_PARTIAL,
    PeakIdentificationPane,
)
from ewald.ui.structure_analysis import (
    COL_PEAK_ID,
    COL_PHASE,
    COL_QXY,
    StructureAnalysisPane,
)


def test_lattice_candidate_refinement_and_guessing_find_cubic_cell():
    q0 = 2.0 * np.pi / 10.0
    peaks = [
        StructurePeak("p1", "P1", q0, 0.0, hkl_label="(1 0 0)"),
        StructurePeak(
            "p2",
            "P2",
            q0 * np.sqrt(2.0),
            0.0,
            hkl_label="(1 1 0)",
        ),
        StructurePeak(
            "p3",
            "P3",
            q0 * np.sqrt(3.0),
            0.0,
            hkl_label="(1 1 1)",
        ),
    ]
    config = CandidateSearchConfig(
        crystal_systems=("Cubic",),
        hkl_max=3,
        q_tolerance=0.035,
        grid_points=8,
        max_candidates=4,
    )

    refined = refine_lattice_candidate(
        peaks,
        LatticeCandidate("guess", "Cubic", 9.2, 9.2, 9.2),
        config,
    )
    candidates = guess_lattice_candidates(peaks, config)

    assert refined.a == pytest.approx(10.0, abs=0.08)
    assert refined.matched_count == 3
    assert candidates
    assert candidates[0].a == pytest.approx(10.0, abs=0.25)


def test_candidate_guessing_prefers_simple_lower_lattice_cells():
    q0 = 2.0 * np.pi / 10.0
    peaks = [StructurePeak("p1", "P1", q0, 0.0)]
    config = CandidateSearchConfig(
        crystal_systems=("Triclinic", "Monoclinic", "Cubic", "Tetragonal"),
        hkl_max=3,
        q_tolerance=0.035,
        lattice_min=2.5,
        lattice_max=25.0,
        grid_points=8,
        max_candidates=6,
    )

    candidates = guess_lattice_candidates(peaks, config)

    assert candidates
    assert candidates[0].crystal_system == "Cubic"
    assert candidates[0].a == pytest.approx(10.0, abs=0.3)
    if len(candidates) > 1:
        assert candidates[0].a <= min(
            candidate.a for candidate in candidates[1:]
        )


def test_projected_axis_guessing_recovers_fiber_texture_cell():
    a = 17.2
    b = 22.0
    c = 4.55
    qa = 2.0 * np.pi / a
    qb = 2.0 * np.pi / b
    qc = 2.0 * np.pi / c
    peaks = [
        StructurePeak("p1", "P1", 0.0, 2.0 * qb),
        StructurePeak("p2", "P2", qa, qb),
        StructurePeak("p3", "P3", qa, 2.0 * qb),
        StructurePeak("p4", "P4", 2.0 * qa, 3.0 * qb),
        StructurePeak("p5", "P5", np.hypot(qa, qc), 2.0 * qb),
        StructurePeak("p6", "P6", np.hypot(2.0 * qa, qc), 3.0 * qb),
    ]
    config = CandidateSearchConfig(
        crystal_systems=("Orthorhombic",),
        hkl_max=3,
        q_tolerance=0.035,
        lattice_min=3.0,
        lattice_max=35.0,
        grid_points=3,
        max_candidates=3,
    )

    candidates = guess_lattice_candidates(peaks, config)

    assert candidates
    best = candidates[0]
    assert best.crystal_system == "Orthorhombic"
    assert best.projection_mode == "fiber_qxy_qz"
    assert sorted([best.a, best.b, best.c]) == pytest.approx(
        sorted([a, b, c]),
        abs=0.25,
    )
    assert best.matched_count == len(peaks)


def test_best_guess_refinement_uses_crystal_orientation_projection():
    q0 = 2.0 * np.pi / 10.0
    orientation = quaternion_from_axis_angle((0.0, 0.0, 1.0), 45.0)
    peaks = [
        StructurePeak(
            "p1",
            "P1",
            q0 / np.sqrt(2.0),
            0.0,
            hkl_label="(1 0 0)",
        ),
        StructurePeak("p2", "P2", 0.0, q0, hkl_label="(0 0 1)"),
    ]
    oriented_config = CandidateSearchConfig(
        crystal_systems=("Cubic",),
        q_tolerance=0.02,
        lattice_min=5.0,
        lattice_max=15.0,
        orientation_quaternion=orientation,
    )
    plain_config = CandidateSearchConfig(
        crystal_systems=("Cubic",),
        q_tolerance=0.02,
        lattice_min=5.0,
        lattice_max=15.0,
    )

    oriented = refine_lattice_candidate(
        peaks,
        LatticeCandidate("guess", "Cubic", 9.0, 9.0, 9.0),
        oriented_config,
    )
    plain = refine_lattice_candidate(
        peaks,
        LatticeCandidate("guess", "Cubic", 9.0, 9.0, 9.0),
        plain_config,
    )

    assert oriented.a == pytest.approx(10.0, abs=0.03)
    assert oriented.score < 1.0e-6
    assert oriented.matched_count == 2
    assert oriented.orientation_quaternion == pytest.approx(orientation)
    assert oriented.assignments[0]["qxy_predicted"] == pytest.approx(
        q0 / np.sqrt(2.0),
        abs=1.0e-6,
    )
    assert plain.score > oriented.score + 0.05


def test_peak_family_grouping_respects_phase_tags():
    peaks = [
        StructurePeak("p1", "P1", 1.00, 0.3),
        StructurePeak("p2", "P2", 1.03, 0.8),
        StructurePeak("p3", "P3", 1.02, 1.1, phase_tag=PHASE_SECONDARY),
    ]

    families = group_peak_families(
        peaks,
        tolerance=0.06,
        phase_tag=DEFAULT_PHASE_TAG,
    )

    assert families
    assert frozenset({"p1", "p2"}) in {
        frozenset(family["peak_ids"]) for family in families
    }
    assert all("p3" not in family["peak_ids"] for family in families)
    reviewed_family = next(
        family
        for family in families
        if {"p1", "p2"} <= set(family["peak_ids"])
    )
    assert reviewed_family["confidence"] > 0.0
    assert "within tolerance" in reviewed_family["reason"]


def test_project_state_peak_helpers_sync_structure_analysis_state():
    project = ProjectState()
    data_id = "synthetic"
    roi = project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            roi_id="roi_1",
            kind="box",
            qxy_min=0.3,
            qxy_max=0.5,
            qz_min=0.9,
            qz_max=1.1,
        )
    )
    project.peak_sets[data_id] = [
        {
            "peak_id": "p1",
            "label": "Peak 1",
            "qxy": 0.4,
            "qz": 1.0,
            "roi_id": roi.roi_id,
            "source": "manual",
        },
        {
            "peak_id": "gap",
            "label": "Gap",
            "qxy": 0.6,
            "qz": 1.4,
            "source": "gap estimate",
            "point_kind": PEAK_POINT_KIND_GAP_ESTIMATED,
            "metadata": {
                "gap_estimate": True,
                "estimate_method": "symmetry interpolation",
            },
        },
    ]

    fitted = project.set_peak_fit_result(
        data_id,
        "p1",
        {
            "center_qxy": 0.42,
            "center_qz": 1.18,
            "status": "ok",
            "statistics": {"r_squared": 0.98},
        },
        roi_id=roi.roi_id,
    )
    project.set_peak_phase_tag(data_id, "p1", PHASE_SECONDARY)
    project.set_peak_hkl_tag(data_id, "p1", h=1, k=0, l=2)
    gap = project.sync_structure_analysis_peak_from_fit(data_id, "gap")

    state = project.analysis_results["structure_analysis"][data_id]
    assert fitted["center_qxy"] == pytest.approx(0.42)
    assert fitted["source"] == PEAK_POINT_KIND_FITTED_CENTER
    assert state["candidate_selection_stale"] is True
    assert state["phase_tags"] == [PHASE_SECONDARY]
    refreshed = next(
        item for item in state["peaks"] if item["peak_id"] == "p1"
    )
    assert refreshed["phase_tag"] == PHASE_SECONDARY
    assert refreshed["hkl_label"] == "(1 0 2)"
    assert gap["gap_estimated"] is True
    assert gap["source"] == PEAK_POINT_KIND_GAP_ESTIMATED
    assert gap["estimate_method"] == "symmetry interpolation"

    project.update_structure_analysis_peak(data_id, "p1", qxy=0.5, qz=1.25)
    project.set_peak_fit_result(
        data_id,
        "p1",
        {
            "center_qxy": 0.44,
            "center_qz": 1.2,
            "statistics": {"r_squared": 0.99},
        },
        roi_id=roi.roi_id,
    )
    edited = next(item for item in state["peaks"] if item["peak_id"] == "p1")
    assert edited["qxy"] == pytest.approx(0.5)
    assert edited["qz"] == pytest.approx(1.25)
    assert edited["source"] == "structure-analysis-manual"


def test_structure_peak_builder_reads_metadata_tags_and_gap_kind():
    peaks = build_structure_peaks(
        [
            {
                "peak_id": "p1",
                "qxy": 0.3,
                "qz": 0.4,
                "metadata": {
                    "phase_tag": PHASE_SECONDARY,
                    "hkl": {"h": 1, "k": 1, "l": 0},
                },
            },
            {
                "peak_id": "gap",
                "qxy": 0.7,
                "qz": 0.2,
                "point_kind": PEAK_POINT_KIND_GAP_ESTIMATED,
                "metadata": {"gap_estimate": True},
            },
        ]
    )

    assert peaks[0].phase_tag == PHASE_SECONDARY
    assert peaks[0].hkl_label == "(1 1 0)"
    assert peaks[1].source == "gap estimate"
    assert peaks[1].phase_tag == "gap-estimated"
    assert peaks[1].status == "gap-estimated"


def test_wyckoff_registry_covers_all_crystal_systems_and_space_groups():
    summary = wyckoff_registry_summary()
    systems = {
        item["crystal_system"]: item for item in summary["crystal_systems"]
    }

    assert set(systems) == set(CRYSTAL_SYSTEM_SPACE_GROUP_RANGES)
    assert summary["space_group_count"] == 230
    assert all(item["space_group_count"] > 0 for item in systems.values())

    all_groups = registered_wyckoff_possibilities(include_sites=False)
    cubic_groups = registered_wyckoff_possibilities(
        "Cubic",
        include_sites=False,
    )
    pm3m = wyckoff_space_group_option(221)
    combinations = wyckoff_site_combinations(
        221,
        site_count=3,
        max_combinations=5,
    )

    assert len(all_groups) == 230
    assert len(cubic_groups) == 36
    assert pm3m.crystal_system == "Cubic"
    assert {site.site_label for site in pm3m.sites} >= {"1a", "1b", "3c"}
    assert wyckoff_combination_count(221, site_count=3) >= len(combinations)
    assert wyckoff_combination_count(221, site_count=3, ordered=True) >= (
        wyckoff_combination_count(221, site_count=3)
    )
    assert combinations
    assert combinations[0]["space_group_number"] == 221
    assert len(combinations[0]["site_labels"]) == 3


def test_generated_cif_records_include_atoms_molecules_and_ranking():
    candidate = LatticeCandidate("candidate_001", "Cubic", 6.3, 6.3, 6.3)

    records = generate_ranked_cif_records(
        candidate,
        atoms=["Pb", "I"],
        molecules=[{"label": "MA", "formula": "CH6N"}],
        stoichiometry="MAPbI3",
        limit=2,
    )

    assert [record["rank"] for record in records] == [1, 2]
    assert records[0]["score"] <= records[1]["score"]
    assert "_cell_length_a 6.300000" in records[0]["cif_text"]
    assert "Pb1 Pb" in records[0]["cif_text"]
    assert "MA" in records[0]["cif_text"]
    assert records[0]["space_group"]["crystal_system"] == "Cubic"
    assert records[0]["wyckoff_combination"]["site_labels"]
    assert "_symmetry_space_group_name_H-M 'P1'" in records[0]["cif_text"]


def test_generated_cif_fallback_writes_full_composition_parseable(tmp_path):
    candidate = LatticeCandidate("candidate_001", "Cubic", 6.3, 6.3, 6.3)

    records = generate_ranked_cif_records(
        candidate,
        atoms=["Pb", "I"],
        molecules=[{"label": "MA", "formula": "CH6N"}],
        stoichiometry="MAPbI3",
        limit=1,
    )
    generated_path = tmp_path / "mapbi3.cif"
    generated_path.write_text(records[0]["cif_text"], encoding="utf-8")

    comparison = compare_cif_atom_coordinates(generated_path, generated_path)

    assert comparison["generated_summary"]["site_count"] == 12
    assert comparison["composition_delta"] == {}


def test_generated_cif_records_expand_hybrid_stoichiometry(tmp_path):
    candidate = LatticeCandidate(
        "candidate_001",
        "Orthorhombic",
        17.4,
        22.1,
        4.6,
    )

    records = generate_ranked_cif_records(
        candidate,
        atoms=["Pb", "I"],
        molecules=[
            {"label": "MA", "formula": "CH6N"},
            {"label": "DMF", "formula": "C3H7NO"},
        ],
        stoichiometry="MA2(DMF)2Pb3I8",
        limit=1,
    )

    assert records[0]["space_group"]["number"] == 58
    assert records[0]["composition_elements"] == {
        "C": 8.0,
        "H": 26.0,
        "N": 4.0,
        "O": 2.0,
        "Pb": 3.0,
        "I": 8.0,
    }
    assert records[0]["coordinate_model"] == "explicit_full_cell_ma_dmf_pb3i8"
    assert records[0]["status"] == "full-cell molecular draft"
    cif_text = records[0]["cif_text"]
    assert "_symmetry_space_group_name_H-M 'P1'" in cif_text
    assert "# inferred parent space group: Pnnm (58)" in cif_text
    assert "_chemical_formula_sum 'C16 H52 I16 N8 O4 Pb6'" in cif_text
    assert "Pb1_01 Pb" in cif_text
    assert "I4_16 I" in cif_text
    assert "DMF" in cif_text
    generated_path = tmp_path / "generated.cif"
    generated_path.write_text(cif_text, encoding="utf-8")

    comparison = compare_cif_atom_coordinates(generated_path, generated_path)

    assert comparison["generated_summary"]["site_count"] == 102
    assert comparison["composition_delta"] == {}
    assert comparison["coordinate_match"]["matched_count"] == 102
    assert comparison["coordinate_match"]["unmatched_count"] == 0
    assert comparison["coordinate_match"]["fractional_rms"] == pytest.approx(
        0.0
    )
    assert (
        comparison["coordinate_match"]["by_element"]["Pb"]["matched_count"]
        == 6
    )


def test_generated_pb3i8_cif_infers_built_in_molecules_from_formula():
    candidate = LatticeCandidate(
        "candidate_001",
        "Orthorhombic",
        17.4,
        22.1,
        4.6,
    )

    records = generate_ranked_cif_records(
        candidate,
        atoms=["Pb", "I"],
        molecules=[],
        stoichiometry="MA2(DMF)2Pb3I8",
        limit=1,
    )

    cif_text = records[0]["cif_text"]
    assert records[0]["coordinate_model"] == "explicit_full_cell_ma_dmf_pb3i8"
    assert "# molecular species: DMF, MA" in cif_text
    assert "_chemical_formula_sum 'C16 H52 I16 N8 O4 Pb6'" in cif_text
    assert "C4_16 C" in cif_text


def test_structure_analysis_table_imports_fit_centers_and_user_edits(qtbot):
    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {
            "peak_id": "p1",
            "label": "Peak 01",
            "qxy": 0.4,
            "qz": 1.0,
            "source": "manual",
        }
    ]
    project.fits["synthetic"] = {
        "peak_fit": {
            "p1": {
                "fit_2d": {
                    "center_qxy": 0.42,
                    "center_qz": 1.18,
                    "statistics": {"r_squared": 0.98},
                }
            }
        }
    }

    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)

    state = project.analysis_results["structure_analysis"]["synthetic"]
    assert state["peaks"][0]["qxy"] == pytest.approx(0.42)
    assert state["peaks"][0]["qz"] == pytest.approx(1.18)
    assert state["peaks"][0]["source"] == "ROI fit"

    pane.peak_table.item(0, COL_QXY).setText("0.55")
    assert state["peaks"][0]["qxy"] == pytest.approx(0.55)
    assert state["peaks"][0]["source"] == "user edit"

    phase_combo = pane.peak_table.cellWidget(0, COL_PHASE)
    phase_combo.setCurrentText(PHASE_SECONDARY)
    assert state["peaks"][0]["phase_tag"] == PHASE_SECONDARY


def test_structure_approximation_tab_uses_top_actions_and_three_columns(qtbot):
    project = ProjectState()
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)

    tab = pane.analysis_tabs.widget(0)
    layout = tab.layout()
    buttons = layout.itemAt(0).layout()
    grid = layout.itemAt(1).layout()

    assert buttons.itemAt(0).widget() is pane.refine_button
    assert buttons.itemAt(1).widget() is pane.guess_button
    assert buttons.itemAt(2).widget() is pane.overlay_button
    assert buttons.itemAt(3).widget() is pane.outliers_button
    assert isinstance(grid, QtWidgets.QGridLayout)

    assert grid.itemAtPosition(0, 1).widget() is pane.phase_filter_combo
    assert grid.itemAtPosition(0, 3).widget() is pane.guess_system_combo
    assert grid.itemAtPosition(0, 5).widget() is pane.hkl_max
    assert grid.itemAtPosition(1, 1).widget() is pane.lattice_a
    assert grid.itemAtPosition(1, 3).widget() is pane.lattice_b
    assert grid.itemAtPosition(1, 5).widget() is pane.lattice_c
    assert grid.itemAtPosition(2, 1).widget() is pane.lattice_alpha
    assert grid.itemAtPosition(2, 3).widget() is pane.lattice_beta
    assert grid.itemAtPosition(2, 5).widget() is pane.lattice_gamma
    assert grid.itemAtPosition(3, 1).widget() is pane.q_tolerance
    assert grid.itemAtPosition(3, 3).widget() is pane.relative_tolerance
    assert grid.itemAtPosition(3, 5).widget() is pane.grid_points


def test_structure_analysis_family_selection_highlights_plot_peaks(qtbot):
    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {"peak_id": "p1", "label": "P1", "qxy": 1.0, "qz": 0.2},
        {"peak_id": "p2", "label": "P2", "qxy": 1.02, "qz": 0.5},
        {"peak_id": "p3", "label": "P3", "qxy": 2.0, "qz": 0.8},
    ]
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)
    if pane.family_highlight_scatter is None:
        pytest.skip("pyqtgraph is unavailable")

    pane.family_tolerance.setValue(0.05)
    families = pane.suggest_peak_families()

    assert families
    assert pane.family_table.rowCount() > 0
    pane.family_table.selectRow(0)

    x_data, y_data = pane.family_highlight_scatter.getData()
    assert list(x_data) == pytest.approx([1.0, 1.02])
    assert list(y_data) == pytest.approx([0.2, 0.5])
    assert pane._selected_family_peak_ids() == {"p1", "p2"}

    pane.family_table.clearSelection()
    x_data, y_data = pane.family_highlight_scatter.getData()
    assert len(x_data) == 0
    assert len(y_data) == 0


def test_structure_analysis_family_review_flags_and_deletes(qtbot):
    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {"peak_id": "p1", "label": "P1", "qxy": 1.0, "qz": 0.2},
        {"peak_id": "p2", "label": "P2", "qxy": 1.02, "qz": 0.6},
    ]
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)

    pane.suggest_peak_families()
    assert pane.family_table.rowCount() > 0
    low_confidence_family = pane._family_records()[0]
    low_confidence_family["confidence"] = 0.25
    pane._sync_families()
    low_confidence_id = low_confidence_family["family_id"]

    pane.family_confidence_filter.setValue(0.5)
    displayed_ids = {
        pane.family_table.item(row, 0).text()
        for row in range(pane.family_table.rowCount())
    }
    assert low_confidence_id not in displayed_ids
    pane.family_confidence_filter.setValue(0.0)

    pane.family_table.selectRow(0)
    family_id = pane.family_table.item(0, 0).text()

    assert any(
        shortcut.key().toString() == "F" for shortcut in pane._family_shortcuts
    )
    pane.toggle_selected_family_flags()
    family = pane._family_by_id(family_id)
    assert family["user_flag"] == "appropriate"
    assert pane.family_table.item(0, 1).text() == "Appropriate"

    pane.set_selected_family_flag("inappropriate")
    assert family["user_flag"] == "inappropriate"
    assert pane.family_table.item(0, 1).text() == "Inappropriate"

    pane.delete_selected_families()
    assert pane._family_by_id(family_id) is None


def test_structure_analysis_family_plot_edits_members(qtbot):
    class FakePoint:
        def __init__(self, payload):
            self._payload = payload

        def data(self):
            return self._payload

    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {"peak_id": "p1", "label": "P1", "qxy": 1.0, "qz": 0.2},
        {"peak_id": "p2", "label": "P2", "qxy": 1.02, "qz": 0.6},
        {"peak_id": "p3", "label": "P3", "qxy": 1.8, "qz": 1.1},
    ]
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)

    pane.suggest_peak_families()
    family_row = next(
        row
        for row in range(pane.family_table.rowCount())
        if {"p1", "p2"}
        <= set(
            pane.family_table.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
        )
    )
    pane.family_table.selectRow(family_row)
    family_id = pane.family_table.item(family_row, 0).text()
    pane.analysis_tabs.setCurrentIndex(1)

    pane._handle_peak_plot_clicked(
        None,
        [FakePoint({"peak_id": "p3"})],
        None,
    )
    family = pane._family_by_id(family_id)
    assert "p3" in family["peak_ids"]
    assert pane.active_family_peak_id == "p3"

    pane._handle_family_plot_clicked(
        None,
        [FakePoint({"peak_id": "p3"})],
        None,
    )
    pane.remove_active_family_ring()

    family = pane._family_by_id(family_id)
    assert "p3" not in family["peak_ids"]
    assert family["manual_edited"] is True


def test_structure_analysis_peak_plot_and_table_selection_sync(qtbot):
    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {"peak_id": "p1", "label": "P1", "qxy": 0.1, "qz": 0.2},
        {"peak_id": "p2", "label": "P2", "qxy": 1.1, "qz": 1.2},
    ]
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)
    if pane.peak_scatter is None:
        pytest.skip("pyqtgraph is unavailable")

    p2_point = next(
        point
        for point in pane.peak_scatter.points()
        if point.data()["peak_id"] == "p2"
    )
    assert p2_point.brush().color() == QtGui.QColor("#22c55e")
    pane._handle_peak_plot_clicked(pane.peak_scatter, [p2_point], None)

    assert pane.active_peak_id == "p2"
    assert (
        pane.peak_table.item(
            pane.peak_table.currentRow(),
            COL_PEAK_ID,
        ).data(QtCore.Qt.ItemDataRole.UserRole)
        == "p2"
    )
    p2_point = next(
        point
        for point in pane.peak_scatter.points()
        if point.data()["peak_id"] == "p2"
    )
    assert p2_point.brush().color() == QtGui.QColor("#2f80ed")

    pane.peak_table.selectRow(0)

    assert pane.active_peak_id == "p1"


def test_structure_analysis_draws_peak_rois_on_plot(qtbot):
    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {
            "peak_id": "p1",
            "label": "P1",
            "qxy": 1.0,
            "qz": 0.2,
            "roi": {
                "kind": "box",
                "qxy_min": 0.8,
                "qxy_max": 1.2,
                "qz_min": 0.0,
                "qz_max": 0.4,
            },
        }
    ]
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)
    if pane.peak_scatter is None:
        pytest.skip("pyqtgraph is unavailable")

    assert len(pane.roi_overlay_items) == 1
    x_values, y_values = pane.roi_overlay_items[0].getData()
    assert min(x_values) == pytest.approx(0.8)
    assert max(x_values) == pytest.approx(1.2)
    assert min(y_values) == pytest.approx(0.0)
    assert max(y_values) == pytest.approx(0.4)


def test_structure_analysis_guess_candidates_shows_progress_dialog(
    qtbot,
    monkeypatch,
):
    from ewald.ui import structure_analysis as structure_ui

    project = ProjectState()
    project.peak_sets["synthetic"] = [
        {"peak_id": "p1", "label": "P1", "qxy": 0.1, "qz": 1.0},
        {"peak_id": "p2", "label": "P2", "qxy": 0.2, "qz": 1.4},
    ]
    orientation = quaternion_from_axis_angle((0.0, 0.0, 1.0), 12.0)
    project.analysis_results.setdefault("crystal_overlays", {})[
        "synthetic"
    ] = {
        "parameters": {
            "crystal_system": "Cubic",
            "a": 6.3,
            "b": 6.3,
            "c": 6.3,
            "orientation_quaternion": list(orientation),
        }
    }
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)

    dialogs = []

    class FakeProgressDialog:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.shown = False
            self.closed = False
            self.deleted = False
            self.value = None
            self.range = (args[2], args[3])
            self.window_title = ""
            self.modality = None
            dialogs.append(self)

        def setWindowTitle(self, title):
            self.window_title = title

        def setWindowModality(self, modality):
            self.modality = modality

        def setCancelButton(self, button):
            self.cancel_button = button

        def setMinimumDuration(self, duration):
            self.minimum_duration = duration

        def setAutoClose(self, enabled):
            self.auto_close = enabled

        def setAutoReset(self, enabled):
            self.auto_reset = enabled

        def show(self):
            self.shown = True

        def setRange(self, minimum, maximum):
            self.range = (minimum, maximum)

        def setValue(self, value):
            self.value = value

        def close(self):
            self.closed = True

        def deleteLater(self):
            self.deleted = True

    def fake_guess_lattice_candidates(peaks, config):
        assert dialogs and dialogs[0].shown
        assert not pane.guess_button.isEnabled()
        assert config.crystal_systems == (
            "Cubic",
            "Tetragonal",
            "Hexagonal",
            "Trigonal",
            "Orthorhombic",
            "Monoclinic",
            "Triclinic",
        )
        assert config.orientation_quaternion == pytest.approx(orientation)
        return [
            LatticeCandidate(
                "candidate_001",
                "Triclinic",
                5.1,
                6.2,
                7.3,
                alpha=82.0,
                beta=91.0,
                gamma=103.0,
                score=0.25,
                matched_count=2,
            )
        ]

    monkeypatch.setattr(
        structure_ui.QtWidgets,
        "QProgressDialog",
        FakeProgressDialog,
    )
    monkeypatch.setattr(
        structure_ui,
        "guess_lattice_candidates",
        fake_guess_lattice_candidates,
    )

    candidates = pane.run_candidate_guessing()

    assert len(candidates) == 1
    assert pane.guess_button.isEnabled()
    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert dialog.args[0] == "Guessing candidate structures..."
    assert dialog.window_title == "Guess Candidates"
    assert dialog.modality == QtCore.Qt.WindowModality.WindowModal
    assert dialog.range == (0, 1)
    assert dialog.value == 1
    assert dialog.closed
    assert dialog.deleted

    headers = [
        pane.candidate_table.horizontalHeaderItem(column).text()
        for column in range(pane.candidate_table.columnCount())
    ]
    assert headers == [
        "Rank",
        "Crystal system",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "Score",
        "Matched",
        "Outliers",
        "Method",
    ]
    assert pane.candidate_table.item(0, 5).text() == "82"
    assert pane.candidate_table.item(0, 6).text() == "91"
    assert pane.candidate_table.item(0, 7).text() == "103"
    assert pane.candidate_table.item(0, 8).text() == "0.25"


def test_structure_analysis_wyckoff_setup_registers_ui_possibilities(qtbot):
    project = ProjectState()
    project.analysis_results.setdefault("structure_analysis", {})[
        "synthetic"
    ] = {
        "peaks": [],
        "families": [],
        "wyckoff": {},
        "candidates": [
            LatticeCandidate(
                "candidate_001",
                "Cubic",
                6.3,
                6.3,
                6.3,
                score=0.1,
            ).as_dict()
        ],
    }
    pane = StructureAnalysisPane(project, "synthetic")
    qtbot.addWidget(pane)

    pane.wyckoff_system_combo.setCurrentText("Cubic")
    assert pane.space_group_combo.count() == 36
    for index in range(pane.space_group_combo.count()):
        if pane.space_group_combo.itemData(index)["number"] == 221:
            pane.space_group_combo.setCurrentIndex(index)
            break
    pane.wyckoff_site_count_spin.setValue(2)

    state = project.analysis_results["structure_analysis"]["synthetic"]
    wyckoff = state["wyckoff"]
    assert wyckoff["space_group"]["number"] == 221
    assert pane.wyckoff_site_table.rowCount() >= 14
    assert wyckoff["registered_combination_count"] == (
        wyckoff_combination_count(221, site_count=2)
    )
    assert wyckoff["registered_assignment_count"] == (
        wyckoff_combination_count(221, site_count=2, ordered=True)
    )
    assert pane.wyckoff_combination_table.rowCount() > 0

    records = pane.generate_candidate_cifs()
    assert records
    assert records[0]["space_group"]["number"] == 221
    assert project.reference_cifs["generated"][records[0]["cif_id"]]
    assert pane.cif_visualizer.cif_id == records[0]["cif_id"]
    assert pane.cif_visualizer.atom_count == 2
    assert pane.cif_visualizer.species_text == "I, Pb"
    assert pane.cif_visualizer.plot_container.hasHeightForWidth()
    assert pane._selected_cif_record()["cif_id"] == records[0]["cif_id"]

    pane.cif_table.setCurrentCell(1, 0)
    pane.cif_table.selectRow(1)
    assert pane.cif_visualizer.cif_id == records[1]["cif_id"]

    group_titles = {
        group.title() for group in pane.findChildren(QtWidgets.QGroupBox)
    }
    assert {
        "Candidate & Symmetry",
        "Composition & Molecules",
        "Wyckoff Registry",
        "Generated CIFs",
    }.issubset(group_titles)


def test_crystal_overlay_hkl_label_modes_default_partial_and_hover(qtbot):
    project = ProjectState()
    pane = PeakIdentificationPane(project, "synthetic")
    qtbot.addWidget(pane)

    pane.crystal_system_combo.setCurrentText("Cubic")
    pane.lattice_a.setValue(10.0)
    pane.h_max.setValue(2)
    pane.k_max.setValue(2)
    pane.l_max.setValue(2)
    pane.positive_qz_check.setChecked(False)
    pane._update_crystal_overlay()

    state = project.analysis_results["crystal_overlays"]["synthetic"]
    partial_count = len(pane.crystal_overlay_graphics)
    all_points = len(pane.crystal_overlay_scatter.points())

    assert pane.hkl_label_mode_combo.currentData() == HKL_LABEL_MODE_PARTIAL
    assert state["hkl_label_mode"] == HKL_LABEL_MODE_PARTIAL
    assert 0 < partial_count < all_points

    pane.hkl_label_mode_combo.setCurrentIndex(
        pane.hkl_label_mode_combo.findData(HKL_LABEL_MODE_ALL)
    )
    pane._update_crystal_overlay()
    assert len(pane.crystal_overlay_graphics) == all_points

    pane.hkl_label_mode_combo.setCurrentIndex(
        pane.hkl_label_mode_combo.findData(HKL_LABEL_MODE_NONE)
    )
    pane._update_crystal_overlay()
    assert pane.crystal_overlay_graphics == []
    assert pane.crystal_overlay_scatter.points()[0].data().startswith("(")
