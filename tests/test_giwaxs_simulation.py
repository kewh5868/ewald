"""Tests for the GIWAXS simulation backend and UI wiring."""

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from ewald.data.models import ImageCorrectionState, ProjectState
from ewald.io.importers import build_data_group_from_paths
from ewald.io.project import load_project
from ewald.simulation.giwaxs import (
    PEAK_TABLE_ATTR,
    SIMULATION_MODE_EWALD_SWEEP,
    EwaldSphereSweepParameters,
    GIWAXSSimulationParameters,
    calculate_giwaxs_peak_rows,
    compare_giwaxs_images,
    load_simulation_data,
    load_structure,
    rank_giwaxs_simulation_fits,
    reconstruct_ewald_sphere_points,
    run_and_store_ewald_sphere_sweep,
    run_and_store_simulation,
    save_giwaxs_comparison_plot,
    simulate_ewald_sphere_sweep,
    simulate_giwaxs_image,
)
from ewald.ui.data_tree import DataTreePane
from ewald.ui.data_viewer import DataViewerPane
from ewald.ui.giwaxs_simulation import (
    STRUCTURE_PREVIEW_MAX_SIDE,
    STRUCTURE_PREVIEW_MIN_SIDE,
    GIWAXSSimulationPane,
    GIWAXSSimulationResultPane,
    GIWAXSSimulationWindow,
    _cpk_color,
    _peak_tip,
)
from ewald.ui.main_window import MainWindow


def test_giwaxs_simulation_backend_runs_and_stores_poscar(tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=32,
        resolution_z=24,
        qxy_min=-3.0,
        qxy_max=3.0,
        qz_min=0.0,
        qz_max=3.0,
    )

    image = simulate_giwaxs_image(poscar, params)

    assert image.dims == ("qz", "qxy")
    assert image.shape == (24, 32)
    assert image.attrs["structure_name"] == "Si_POSCAR"
    assert image.attrs["peak_count"] > 0
    peak_rows = json.loads(image.attrs[PEAK_TABLE_ATTR])
    assert len(peak_rows) == image.attrs["peak_count"]
    assert {"h", "k", "l", "qxy", "qz", "intensity"} <= set(peak_rows[0])
    assert len(calculate_giwaxs_peak_rows(poscar, params)) == len(peak_rows)

    project = ProjectState(name="Simulation Project")
    record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=params,
    )

    assert record["simulation_id"] in project.simulations
    assert Path(record["dataset_uri"]).exists()
    loaded = load_simulation_data(record)
    assert loaded.shape == (24, 32)
    loaded_peak_rows = json.loads(loaded.attrs[PEAK_TABLE_ATTR])
    assert len(loaded_peak_rows) == len(peak_rows)


def test_giwaxs_peak_rows_flag_forbidden_reflections(tmp_path):
    poscar = _write_species_poscar(
        tmp_path / "bcc_Si_POSCAR",
        ["Si"],
        [2],
        [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    )
    params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=32,
        resolution_z=24,
        qxy_min=-4.0,
        qxy_max=4.0,
        qz_min=0.0,
        qz_max=4.0,
    )

    data = simulate_giwaxs_image(poscar, params)
    peak_rows = json.loads(data.attrs[PEAK_TABLE_ATTR])
    forbidden = [row for row in peak_rows if row["forbidden_reflection"]]

    assert forbidden
    assert all(row["excluded_from_indexing"] for row in forbidden)
    assert all(row["amplitude"] == 0.0 for row in forbidden)
    assert data.attrs["forbidden_reflection_count"] == len(forbidden)
    assert data.attrs["peak_count"] == len(peak_rows) - len(forbidden)
    assert any(
        abs(row["h"]) + abs(row["k"]) + abs(row["l"]) == 1 for row in forbidden
    )


def test_giwaxs_missing_wedge_masks_image_and_peak_rows(tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    params = GIWAXSSimulationParameters(
        hkl_extent=2,
        resolution_x=48,
        resolution_z=44,
        qxy_min=-3.0,
        qxy_max=3.0,
        qz_min=0.0,
        qz_max=3.0,
        wavelength_angstrom=1.0,
        incident_angle_deg=8.0,
        missing_wedge_correction=True,
    )

    image = simulate_giwaxs_image(poscar, params)

    horizon_qz = image.attrs["missing_wedge_horizon_qz"]
    low_qz = np.asarray(image.coords["qz"].values) < horizon_qz
    assert image.attrs["missing_wedge_correction_applied"] == 1
    assert image.attrs["missing_wedge_masked_fraction"] > 0.0
    assert np.all(np.asarray(image.values)[low_qz, :] == 0.0)
    peak_rows = json.loads(image.attrs[PEAK_TABLE_ATTR])
    assert peak_rows
    assert min(float(row["qz"]) for row in peak_rows) >= horizon_qz - 1.0e-9


def test_giwaxs_image_comparison_fits_scale_and_writes_plot(tmp_path):
    import xarray as xr

    qxy = np.linspace(-1.0, 1.0, 48)
    qz = np.linspace(0.0, 2.0, 40)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    base = np.exp(
        -0.5 * ((qxy_grid - 0.2) / 0.08) ** 2
        - 0.5 * ((qz_grid - 0.9) / 0.12) ** 2
    )
    experimental = xr.DataArray(
        2.5 * base + 0.1,
        dims=("qz", "qxy"),
        coords={"qz": qz, "qxy": qxy},
    )
    matched = xr.DataArray(
        base,
        dims=("qz", "qxy"),
        coords={"qz": qz, "qxy": qxy},
    )
    shifted = xr.DataArray(
        np.roll(base, 7, axis=1),
        dims=("qz", "qxy"),
        coords={"qz": qz, "qxy": qxy},
    )

    matched_comparison = compare_giwaxs_images(experimental, matched)
    shifted_comparison = compare_giwaxs_images(experimental, shifted)
    plot_path = save_giwaxs_comparison_plot(
        matched_comparison,
        tmp_path / "comparison.png",
        title="Synthetic",
    )

    assert matched_comparison.target.shape == experimental.shape
    assert matched_comparison.experimental is matched_comparison.target
    assert matched_comparison.difference.shape == experimental.shape
    assert matched_comparison.target_label == "Target"
    assert np.nanmax(np.abs(matched_comparison.difference.values)) < 1.0e-8
    assert matched_comparison.metrics["fit_score"] == pytest.approx(
        matched_comparison.metrics["difference_rmse"]
    )
    assert matched_comparison.metrics["difference_rmse"] < 1.0e-8
    assert matched_comparison.metrics["fit_score"] < (
        shifted_comparison.metrics["fit_score"]
    )
    assert matched_comparison.metrics["peak_focus_score"] < (
        shifted_comparison.metrics["peak_focus_score"]
    )
    assert matched_comparison.metrics["peak_overlap_jaccard"] > (
        shifted_comparison.metrics["peak_overlap_jaccard"]
    )
    assert matched_comparison.metrics["correlation"] > 0.99
    assert plot_path.exists()


def test_giwaxs_point_tip_uses_compact_numeric_values():
    tip = _peak_tip(
        x=1.23456789,
        y=0.987654321,
        data={
            "h": 1,
            "k": 2,
            "l": 3,
            "qxy": 1.23456789,
            "qz": 0.987654321,
            "intensity": 12345.6789,
            "forbidden_reflection": True,
        },
    )

    assert "(1, 2, 3)" in tip
    assert "1.23" in tip
    assert "0.988" in tip
    assert "1.23e+04" in tip
    assert "1.23456789" not in tip
    assert "0.987654321" not in tip
    assert "12345.6789" not in tip
    assert "Forbidden reflection" in tip


def test_giwaxs_result_table_marks_forbidden_reflections(qtbot):
    pane = GIWAXSSimulationResultPane(ProjectState(name="Forbidden UI"))
    qtbot.addWidget(pane)

    pane._set_peak_rows(
        [
            {
                "h": 1,
                "k": 0,
                "l": 0,
                "qxy": 1.0,
                "qz": 0.0,
                "intensity": 1.0e-16,
                "amplitude": 0.0,
                "forbidden_reflection": True,
                "excluded_from_indexing": True,
                "reflection_status": "forbidden",
            },
            {
                "h": 1,
                "k": 1,
                "l": 0,
                "qxy": 1.5,
                "qz": 0.2,
                "intensity": 4.0,
                "amplitude": 4.0,
            },
        ]
    )

    assert pane.hkl_table.item(0, 4).text() == "Forbidden"
    assert pane.hkl_table.item(1, 4).text() == "Indexable"
    assert "1 indexable point" in pane.hkl_count_label.text()
    assert "1 forbidden" in pane.hkl_count_label.text()


def test_rank_giwaxs_simulation_fits_orders_candidate_parameters(tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    good_params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=28,
        resolution_z=22,
        theta_x_deg=90.0,
        theta_y_deg=0.0,
    )
    poor_params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=28,
        resolution_z=22,
        theta_x_deg=45.0,
        theta_y_deg=30.0,
    )
    experimental = simulate_giwaxs_image(poscar, good_params)
    comparison = compare_giwaxs_images(
        experimental,
        simulate_giwaxs_image(poscar, poor_params),
    )

    ranked = rank_giwaxs_simulation_fits(
        experimental,
        [("good_structure", poscar)],
        [poor_params, good_params],
    )

    assert comparison.target_label == "Simulated target"
    assert [record.rank for record in ranked] == [1, 2]
    assert ranked[0].parameters["theta_x_deg"] == pytest.approx(90.0)
    assert ranked[0].metrics["fit_score"] < ranked[1].metrics["fit_score"]
    assert ranked[0].as_dict()["structure_name"] == "good_structure"


@pytest.mark.parametrize(
    "warning_message",
    [
        (
            "Issues encountered while parsing CIF: 16 fractional coordinates "
            "rounded to ideal values to avoid issues with finite precision."
        ),
        (
            "No _symmetry_equiv_pos_as_xyz type key found. Spacegroup from "
            "_symmetry_space_group_name_H-M used."
        ),
        (
            "Issues encountered while parsing CIF: No "
            "_symmetry_equiv_pos_as_xyz type key found. Spacegroup from "
            "_symmetry_space_group_name_H-M used."
        ),
    ],
)
def test_load_structure_suppresses_known_pymatgen_cif_warnings(
    monkeypatch, tmp_path, warning_message
):
    from pymatgen.core import Structure

    class FakeStructure(list):
        lattice = SimpleNamespace(matrix=np.eye(3))

    def fake_from_file(path):
        warnings.warn(
            warning_message,
            UserWarning,
            stacklevel=2,
        )
        return FakeStructure(
            [
                SimpleNamespace(
                    specie=SimpleNamespace(symbol="Si"),
                    frac_coords=np.array([0.0, 0.0, 0.0]),
                )
            ]
        )

    monkeypatch.setattr(Structure, "from_file", staticmethod(fake_from_file))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = load_structure(tmp_path / "rounded.cif")

    assert caught == []
    assert structure.species == ["Si"]


def test_giwaxs_simulation_backend_links_data_file(tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Simulation Project")
    data_file = project.add_data_file("sample_001.tiff")

    record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=GIWAXSSimulationParameters(
            hkl_extent=1,
            resolution_x=24,
            resolution_z=20,
        ),
        target_data_id=data_file.data_id,
    )

    assert record["data_id"] == data_file.data_id
    assert project.simulations_for_data_file(data_file.data_id) == [
        (record["simulation_id"], record)
    ]

    unlinked = project.link_simulation_to_data_file(
        record["simulation_id"],
        None,
    )

    assert "data_id" not in unlinked
    assert project.simulations_for_data_file(data_file.data_id) == []


def test_ewald_sphere_sweep_backend_stores_and_reconstructs(tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    params = EwaldSphereSweepParameters(
        hkl_extent=1,
        resolution_x=18,
        resolution_z=14,
        theta_x_min_deg=0.0,
        theta_x_max_deg=10.0,
        theta_x_step_deg=10.0,
        theta_y_min_deg=0.0,
        theta_y_max_deg=20.0,
        theta_y_step_deg=20.0,
    )

    sweep = simulate_ewald_sphere_sweep(poscar, params)

    assert sweep.dims == ("theta_y", "theta_x", "qz", "qxy")
    assert sweep.shape == (2, 2, 14, 18)
    assert sweep.attrs["simulation_mode"] == SIMULATION_MODE_EWALD_SWEEP

    project = ProjectState(name="Sweep Project")
    record = run_and_store_ewald_sphere_sweep(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=params,
    )
    loaded = load_simulation_data(record)
    points, intensities = reconstruct_ewald_sphere_points(
        loaded,
        intensity_quantile=0.9,
        max_points=200,
    )

    assert record["simulation_mode"] == SIMULATION_MODE_EWALD_SWEEP
    assert loaded.shape == (2, 2, 14, 18)
    assert points.shape[1] == 3
    assert len(points) == len(intensities)
    assert len(points) > 0


def test_simulation_window_runs_multiple_project_simulations(qtbot, tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Linked Project")
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    structure_id = window.import_structure_path(poscar)
    window.hkl_extent.setValue(1)
    window.resolution_x.setValue(24)
    window.resolution_z.setValue(20)

    first = window.run_selected_simulation()
    window.theta_y.setValue(5.0)
    second = window.run_selected_simulation()

    assert first is not None
    assert second is not None
    assert first["simulation_id"] != second["simulation_id"]
    assert len(project.simulations) == 2
    assert len(window.structures[structure_id]["simulation_ids"]) == 2
    assert window.result_pane.simulation_id == second["simulation_id"]
    assert window.result_pane.peak_rows
    assert window.result_pane.hkl_table.rowCount() == len(
        window.result_pane.peak_rows
    )
    assert window.result_pane.hkl_table.item(0, 0).text().startswith("(")
    if window.result_pane.hkl_scatter is not None:
        assert window.result_pane.hkl_scatter.isVisible()
        window.result_pane.show_hkl_points.setChecked(False)
        assert not window.result_pane.hkl_scatter.isVisible()


def test_simulation_window_recalls_project_loaded_cifs(qtbot, tmp_path):
    from pymatgen.core import Lattice, Structure

    cif_path = tmp_path / "reference_tetragonal.cif"
    structure = Structure(
        Lattice.from_parameters(4.2, 4.2, 7.1, 90.0, 90.0, 90.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )
    cif_text = structure.to(fmt="cif")
    cif_path.write_text(cif_text, encoding="utf-8")
    project = ProjectState(name="Loaded CIF Project")
    project.remember_loaded_cif(
        cif_path,
        cif_text=cif_text,
        lattice={
            "a": 4.2,
            "b": 4.2,
            "c": 7.1,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        crystal_system="Tetragonal",
    )

    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)

    entries = [
        window.import_structure_combo.itemText(index)
        for index in range(window.import_structure_combo.count())
    ]
    assert str(cif_path) in entries

    window.import_structure_combo.setCurrentText(str(cif_path))
    structure_id = window.import_structure_from_field()

    assert structure_id is not None
    assert structure_id in window.structures
    assert window.structures[structure_id]["metadata"]["atom_count"] == 1


def test_simulation_window_reuses_cached_larger_pattern(qtbot, tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Cache Project")
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.import_structure_path(poscar)
    window.hkl_extent.setValue(2)
    window.resolution_x.setValue(28)
    window.resolution_z.setValue(22)

    first = window.run_selected_simulation()
    window.hkl_extent.setValue(1)
    window.resolution_x.setValue(20)
    window.resolution_z.setValue(18)
    reused = window.run_selected_simulation()

    assert first is not None
    assert reused is first
    assert len(project.simulations) == 1
    assert window.result_pane.simulation_id == first["simulation_id"]
    assert "Cached result displayed" in window.cache_status_label.text()


def test_simulation_window_rotation_buttons_restore_cached_orientation(
    qtbot, tmp_path
):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Rotation Cache Project")
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.import_structure_path(poscar)
    window.hkl_extent.setValue(1)
    window.resolution_x.setValue(20)
    window.resolution_z.setValue(18)

    first = window.run_selected_simulation()
    window.rotation_increment.setValue(5.0)
    window.rotate_x_neg_button.click()
    second = window.run_selected_simulation()
    window.rotate_x_pos_button.click()

    assert first is not None
    assert second is not None
    assert second["simulation_id"] != first["simulation_id"]
    assert window.theta_x.value() == 90.0
    assert window.result_pane.simulation_id == first["simulation_id"]
    assert "Cached result displayed" in window.cache_status_label.text()


def test_simulation_window_runs_sweep_with_video_playback(qtbot, tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Sweep Window Project")
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.import_structure_path(poscar)
    window.hkl_extent.setValue(1)
    window.resolution_x.setValue(18)
    window.resolution_z.setValue(14)
    window.simulation_mode.setCurrentIndex(
        window.simulation_mode.findData(SIMULATION_MODE_EWALD_SWEEP)
    )
    window.sweep_theta_x_min.setValue(0.0)
    window.sweep_theta_x_max.setValue(10.0)
    window.sweep_theta_x_step.setValue(10.0)
    window.sweep_theta_y_min.setValue(0.0)
    window.sweep_theta_y_max.setValue(20.0)
    window.sweep_theta_y_step.setValue(20.0)

    record = window.run_selected_mode()

    assert record is not None
    assert record["simulation_mode"] == SIMULATION_MODE_EWALD_SWEEP
    assert window.result_pane.sweep_data is not None
    assert window.result_pane.frame_slider.maximum() == 3
    assert window.result_pane.playback_controls.isVisibleTo(window)
    assert window.open_reconstruction_action.isEnabled()


def test_sweep_video_animates_structure_and_stop_restores_selection(
    qtbot, tmp_path
):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    window = GIWAXSSimulationWindow(
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.import_structure_path(poscar)
    window.hkl_extent.setValue(1)
    window.resolution_x.setValue(18)
    window.resolution_z.setValue(14)
    window.simulation_mode.setCurrentIndex(
        window.simulation_mode.findData(SIMULATION_MODE_EWALD_SWEEP)
    )
    window.sweep_theta_x_min.setValue(0.0)
    window.sweep_theta_x_max.setValue(10.0)
    window.sweep_theta_x_step.setValue(10.0)
    window.sweep_theta_y_min.setValue(0.0)
    window.sweep_theta_y_max.setValue(20.0)
    window.sweep_theta_y_step.setValue(20.0)
    window.run_selected_mode()

    selected = (window.theta_x.value(), window.theta_y.value())
    window.result_pane.play_button.click()
    window.result_pane._set_sweep_frame(3)
    window.result_pane.play_button.click()

    assert window.result_pane.stop_button.isEnabled()
    assert window.structure_viewer.theta_x_deg == 10.0
    assert window.structure_viewer.theta_y_deg == 20.0
    assert (window.theta_x.value(), window.theta_y.value()) == selected

    window.result_pane.stop_button.click()

    assert window.structure_viewer.theta_x_deg == selected[0]
    assert window.structure_viewer.theta_y_deg == selected[1]
    assert (window.theta_x.value(), window.theta_y.value()) == selected


def test_simulation_window_links_runs_to_selected_data_file(qtbot, tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Linked Project")
    data_file = project.add_data_file("measurement_001.tiff")
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        initial_data_id=data_file.data_id,
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.import_structure_path(poscar)
    window.hkl_extent.setValue(1)
    window.resolution_x.setValue(24)
    window.resolution_z.setValue(20)

    record = window.run_selected_simulation()

    assert record is not None
    assert record["data_id"] == data_file.data_id
    assert window.selected_data_id() == data_file.data_id

    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    file_item = _child_with_text(pane.tree.topLevelItem(0), data_file.name)
    linked_item = _child_with_text(file_item, "Linked Simulations")
    simulation_item = _child_with_text(linked_item, record["structure_name"])

    assert linked_item.text(1) == "1"
    assert (
        _child_with_text(simulation_item, "Data id").text(1)
        == data_file.data_id
    )
    assert (
        _child_with_text(simulation_item, "Linked data file").text(1)
        == data_file.name
    )


def test_simulation_window_compares_displayed_result_to_qspace_product(
    qtbot,
    tmp_path,
):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Comparison Project")
    data_file = project.add_data_file("measurement_001.tiff")
    record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=GIWAXSSimulationParameters(
            hkl_extent=1,
            resolution_x=24,
            resolution_z=20,
        ),
        target_data_id=data_file.data_id,
    )
    simulated = load_simulation_data(record)
    qspace_path = tmp_path / "experimental_qspace.npz"
    np.savez(
        qspace_path,
        intensity=np.asarray(simulated.values, dtype=float) * 2.0,
        q_ip=np.asarray(simulated.coords["qxy"].values, dtype=float),
        q_oop=np.asarray(simulated.coords["qz"].values, dtype=float),
    )
    project.processed_products[data_file.data_id] = str(qspace_path)
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        initial_data_id=data_file.data_id,
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    simulation_item = _child_with_text(
        window.tree.topLevelItem(0).child(0),
        record["simulation_id"],
    )
    window.tree.setCurrentItem(simulation_item)

    comparison = window.compare_displayed_simulation_to_target()

    assert comparison is not None
    assert comparison.target_label == "Experimental data"
    assert comparison.metrics["correlation"] > 0.99
    assert window.comparison_pane.comparison is comparison
    assert window.right_tabs.currentWidget() is window.comparison_pane
    assert (
        "fit_score"
        in project.simulations[record["simulation_id"]]["fit_metrics"]
    )


def test_simulation_window_ranks_stored_simulations_against_qspace_product(
    qtbot,
    tmp_path,
):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Rank Project")
    data_file = project.add_data_file("measurement_001.tiff")
    good_params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=28,
        resolution_z=22,
        theta_x_deg=90.0,
        theta_y_deg=0.0,
    )
    poor_params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=28,
        resolution_z=22,
        theta_x_deg=35.0,
        theta_y_deg=20.0,
    )
    good_record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=good_params,
        target_data_id=data_file.data_id,
    )
    poor_record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=poor_params,
        target_data_id=data_file.data_id,
    )
    experimental = load_simulation_data(good_record)
    qspace_path = tmp_path / "experimental_qspace.npz"
    np.savez(
        qspace_path,
        intensity=np.asarray(experimental.values, dtype=float),
        q_ip=np.asarray(experimental.coords["qxy"].values, dtype=float),
        q_oop=np.asarray(experimental.coords["qz"].values, dtype=float),
    )
    project.processed_products[data_file.data_id] = str(qspace_path)
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        initial_data_id=data_file.data_id,
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)

    ranked = window.rank_stored_simulations_against_target()

    assert [item["simulation_id"] for item in ranked] == [
        good_record["simulation_id"],
        poor_record["simulation_id"],
    ]
    assert project.simulations[good_record["simulation_id"]]["fit_rank"] == 1
    assert project.simulations[poor_record["simulation_id"]]["fit_rank"] == 2
    assert window.result_pane.simulation_id == good_record["simulation_id"]
    assert window.comparison_pane.comparison is ranked[0]["comparison"]
    assert window.right_tabs.currentWidget() is window.comparison_pane


def test_simulation_window_runs_generated_cif_difference_comparisons(
    qtbot,
    tmp_path,
):
    from pymatgen.core import Lattice, Structure

    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    params = GIWAXSSimulationParameters(
        hkl_extent=1,
        resolution_x=28,
        resolution_z=22,
        theta_x_deg=90.0,
        theta_y_deg=0.0,
    )
    experimental = simulate_giwaxs_image(poscar, params)
    qspace_path = tmp_path / "experimental_qspace.npz"
    np.savez(
        qspace_path,
        intensity=np.asarray(experimental.values, dtype=float),
        q_ip=np.asarray(experimental.coords["qxy"].values, dtype=float),
        q_oop=np.asarray(experimental.coords["qz"].values, dtype=float),
    )
    project = ProjectState(name="Generated CIF Compare Project")
    data_file = project.add_data_file("measurement_001.tiff")
    project.processed_products[data_file.data_id] = str(qspace_path)
    generated_structure = Structure(
        Lattice.cubic(3.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )
    decoy_structure = Structure(
        Lattice.cubic(4.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )
    cif_id = "candidate_001_cif_01"
    decoy_cif_id = "candidate_001_cif_02"
    project.reference_cifs["generated"] = {
        cif_id: {
            "cif_id": cif_id,
            "candidate_id": "candidate_001",
            "rank": 1,
            "score": 0.1,
            "data_id": data_file.data_id,
            "cif_text": generated_structure.to(fmt="cif"),
        },
        decoy_cif_id: {
            "cif_id": decoy_cif_id,
            "candidate_id": "candidate_001",
            "rank": 2,
            "score": 0.2,
            "data_id": data_file.data_id,
            "cif_text": decoy_structure.to(fmt="cif"),
        },
    }
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        initial_data_id=data_file.data_id,
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.hkl_extent.setValue(params.hkl_extent)
    window.resolution_x.setValue(params.resolution_x)
    window.resolution_z.setValue(params.resolution_z)

    ranked = window.run_top_generated_cif_fit_comparisons(limit=2)

    assert len(ranked) == 2
    assert ranked[0]["record"]["generated_cif_id"] == cif_id
    assert ranked[0]["record"]["fit_rank"] == 1
    assert ranked[0]["comparison"].metrics["difference_rmse"] < 1.0e-8
    assert ranked[1]["comparison"].metrics["difference_rmse"] > (
        ranked[0]["comparison"].metrics["difference_rmse"]
    )
    assert window.comparison_pane.comparison is ranked[0]["comparison"]
    assert window.comparison_pane.ranked_table.rowCount() == 2
    window.comparison_pane.ranked_table.selectRow(1)
    assert window.comparison_pane.comparison is ranked[1]["comparison"]
    assert window.right_tabs.currentWidget() is window.comparison_pane
    generated_path = Path(project.reference_cifs["generated"][cif_id]["path"])
    assert generated_path.exists()
    assert ranked[0]["record"]["cif_path"] == str(generated_path)

    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    simulations_item = _child_with_text(
        pane.tree.topLevelItem(0), "GIWAXS Simulations"
    )
    simulation_item = _child_with_text(
        simulations_item, ranked[0]["record"]["structure_name"]
    )
    assert _child_with_text(simulation_item, "CIF path").text(1) == str(
        generated_path
    )


def test_main_giwaxs_tab_embeds_generated_structure_workflow(
    qtbot,
    tmp_path,
    repo_root,
):
    from pymatgen.core import Lattice, Structure

    data_path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([data_path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState(name="Embedded Simulation Project")
    project.add_data_group(group)
    project.set_image_corrections(
        ImageCorrectionState(target_id=data_id, confirmed=True)
    )

    cif_id = "candidate_001_cif_01"
    project.reference_cifs["generated"] = {
        cif_id: {
            "cif_id": cif_id,
            "candidate_id": "candidate_001",
            "rank": 1,
            "score": 0.1,
            "data_id": data_id,
            "cif_text": Structure(
                Lattice.cubic(3.0),
                ["Si"],
                [[0.0, 0.0, 0.0]],
            ).to(fmt="cif"),
        }
    }
    window = MainWindow(
        project=project,
        project_path=tmp_path / "embedded_project.ewld",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)

    tab_index = [
        window.tabs.tabText(index) for index in range(window.tabs.count())
    ].index("GIWAXS Simulation")
    pane = window.tabs.widget(tab_index)
    assert isinstance(pane, GIWAXSSimulationPane)
    assert pane.selected_data_id() == data_id
    assert any(
        structure["metadata"].get("generated_cif_id") == cif_id
        for structure in pane.structures.values()
    )

    pane.output_directory = tmp_path / "simulations"
    pane.hkl_extent.setValue(1)
    pane.resolution_x.setValue(24)
    pane.resolution_z.setValue(20)
    record = pane.run_selected_simulation()

    assert record is not None
    assert record["data_id"] == data_id
    assert record["generated_cif_id"] == cif_id
    assert Path(record["cif_path"]).exists()

    file_item = _child_with_text(
        window.data_tree.tree.topLevelItem(0),
        group.data_files[0].name,
    )
    linked_item = _child_with_text(file_item, "Linked Simulations")
    assert linked_item.text(1) == "1"


def test_simulation_window_relinks_selected_simulation(qtbot, tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Linked Project")
    first_data = project.add_data_file("first_sample.tiff")
    second_data = project.add_data_file("second_sample.tiff")
    record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=GIWAXSSimulationParameters(
            hkl_extent=1,
            resolution_x=24,
            resolution_z=20,
        ),
        target_data_id=first_data.data_id,
    )
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    structure_item = window.tree.topLevelItem(0).child(0)
    simulation_item = _child_with_text(structure_item, record["simulation_id"])
    window.tree.setCurrentItem(simulation_item)
    window.set_target_data_id(second_data.data_id)

    linked = window.link_selected_simulation_to_data_file()

    assert linked is not None
    assert linked["data_id"] == second_data.data_id
    assert project.simulations_for_data_file(first_data.data_id) == []
    assert project.simulations_for_data_file(second_data.data_id) == [
        (record["simulation_id"], record)
    ]


def test_simulation_window_shows_active_unit_cell_structure(qtbot, tmp_path):
    first_poscar = _write_poscar(tmp_path / "Si_POSCAR")
    second_poscar = _write_species_poscar(
        tmp_path / "LiCoO2_POSCAR",
        ["Li", "Co", "O"],
        [1, 1, 2],
        [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.25, 0.25, 0.25),
            (0.75, 0.75, 0.75),
        ],
    )
    window = GIWAXSSimulationWindow(
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)

    first_structure_id = window.import_structure_path(first_poscar)
    second_structure_id = window.import_structure_path(second_poscar)

    assert window.structure_viewer.structure_id == second_structure_id
    assert window.structure_viewer.atom_count == 4
    assert window.structure_viewer.species_text == "Co, Li, O"
    assert "Loaded: LiCoO2_POSCAR" in window.structure_viewer.info_label.text()
    assert _cpk_color("O") == "#ff0d0d"
    assert _cpk_color("co") == "#f090a0"

    root = window.tree.topLevelItem(0)
    window.tree.setCurrentItem(_child_with_text(root, "Si_POSCAR"))

    assert window.structure_viewer.structure_id == first_structure_id
    assert window.structure_viewer.atom_count == 1
    assert window.structure_viewer.species_text == "Si"


def test_simulation_window_has_scrollable_compact_inputs_and_presets(
    qtbot, tmp_path
):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    window = GIWAXSSimulationWindow(
        output_directory=tmp_path / "simulations",
        settings=_settings(tmp_path),
    )
    qtbot.addWidget(window)
    window.import_structure_path(poscar)

    assert isinstance(window.left_scroll, QtWidgets.QScrollArea)
    assert window.run_bar.parentWidget() is not window.left_scroll_content
    scroll_layout = window.left_scroll_content.layout()
    assert scroll_layout.itemAt(0).widget() is window.controls
    assert scroll_layout.itemAt(1).widget() is window.structure_viewer
    assert (
        scroll_layout.itemAt(2).widget()
        is window.orientation_distribution_view
    )
    controls_layout = window.controls.layout()
    assert window.sweep_controls.isVisibleTo(window)
    assert window.sweep_controls.title() == "Ewald sphere sweep"
    assert controls_layout.itemAt(2).widget() is window.sweep_controls
    assert window.structure_viewer.plot_container.hasHeightForWidth()
    assert (
        window.structure_viewer.plot_container.heightForWidth(430)
        == STRUCTURE_PREVIEW_MAX_SIDE
    )
    assert (
        window.structure_viewer.plot_widget.minimumHeight()
        == STRUCTURE_PREVIEW_MIN_SIDE
    )

    window.orientation_distribution_view.auto_update_check.setChecked(False)
    isotropic_index = window.preset_combo.findData("isotropic")
    window.preset_combo.setCurrentIndex(isotropic_index)
    window.apply_preset_button.click()

    assert window.theta_x.value() == 0.0
    assert window.theta_y.value() == 0.0
    assert np.isclose(window.sigma_phi.value(), np.pi, atol=1.0e-5)
    assert window.orientation_distribution_view._pending_parameters is not None

    window.orientation_distribution_view.refresh_button.click()

    assert window.orientation_distribution_view._pending_parameters is None
    status = window.orientation_distribution_view.status_label.text()
    assert "crystallites" in status or "unavailable" in status


def test_simulation_result_pane_styles_independently_from_data_viewer(
    qtbot, tmp_path, repo_root
):
    data_path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([data_path], group_name="Example")
    data_id = group.data_files[0].data_id
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Style Project")
    project.add_data_group(group)
    record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=GIWAXSSimulationParameters(
            hkl_extent=1,
            resolution_x=24,
            resolution_z=20,
        ),
    )

    data_viewer = DataViewerPane(project, data_id)
    simulation_viewer = GIWAXSSimulationResultPane(
        project,
        record["simulation_id"],
    )
    qtbot.addWidget(data_viewer)
    qtbot.addWidget(simulation_viewer)

    data_viewer.quantile_low.setValue(10.0)
    simulation_viewer.quantile_low.setValue(5.0)
    simulation_viewer.quantile_high.setValue(95.0)
    assert simulation_viewer.image_data is not None
    finite = simulation_viewer.image_data[
        np.isfinite(simulation_viewer.image_data)
    ]
    expected_levels = np.nanquantile(finite, [0.05, 0.95])

    assert simulation_viewer.colormap_combo.findData("magma") >= 0
    assert simulation_viewer.quantile_check.isChecked()
    assert simulation_viewer.zoom_fit_button.text() == "Autoscale"
    for button in (
        simulation_viewer.zoom_in_button,
        simulation_viewer.zoom_out_button,
        simulation_viewer.zoom_fit_button,
        simulation_viewer.pan_button,
    ):
        assert button.parentWidget() is simulation_viewer.plot_toolbar
    assert data_viewer.quantile_low.value() == 10.0
    assert simulation_viewer.quantile_low.value() == 5.0
    if simulation_viewer.image_item is not None:
        expected_image_levels = expected_levels
        if expected_image_levels[0] == expected_image_levels[1]:
            expected_image_levels = (
                expected_image_levels[0],
                expected_image_levels[1] + 1.0,
            )
        assert np.allclose(
            simulation_viewer.image_item.getLevels(),
            expected_image_levels,
            rtol=1.0e-4,
            atol=1.0e-4,
        )

    simulation_viewer.auto_contrast_button.click()

    assert np.allclose(
        [
            simulation_viewer.level_min.value(),
            simulation_viewer.level_max.value(),
        ],
        expected_levels,
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_simulation_window_remembers_imported_structure_paths(qtbot, tmp_path):
    settings = _settings(tmp_path)
    first_poscar = _write_poscar(tmp_path / "Si_POSCAR")
    second_poscar = _write_poscar(tmp_path / "Ge_POSCAR")
    window = GIWAXSSimulationWindow(
        output_directory=tmp_path / "simulations",
        settings=settings,
    )
    qtbot.addWidget(window)

    first_structure_id = window.import_structure_path(first_poscar)
    window.import_structure_path(second_poscar)

    assert _combo_items(window.import_structure_combo) == [
        str(second_poscar),
        str(first_poscar),
    ]
    assert window.import_structure_combo.currentText() == str(second_poscar)

    restored_settings = _settings(tmp_path)
    restored_window = GIWAXSSimulationWindow(
        output_directory=tmp_path / "restored_simulations",
        settings=restored_settings,
    )
    qtbot.addWidget(restored_window)
    restored_window.import_structure_combo.setCurrentText(str(first_poscar))

    assert _combo_items(restored_window.import_structure_combo) == [
        str(second_poscar),
        str(first_poscar),
    ]
    assert restored_window.import_structure_from_field() == first_structure_id
    assert first_structure_id in restored_window.structures


def test_project_tree_simulation_selection_opens_main_simulation_tab(
    qtbot, tmp_path
):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Tree Project")
    record = run_and_store_simulation(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=GIWAXSSimulationParameters(
            hkl_extent=1,
            resolution_x=24,
            resolution_z=20,
        ),
    )
    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    simulations_item = _child_with_text(
        pane.tree.topLevelItem(0), "GIWAXS Simulations"
    )
    simulation_item = simulations_item.child(0)

    assert simulation_item.text(1) == "GIWAXS simulation"
    assert _child_with_text(simulation_item, "Dataset uri").text(1)

    window = MainWindow(project=project)
    qtbot.addWidget(window)
    window.data_tree.tree.setCurrentItem(
        _child_with_text(
            _child_with_text(
                window.data_tree.tree.topLevelItem(0), "GIWAXS Simulations"
            ),
            record["structure_name"],
        )
    )

    assert window.tabs.count() == 1
    assert window.tabs.tabText(0) == "GIWAXS Simulation"
    assert window.tabs.widget(0).simulation_id == record["simulation_id"]
    assert window.giwaxs_simulation_action.isEnabled()


def test_project_tree_labels_ewald_sphere_sweeps(qtbot, tmp_path):
    poscar = _write_poscar(tmp_path / "Si_POSCAR")
    project = ProjectState(name="Sweep Tree Project")
    record = run_and_store_ewald_sphere_sweep(
        project,
        poscar,
        tmp_path / "simulations",
        parameters=EwaldSphereSweepParameters(
            hkl_extent=1,
            resolution_x=18,
            resolution_z=14,
            theta_x_min_deg=0.0,
            theta_x_max_deg=0.0,
            theta_y_min_deg=0.0,
            theta_y_max_deg=0.0,
        ),
    )
    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)
    simulations_item = _child_with_text(
        pane.tree.topLevelItem(0), "GIWAXS Simulations"
    )
    simulation_item = simulations_item.child(0)

    assert record["simulation_id"] in project.simulations
    assert simulation_item.text(1) == "Ewald sphere sweep"


def test_main_window_runs_cif_simulation_without_data_and_saves_project(
    qtbot,
    tmp_path,
    monkeypatch,
):
    from pymatgen.core import Lattice, Structure

    cif_path = tmp_path / "standalone_structure.cif"
    cif_path.write_text(
        Structure(
            Lattice.cubic(3.0),
            ["Si"],
            [[0.0, 0.0, 0.0]],
        ).to(fmt="cif"),
        encoding="utf-8",
    )
    saved_path = tmp_path / "cif_only_project.ewld"
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.project_active
    assert window.giwaxs_simulation_action.isEnabled()

    window.open_giwaxs_simulation_tool()
    simulation_window = window.giwaxs_simulation_window
    simulation_window.output_directory = tmp_path / "simulations"
    simulation_window.import_structure_path(cif_path)
    loaded_cifs = window.project.reference_cifs.get("loaded", {})
    assert len(loaded_cifs) == 1
    loaded_cif_id = next(iter(loaded_cifs))
    uploaded_cif = loaded_cifs[loaded_cif_id]
    assert uploaded_cif["lattice"]["a"] == pytest.approx(3.0)
    assert uploaded_cif["crystal_system"] == "Cubic"
    assert window.project_active
    assert window.save_project_action.isEnabled()
    simulation_window.hkl_extent.setValue(1)
    simulation_window.resolution_x.setValue(24)
    simulation_window.resolution_z.setValue(20)

    record = simulation_window.run_selected_simulation()

    assert record is not None
    assert record["data_id"] is None
    assert record["cif_path"] == str(cif_path)
    assert record["loaded_cif_id"] == loaded_cif_id
    assert window.project_active
    assert window.save_project_action.isEnabled()
    assert _project_data_count(window.project) == 0

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            str(saved_path),
            "EWALD Projects (*.ewld *.ewald.json)",
        ),
    )

    assert window.save_project_as()

    cif_path.unlink()
    loaded = load_project(saved_path)
    loaded_record = loaded.simulations[record["simulation_id"]]
    loaded_cif = loaded.reference_cifs["loaded"][loaded_cif_id]
    assert _project_data_count(loaded) == 0
    assert loaded_record["data_id"] is None
    assert loaded_record["loaded_cif_id"] == loaded_cif_id
    assert Path(loaded_cif["local_path"]).exists()
    assert Path(loaded_cif["local_path"]).read_text(encoding="utf-8")
    assert loaded_record["cif_path"] == loaded_cif["local_path"]
    assert loaded_record["structure_path"] == loaded_cif["local_path"]
    assert Path(loaded_record["dataset_uri"]).exists()
    assert saved_path.with_suffix(".ewald.json").exists()
    readable = load_project(saved_path.with_suffix(".ewald.json"))
    readable_record = readable.simulations[record["simulation_id"]]
    readable_cif = readable.reference_cifs["loaded"][loaded_cif_id]
    assert readable_record["cif_path"] == readable_cif["local_path"]
    assert Path(readable_record["cif_path"]).exists()


def _write_poscar(path: Path) -> Path:
    return _write_species_poscar(
        path,
        ["Si"],
        [1],
        [(0.0, 0.0, 0.0)],
    )


def _write_species_poscar(
    path: Path,
    species: list[str],
    counts: list[int],
    coords: list[tuple[float, float, float]],
) -> Path:
    path.write_text(
        "\n".join(
            [
                path.stem,
                "1.0",
                "3.0 0.0 0.0",
                "0.0 3.0 0.0",
                "0.0 0.0 3.0",
                " ".join(species),
                " ".join(str(count) for count in counts),
                "Direct",
                *["{:.6g} {:.6g} {:.6g}".format(*coord) for coord in coords],
            ]
        ),
        encoding="utf-8",
    )
    return path


def _settings(tmp_path: Path) -> QtCore.QSettings:
    return QtCore.QSettings(
        str(tmp_path / "settings.ini"),
        QtCore.QSettings.Format.IniFormat,
    )


def _combo_items(combo) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _project_data_count(project: ProjectState) -> int:
    return len(project.data_files) + sum(
        len(group.data_files) for group in project.data_groups
    )


def _child_with_text(parent, text):
    for index in range(parent.childCount()):
        child = parent.child(index)
        if child.text(0) == text:
            return child
    raise AssertionError(f"No child named {text!r}")
