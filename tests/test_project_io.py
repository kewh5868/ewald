"""Project model and .ewld archive tests."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from ewald.data.models import (
    PEAK_POINT_KIND_COMMITTED,
    PEAK_POINT_KIND_GAP_ESTIMATED,
    ImageCorrectionState,
    ProcessingRecord,
    ProjectState,
    ROIRegion,
    roi_hkl_metadata,
    roi_pole_figure_status,
)
from ewald.io.importers import (
    build_data_group_from_folder,
    build_data_group_from_paths,
)
from ewald.io.project import (
    READABLE_PROJECT_EXTENSION,
    load_project,
    save_project,
)


def test_project_round_trip(tmp_path):
    project = ProjectState(name="Round Trip")
    project.add_data_file("detector_001.tiff", incident_angle_deg=0.3)
    project.peak_sets["detector_001"] = [
        {"qx": 1.0, "qz": 0.5, "intensity": 42.0}
    ]
    project.remember_film_material("CH3NH3PbI3", 4.16)

    path = save_project(project, tmp_path / "round_trip")
    loaded = load_project(path)

    assert path.suffix == ".ewld"
    assert loaded.name == "Round Trip"
    assert loaded.data_files[0].path.name == "detector_001.tiff"
    assert loaded.peak_sets["detector_001"][0]["intensity"] == 42.0
    assert loaded.film_material_memory[0]["stoichiometry"] == "CH3NH3PbI3"
    assert loaded.film_material_memory[0]["density_g_cm3"] == 4.16


def test_project_round_trip_preserves_data_groups(repo_root, tmp_path):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState(name="Grouped")
    project.add_data_group(group)
    data_id = group.data_files[0].data_id
    project.integration_regions[data_id] = [
        {"name": "fit-window", "q_ip_min": 0.1, "q_ip_max": 0.4}
    ]
    project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="box",
            name="Fit window",
            qxy_min=0.1,
            qxy_max=0.4,
            qz_min=0.2,
            qz_max=0.7,
            integration_axis="qxy",
            integration_direction="horizontal",
        )
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            xray_energy_kev=12.7,
            image_rotation_deg=270,
            image_mirrored_y=True,
            pyfai_sample_orientation=7,
            correct_solid_angle=False,
            polarization_factor=None,
            normalization_factor=1.5,
            dummy=-1.0,
            delta_dummy=0.1,
            reflected_beam_x_px=10.0,
            reflected_beam_y_px=20.0,
            critical_angle_deg=0.11,
            sample_stoichiometry="CH3NH3PbI3",
            sample_density_g_cm3=4.16,
            refractive_index_delta=3.2e-6,
            artifact_regions=[{"label": "beamstop", "x": 1, "y": 2}],
            confirmed=True,
        )
    )
    project.fits[data_id] = [{"model_name": "gaussian-2d"}]
    project.analysis_results[data_id] = {"status": "draft"}

    saved = save_project(project, tmp_path / "grouped")
    loaded = load_project(saved)

    assert loaded.data_groups[0].name == "Example"
    assert loaded.data_groups[0].data_files[0].metadata["sample_number"] == 22
    assert loaded.integration_regions[data_id][0]["name"] == "fit-window"
    assert loaded.roi_regions[data_id][0].name == "Fit window"
    assert loaded.roi_regions[data_id][0].integration_axis == "qxy"
    assert loaded.image_corrections[data_id].confirmed
    assert loaded.image_corrections[data_id].xray_energy_kev == 12.7
    assert loaded.image_corrections[data_id].image_rotation_deg == 270
    assert loaded.image_corrections[data_id].image_mirrored_y
    assert loaded.image_corrections[data_id].pyfai_sample_orientation == 7
    assert not loaded.image_corrections[data_id].correct_solid_angle
    assert loaded.image_corrections[data_id].polarization_factor is None
    assert loaded.image_corrections[data_id].normalization_factor == 1.5
    assert loaded.image_corrections[data_id].dummy == -1.0
    assert loaded.image_corrections[data_id].delta_dummy == 0.1
    assert loaded.image_corrections[data_id].critical_angle_deg == 0.11
    assert (
        loaded.image_corrections[data_id].sample_stoichiometry == "CH3NH3PbI3"
    )
    assert loaded.image_corrections[data_id].sample_density_g_cm3 == 4.16
    assert loaded.image_corrections[data_id].refractive_index_delta == 3.2e-6
    assert loaded.fits[data_id][0]["model_name"] == "gaussian-2d"
    assert loaded.analysis_results[data_id]["status"] == "draft"


def test_project_film_material_memory_add_update_delete():
    project = ProjectState()

    first = project.remember_film_material("  C8H8  ", 1.05)
    updated = project.remember_film_material(
        "C8H8",
        1.07,
        memory_id=first["memory_id"],
        label="PS updated",
        refractive_index_delta=1.2e-6,
    )

    assert len(project.film_material_memory) == 1
    assert updated["stoichiometry"] == "C8H8"
    assert project.film_material_memory[0]["density_g_cm3"] == 1.07
    assert project.film_material_memory[0]["label"] == "PS updated"
    assert project.remove_film_material_memory(first["memory_id"]) is True
    assert project.film_material_memory == []
    assert project.remove_film_material_memory(first["memory_id"]) is False


def test_project_reload_restores_complete_manifest_state(repo_root, tmp_path):
    image = next((repo_root / "example").glob("*.tiff"))
    file_group, _ = build_data_group_from_paths(
        [image],
        group_name="Single Image",
    )
    folder_group, _ = build_data_group_from_folder(repo_root / "example")
    folder_group.name = "Time Series"
    folder_group.group_id = "time_series"
    project = ProjectState(name="Complete Reload")
    project.add_data_group(file_group)
    project.add_data_group(folder_group)
    data_id = file_group.data_files[0].data_id
    mask = project.add_mask(
        repo_root / "example" / "mask.edf",
        target_ids=[data_id, folder_group.group_id],
        metadata={"purpose": "beamstop"},
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni",
        target_ids=[data_id],
        metadata={"calibrant": "AgBH"},
    )
    project.processed_products[data_id] = "products/qspace.zarr"
    project.integration_regions[data_id] = [
        {"name": "vertical linecut", "qxy": 0.1, "qz_min": 0.2}
    ]
    box_roi = project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="box",
            name="Box Fit",
            qxy_min=0.1,
            qxy_max=0.4,
            qz_min=0.2,
            qz_max=0.7,
            integration_axis="qz",
            integration_direction="vertical",
            metadata={"table_row": 1},
        )
    )
    arch_roi = project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="arch",
            name="Arch Fit",
            qr_min=0.5,
            qr_max=0.9,
            chi_min=-35.0,
            chi_max=35.0,
            integration_axis="chi",
            integration_direction="azimuthal",
        )
    )
    box_roi.metadata.update(
        {
            "coupling_id": "roi_pair_1",
            "coupled_role": "box",
            "coupled_roi_id": arch_roi.roi_id,
            "coupled_roi_ids": [arch_roi.roi_id],
            "shared_center": True,
        }
    )
    arch_roi.metadata.update(
        {
            "coupling_id": "roi_pair_1",
            "coupled_role": "arch",
            "coupled_roi_id": box_roi.roi_id,
            "coupled_roi_ids": [box_roi.roi_id],
            "shared_center": True,
        }
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            xray_energy_kev=13.5,
            image_rotation_deg=90,
            image_mirrored_y=True,
            pyfai_sample_orientation=5,
            correct_solid_angle=False,
            polarization_factor=0.91,
            normalization_factor=2.5,
            dummy=-1.0,
            delta_dummy=0.2,
            reflected_beam_x_px=100.0,
            reflected_beam_y_px=200.0,
            critical_angle_deg=0.12,
            artifact_regions=[
                {
                    "label": "beamstop",
                    "x": 1.0,
                    "y": 2.0,
                    "width": 3.0,
                    "height": 4.0,
                }
            ],
            confirmed=True,
            metadata={"workflow": "reload-test"},
        )
    )
    project.peak_sets[data_id] = [
        {
            "peak_id": "peak_1",
            "qxy": 1.0,
            "qz": 0.5,
            "intensity": 42.0,
            "roi": {
                "kind": "box",
                "qxy_min": 0.9,
                "qxy_max": 1.1,
                "qz_min": 0.4,
                "qz_max": 0.6,
            },
        }
    ]
    project.fits[data_id] = [
        {
            "model_name": "gaussian-2d",
            "center_qx": 1.0,
            "center_qz": 0.5,
            "statistics": {"r_squared": 0.99},
        }
    ]
    project.analysis_results[data_id] = {
        "status": "draft",
        "fit_points": [{"qx": 1.0, "qz": 0.5}],
    }
    project.structures[data_id] = {
        "space_group": "P1",
        "cell": {"a": 6.3, "b": 6.3, "c": 6.3},
    }
    project.reference_cifs[data_id] = {
        "path": "reference/generated.cif",
        "source": "generated",
        "score": 0.12,
    }
    project.simulations[data_id] = {
        "dataset_uri": "simulations/best.zarr",
        "objective": 0.03,
        "parameters": {"iterations": 25},
    }
    project.processing_history.append(
        ProcessingRecord(
            stage="reload.audit",
            parameters={"data_id": data_id},
            outputs={"project_asset": "products/qspace.zarr"},
            notes="Complete reload contract",
        )
    )

    saved = save_project(project, tmp_path / "complete_reload")
    with ZipFile(saved) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        archive_members = set(archive.namelist())
    loaded = load_project(saved)

    assert loaded.as_dict() == manifest
    assert loaded.data_groups[0].data_files[0].archive_path in archive_members
    assert loaded.data_groups[0].data_files[0].local_path is not None
    assert loaded.data_groups[0].data_files[0].local_path.exists()
    assert loaded.data_groups[1].import_kind == "folder"
    assert loaded.data_groups[1].data_files[0].archive_path is None
    assert loaded.masks[0].local_path is not None
    assert loaded.masks[0].local_path.exists()
    assert loaded.calibrants[0].local_path is not None
    assert loaded.calibrants[0].local_path.exists()
    assert loaded.image_corrections[data_id].confirmed
    assert loaded.image_corrections[data_id].pyfai_sample_orientation == 5
    assert len(loaded.roi_regions[data_id]) == 2
    assert loaded.peak_sets[data_id][0]["roi"]["qxy_min"] == 0.9
    assert loaded.roi_regions[data_id][1].kind == "arch"
    assert (
        loaded.roi_regions[data_id][0].metadata["coupled_roi_id"]
        == arch_roi.roi_id
    )
    assert (
        loaded.roi_regions[data_id][1].metadata["coupled_roi_id"]
        == box_roi.roi_id
    )
    assert loaded.fits[data_id][0]["statistics"]["r_squared"] == 0.99
    assert loaded.analysis_results[data_id]["fit_points"][0]["qx"] == 1.0
    assert loaded.structures[data_id]["space_group"] == "P1"
    assert loaded.reference_cifs[data_id]["score"] == 0.12
    assert loaded.simulations[data_id]["objective"] == 0.03
    assert loaded.processing_history[-1].stage == "reload.audit"


def test_project_round_trip_preserves_masks_and_calibrants(
    repo_root, tmp_path
):
    image = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([image], group_name="Example")
    project = ProjectState(name="Corrections")
    project.add_data_group(group)
    data_id = group.data_files[0].data_id

    mask = project.add_mask(
        repo_root / "example" / "mask.edf",
        target_ids=[group.group_id],
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni",
        target_ids=[data_id],
    )

    saved = save_project(project, tmp_path / "corrections")
    loaded = load_project(saved)

    assert loaded.masks[0].asset_id == mask.asset_id
    assert loaded.masks[0].target_ids == [group.group_id]
    assert loaded.masks[0].path.name == "mask.edf"
    assert loaded.masks[0].local_path is not None
    assert loaded.masks[0].local_path.exists()
    assert loaded.calibrants[0].asset_id == calibrant.asset_id
    assert loaded.calibrants[0].target_ids == [data_id]
    assert loaded.calibrants[0].local_path is not None
    assert loaded.calibrants[0].local_path.exists()


def test_project_round_trip_preserves_new_analysis_metadata(tmp_path):
    project = ProjectState(name="Consistency Metadata")
    data_id = "detector"
    project.remember_film_material(
        "CH3NH3PbI3",
        4.16,
        label="MAPbI3",
        refractive_index_delta=3.2e-6,
        critical_angle_deg=0.11,
        metadata={"source_file": "reference.cif"},
    )
    box_roi = project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="box",
            roi_id="box_roi",
            name="Box ROI",
            qxy_min=0.1,
            qxy_max=0.3,
            qz_min=0.4,
            qz_max=0.6,
        )
    )
    arch_roi = project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="arch",
            roi_id="arch_roi",
            name="Arch ROI",
            qxy_center=0.2,
            qz_center=0.5,
            qr_min=0.8,
            qr_max=1.0,
            chi_min=-30.0,
            chi_max=30.0,
        )
    )
    box_roi.metadata.update(
        {
            "coupling_id": "pair_1",
            "coupled_role": "box",
            "coupled_roi_id": arch_roi.roi_id,
            "coupled_roi_ids": [arch_roi.roi_id],
            "shared_center": True,
        }
    )
    arch_roi.metadata.update(
        {
            "coupling_id": "pair_1",
            "coupled_role": "arch",
            "coupled_roi_id": box_roi.roi_id,
            "coupled_roi_ids": [box_roi.roi_id],
            "shared_center": True,
        }
    )
    project.set_roi_hkl_tag(
        data_id,
        box_roi.roi_id or "",
        h=1,
        k=0,
        l=1,
        label="(101) main",
    )
    project.set_roi_pole_figure_metadata(
        data_id,
        box_roi.roi_id or "",
        {
            "output_path": str(tmp_path / "missing_pole_figure.npy"),
            "settings": {"chi_bin_width_deg": 5.0},
            "status": "generated",
        },
    )
    project.peak_sets[data_id] = [
        {
            "peak_id": "p1",
            "label": "Peak 1",
            "qxy": 0.2,
            "qz": 0.5,
            "roi_id": box_roi.roi_id,
            "point_kind": PEAK_POINT_KIND_COMMITTED,
            "phase_tag": "alpha",
            "hkl": {"h": 1, "k": 0, "l": 1, "label": "(101) main"},
        },
        {
            "peak_id": "gap_1",
            "label": "Gap 1",
            "qxy": 0.7,
            "qz": 0.8,
            "source": "gap estimate",
            "point_kind": PEAK_POINT_KIND_GAP_ESTIMATED,
            "gap_estimated": True,
            "metadata": {
                "gap_estimate": True,
                "estimate_method": "missing wedge interpolation",
            },
        },
    ]
    project.set_peak_fit_result(
        data_id,
        "p1",
        {
            "center_qxy": 0.21,
            "center_qz": 0.51,
            "status": "ok",
            "statistics": {"r_squared": 0.97, "sigma_qxy": 0.01},
        },
        roi_id=box_roi.roi_id,
    )
    project.sync_structure_analysis_peak_from_fit(data_id, "gap_1")
    structure_state = project.analysis_results["structure_analysis"][data_id]
    structure_state["candidates"] = [{"candidate_id": "c1", "score": 0.12}]
    structure_state["wyckoff"] = {
        "atoms": [{"element": "Pb", "site": "1a"}],
        "generated_cifs": [{"cif_id": "cif_1", "path": "generated/cif_1.cif"}],
    }
    project.reference_cifs["generated"] = {
        "cif_1": {"path": "generated/cif_1.cif", "status": "written"}
    }
    project.simulations["sim_1"] = {
        "data_id": data_id,
        "cache_key": "abc123",
        "orientation_preset": "edge-on",
        "metadata": {"computed": True},
    }
    project.mark_roi_pole_figures_stale(
        data_id,
        arch_roi.roi_id or "",
        reason="parent ROI moved",
    )

    saved = save_project(project, tmp_path / "consistency_metadata")
    loaded = load_project(saved)

    loaded_box, loaded_arch = loaded.rois_for_target(data_id)
    assert loaded.film_material_memory[0]["stoichiometry"] == "CH3NH3PbI3"
    assert loaded.film_material_memory[0]["critical_angle_deg"] == 0.11
    assert loaded_box.metadata["coupled_roi_id"] == arch_roi.roi_id
    assert loaded_arch.metadata["coupled_roi_id"] == box_roi.roi_id
    assert roi_hkl_metadata(loaded_box)["label"] == "(101) main"
    assert roi_pole_figure_status(loaded_box) == "Stale"
    pole_record = loaded.analysis_results["pole_figures"][data_id][
        box_roi.roi_id
    ]
    assert pole_record["current"] is False
    assert pole_record["stale_reason"] == "parent ROI moved"
    assert (
        loaded.peak_sets[data_id][1]["point_kind"]
        == PEAK_POINT_KIND_GAP_ESTIMATED
    )
    peaks = loaded.analysis_results["structure_analysis"][data_id]["peaks"]
    fitted = next(item for item in peaks if item["peak_id"] == "p1")
    gap = next(item for item in peaks if item["peak_id"] == "gap_1")
    assert fitted["center_qxy"] == 0.21
    assert fitted["fit_metrics"]["r_squared"] == 0.97
    assert fitted["phase_tag"] == "alpha"
    assert gap["gap_estimated"] is True
    assert gap["estimate_method"] == "missing wedge interpolation"
    assert loaded.reference_cifs["generated"]["cif_1"]["status"] == "written"
    assert loaded.simulations["sim_1"]["cache_key"] == "abc123"


def test_project_archive_embeds_generated_cif_files(tmp_path):
    project = ProjectState(name="Generated CIF archive")
    cif_path = tmp_path / "candidate_001.cif"
    cif_text = "data_candidate_001\n_cell_length_a 6.3\n"
    cif_path.write_text(cif_text, encoding="utf-8")
    project.reference_cifs["generated"] = {
        "candidate_001": {
            "cif_id": "candidate_001",
            "path": str(cif_path),
            "cif_text": cif_text,
            "status": "written",
        }
    }
    project.structures["candidate_001"] = {
        "structure_id": "candidate_001",
        "source": "structure_analysis_generated_cif",
        "path": str(cif_path),
        "cif_text": cif_text,
    }
    project.analysis_results["structure_analysis"] = {
        "synthetic": {
            "wyckoff": {
                "generated_cifs": [
                    {
                        "cif_id": "candidate_001",
                        "path": str(cif_path),
                        "cif_text": cif_text,
                    }
                ]
            }
        }
    }

    saved = save_project(project, tmp_path / "generated_cif_archive")
    with ZipFile(saved) as archive:
        members = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    archive_path = manifest["reference_cifs"]["generated"]["candidate_001"][
        "archive_path"
    ]
    cif_path.unlink()
    loaded = load_project(saved)

    assert archive_path in members
    assert archive_path.startswith("structures/generated_cifs/candidate_001/")
    loaded_record = loaded.reference_cifs["generated"]["candidate_001"]
    assert loaded_record["local_path"].endswith("candidate_001.cif")
    assert loaded_record["path"] == loaded_record["local_path"]
    assert loaded.structures["candidate_001"]["path"] == loaded_record["path"]
    assert (
        loaded.analysis_results["structure_analysis"]["synthetic"]["wyckoff"][
            "generated_cifs"
        ][0]["path"]
        == loaded_record["path"]
    )


def test_project_archive_embeds_loaded_cif_files(tmp_path):
    project = ProjectState(name="Loaded CIF archive")
    cif_path = tmp_path / "reference_001.cif"
    cif_text = (
        "data_reference_001\n"
        "_cell_length_a 4.2\n"
        "_cell_length_b 4.2\n"
        "_cell_length_c 7.1\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
    )
    cif_path.write_text(cif_text, encoding="utf-8")
    record = project.remember_loaded_cif(
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
        target_id="synthetic",
    )

    saved = save_project(project, tmp_path / "loaded_cif_archive")
    with ZipFile(saved) as archive:
        members = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    archive_path = manifest["reference_cifs"]["loaded"][record["cif_id"]][
        "archive_path"
    ]
    cif_path.unlink()
    loaded = load_project(saved)
    loaded_record = loaded.reference_cifs["loaded"][record["cif_id"]]

    assert archive_path in members
    assert archive_path.startswith("structures/loaded_cifs/")
    assert loaded_record["local_path"].endswith("reference_001.cif")
    assert loaded_record["path"] == loaded_record["local_path"]
    assert loaded_record["lattice"]["c"] == pytest.approx(7.1)


def test_project_save_writes_human_readable_sidecar_with_cif_file(tmp_path):
    project = ProjectState(name="Readable CIF project")
    cif_text = "data_candidate_001\n_cell_length_a 6.3\n"
    project.reference_cifs["generated"] = {
        "candidate_001": {
            "cif_id": "candidate_001",
            "cif_text": cif_text,
            "status": "generated",
        }
    }

    saved = save_project(project, tmp_path / "readable_cif_project")
    readable_path = saved.with_suffix(READABLE_PROJECT_EXTENSION)
    payload = json.loads(readable_path.read_text(encoding="utf-8"))
    materialized_cif_path = Path(
        payload["reference_cifs"]["generated"]["candidate_001"]["path"]
    )
    loaded = load_project(readable_path)

    assert readable_path.exists()
    assert readable_path.read_text(encoding="utf-8").startswith("{\n")
    assert materialized_cif_path.exists()
    assert materialized_cif_path.read_text(encoding="utf-8") == cif_text
    assert materialized_cif_path.parent.name == "candidate_001"
    assert loaded.reference_cifs["generated"]["candidate_001"]["path"] == str(
        materialized_cif_path
    )


def test_readable_project_file_round_trips_as_json(tmp_path):
    project = ProjectState(name="Readable project")
    project.add_data_file("detector_001.tiff", incident_angle_deg=0.3)

    saved = save_project(project, tmp_path / "readable_project.ewald.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    loaded = load_project(saved)

    assert saved.name == "readable_project.ewald.json"
    assert payload["name"] == "Readable project"
    assert loaded.data_files[0].path.name == "detector_001.tiff"


def test_project_archive_embeds_assets_and_single_imports(repo_root, tmp_path):
    image = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([image], group_name="Single")
    project = ProjectState(name="Archive")
    project.add_data_group(group)
    project.add_mask(repo_root / "example" / "mask.edf")
    project.add_calibrant(repo_root / "example" / "calib.poni")

    saved = save_project(project, tmp_path / "archive")

    with ZipFile(saved) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    archived_data = manifest["data_groups"][0]["data_files"][0]["archive_path"]
    archived_mask = manifest["masks"][0]["archive_path"]
    archived_calibrant = manifest["calibrants"][0]["archive_path"]

    assert archived_data.startswith("data_files/")
    assert archived_mask.startswith("assets/masks/")
    assert archived_calibrant.startswith("assets/calibrants/")
    assert archived_data in names
    assert archived_mask in names
    assert archived_calibrant in names


def test_project_archive_keeps_folder_imports_as_external_pointers(
    repo_root, tmp_path
):
    folder = repo_root / "example"
    group, _ = build_data_group_from_folder(folder)
    project = ProjectState(name="Folder Pointer")
    project.add_data_group(group)

    saved = save_project(project, tmp_path / "folder_pointer")

    with ZipFile(saved) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["data_groups"][0]["path"] == str(folder)
    assert manifest["data_groups"][0]["import_kind"] == "folder"
    assert manifest["data_groups"][0]["data_files"][0]["archive_path"] is None
