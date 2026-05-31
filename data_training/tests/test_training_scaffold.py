# flake8: noqa: E402
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.artifact_features import (
    annotate_peaks_with_artifacts,
    artifact_weight_map_from_assessment,
    build_artifact_assessment,
    estimate_retrieval_quality,
)
from ewald_data_training.artifacts import apply_artifacts
from ewald_data_training.conditions import load_generation_plan
from ewald_data_training.detectors import (
    resolve_detector_preset,
    scaled_detector_gap_mask,
)
from ewald_data_training.hybrid3 import (
    convert_structure_file,
    download_structure_files,
    extract_structure_file_metadata,
    infer_structure_format,
    is_structure_like_path,
    load_fixture_records,
    parse_formula_counts,
)
from ewald_data_training.manifests import (
    read_jsonl_manifest,
    write_jsonl_manifest,
)
from ewald_data_training.ranking import (
    peak_table_vector,
    rank_image_candidates,
)
from ewald_data_training.schemas import (
    ArtifactProfile,
    DatasetSample,
    DetectorGeometry,
)


def test_generation_plan_expands_demo_sweep() -> None:
    loaded = load_generation_plan(
        ROOT / "data_training" / "configs" / "simulation_sweep.example.yaml"
    )

    assert len(loaded["structures"]) == 3
    assert len(loaded["conditions"]) == 2
    first = loaded["conditions"][0]
    assert first.detector.resolution == (384, 256)
    assert first.detector.qxy_range == (-2.8, 2.8)
    assert first.detector.qz_range == (0.0, 2.8)
    assert first.detector.incident_angle_deg == 0.2
    assert first.detector.missing_wedge_correction is True
    assert first.as_giwaxs_parameters()["hkl_extent"] == 14
    assert first.as_giwaxs_parameters()["incident_angle_deg"] == 0.2
    assert first.as_giwaxs_parameters()["missing_wedge_correction"] is True
    assert first.as_giwaxs_parameters()["q_dependent_sigma_r"] == 0.25


def test_generation_plan_expands_multiple_detectors() -> None:
    loaded = load_generation_plan(
        ROOT
        / "data_training"
        / "configs"
        / "simulation_alpine_fibril_training.example.yaml"
    )

    detector_ids = {
        condition.metadata["detector_sweep_id"]
        for condition in loaded["conditions"]
    }

    assert len(loaded["detectors"]) == 3
    assert len(loaded["conditions"]) == 432
    assert detector_ids == {
        "pilatus1m_2p8a_10kev",
        "eiger2_1mw_2p6a_12kev_low_alpha",
        "perkin_elmer_3p0a_10kev_high_alpha",
    }
    assert {
        condition.detector.incident_angle_deg
        for condition in loaded["conditions"]
    } == {0.16, 0.20, 0.24}


def test_local_hybrid3_scaffold_plan_enables_missing_wedge(tmp_path) -> None:
    module = _load_local_scaffold_module()
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("structures: []\n", encoding="utf-8")

    plan_path = module._write_generation_plan(
        tmp_path,
        catalog_path=catalog,
        hkl_extent=18,
        qxy_range=(-2.8, 2.8),
        qz_range=(0.0, 2.8),
    )
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))

    assert payload["detector"]["missing_wedge_correction"] is True
    assert payload["detector"]["solid_angle_correction"] is True
    assert payload["detector"]["incident_angle_deg"] == 0.2
    assert payload["detector"]["wavelength_angstrom"] == 1.0


def test_example_artifact_profiles_include_surface_scattering() -> None:
    payload = yaml.safe_load(
        (
            ROOT / "data_training" / "configs" / "artifacts.example.yaml"
        ).read_text(encoding="utf-8")
    )
    profiles = {
        profile_id: ArtifactProfile.from_mapping(
            {"profile_id": profile_id, **(profile or {})}
        )
        for profile_id, profile in payload["artifact_profiles"].items()
    }

    for profile_id in ("default", "harsh_detector", "pilatus1m_reference"):
        profile = profiles[profile_id]
        assert profile.enabled is True
        assert profile.surface_scattering is True
        assert profile.direct_beam is True
        assert profile.yoneda_peak is True
        assert profile.critical_peak_splitting is True
        assert profile.substrate_horizon is True
        assert profile.spillage_broadening is True


def test_artifacts_are_seed_reproducible() -> None:
    image = np.zeros((32, 64), dtype=np.float32)
    image[12:15, 20:23] = 1.0
    profile = ArtifactProfile(
        profile_id="test",
        diffuse_ring_count=2,
        detector_layout="pilatus1m_pyfai",
        parasitic_streaks=0,
    )

    first, first_meta = apply_artifacts(image, profile, seed=42)
    second, second_meta = apply_artifacts(image, profile, seed=42)

    np.testing.assert_allclose(first, second)
    assert first_meta == second_meta
    assert "beamstop_shadow" in first_meta["operations"]
    assert "structure_correlated_diffuse_rings" in first_meta["operations"]
    assert first_meta["detector_preset"]["preset_id"] == "pilatus1m_pyfai"


def test_surface_scattering_uses_incident_angle_and_critical_angle() -> None:
    image = np.zeros((48, 64), dtype=np.float32)
    image[20, 28] = 1.0
    detector = DetectorGeometry(
        qxy_range=(-4.0, 4.0),
        qz_range=(0.0, 4.0),
        resolution=(64, 48),
        wavelength_angstrom=1.0,
        incident_angle_deg=0.2,
        detector="perkin_elmer_xrd_1621",
    )
    profile = ArtifactProfile(
        profile_id="surface",
        beamstop=False,
        detector_layout="perkin_elmer_xrd_1621",
        diffuse_ring_count=0,
        direct_beam=True,
        yoneda_peak=True,
        critical_peak_splitting=True,
        critical_angle_deg=0.24,
        substrate_length_mm=2.0,
        beam_height_um=60.0,
        spillage_broadening_strength=0.5,
    )

    _augmented, metadata = apply_artifacts(
        image,
        profile,
        seed=7,
        detector=detector,
    )

    operations = set(metadata["operations"])
    surface = metadata["surface_scattering"]
    assert "direct_beam_specular" in operations
    assert "yoneda_band" in operations
    assert "critical_angle_peak_splitting" in operations
    assert "substrate_horizon_shadow" in operations
    assert "footprint_spillage_peak_broadening" in operations
    assert surface["incident_angle_deg"] == 0.2
    assert surface["critical_angle_deg"] == 0.24
    assert surface["critical_angle_source"] == "artifact_profile"
    assert surface["horizon_qz"] > surface["direct_beam_qz"]
    assert surface["horizon_qz"] < surface["specular_qz"]
    assert surface["yoneda_qz"] > surface["specular_qz"]
    assert surface["beam_footprint_length_mm"] > surface["substrate_length_mm"]
    assert surface["footprint_spillage_fraction"] > 0.0


def test_artifact_assessment_labels_surface_and_mask_regions() -> None:
    image = np.zeros((48, 64), dtype=np.float32)
    image[20, 28] = 1.0
    detector = DetectorGeometry(
        qxy_range=(-4.0, 4.0),
        qz_range=(0.0, 4.0),
        resolution=(64, 48),
        wavelength_angstrom=1.0,
        incident_angle_deg=0.2,
        detector="pilatus1m_pyfai",
    )
    profile = ArtifactProfile(
        profile_id="surface",
        detector_layout="pilatus1m_pyfai",
        direct_beam=True,
        yoneda_peak=True,
        critical_peak_splitting=True,
        critical_angle_deg=0.24,
        substrate_length_mm=2.0,
        beam_height_um=60.0,
        diffuse_ring_count=0,
    )

    _augmented, metadata = apply_artifacts(
        image,
        profile,
        seed=8,
        detector=detector,
    )
    assessment = build_artifact_assessment(
        artifact_metadata=metadata,
        artifact_profile=profile,
        detector=detector,
        image_shape=image.shape,
    )
    weights = artifact_weight_map_from_assessment(
        assessment,
        detector=detector,
        image_shape=image.shape,
    )
    annotated = annotate_peaks_with_artifacts(
        [{"h": 0, "k": 0, "l": 1, "qxy": 0.0, "qz": 0.0}],
        assessment,
    )

    kinds = {region["kind"] for region in assessment["regions"]}
    assert "direct_beam" in kinds
    assert "yoneda_band" in kinds
    assert "substrate_horizon" in kinds
    assert "critical_angle_peak_split" in kinds
    assert "detector_module_gap_mask" in kinds
    assert "beamstop_shadow" in kinds
    assert 0.0 < assessment["mean_training_weight"] < 1.0
    assert 0.0 < assessment["usable_fraction"] < 1.0
    assert float(weights.min()) == 0.0
    assert float(weights.mean()) < 1.0
    assert "direct_beam" in annotated[0]["artifact_overlap"]
    assert annotated[0]["bragg_training_weight"] < 1.0


def test_training_peak_vectors_exclude_forbidden_reflections() -> None:
    peaks = [
        {
            "qxy": 0.0,
            "qz": 0.5,
            "amplitude": 2.0,
            "forbidden_reflection": True,
            "excluded_from_indexing": True,
        },
        {"qxy": 0.5, "qz": 0.5, "amplitude": 3.0},
    ]

    vector = peak_table_vector(
        peaks,
        qxy_range=(-1.0, 1.0),
        qz_range=(0.0, 1.0),
        bins=(5, 5),
    )
    annotated = annotate_peaks_with_artifacts(peaks, {"regions": []})

    assert np.count_nonzero(vector) == 1
    assert np.isclose(np.max(vector), 3.0)
    assert annotated[0]["bragg_training_weight"] == 0.0
    assert annotated[0]["training_excluded"] is True


def test_artifact_aware_ranking_downweights_direct_beam() -> None:
    observed = np.zeros((24, 32), dtype=np.float32)
    true_candidate = np.zeros_like(observed)
    artifact_candidate = np.zeros_like(observed)
    observed[10, 22] = 1.0
    true_candidate[10, 22] = 1.0
    observed[1, 16] = 20.0
    artifact_candidate[1, 16] = 20.0
    assessment = {
        "regions": [
            {
                "kind": "beamstop_shadow",
                "pixel_spans": {
                    "row_start": 0,
                    "row_stop": 4,
                    "col_start": 14,
                    "col_stop": 19,
                },
                "training_weight": 0.0,
            }
        ]
    }
    weights = artifact_weight_map_from_assessment(
        assessment,
        detector=None,
        image_shape=observed.shape,
    )

    unweighted = rank_image_candidates(
        observed,
        {"true": true_candidate, "artifact": artifact_candidate},
    )
    weighted = rank_image_candidates(
        observed,
        {"true": true_candidate, "artifact": artifact_candidate},
        weights=weights,
    )

    assert unweighted[0].candidate_id == "artifact"
    assert weighted[0].candidate_id == "true"


def _load_local_scaffold_module():
    path = (
        ROOT
        / "data_training"
        / "scripts"
        / "run_local_hybrid3_scaffold_test.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_local_hybrid3_scaffold_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retrieval_quality_flags_solvable_and_unsolvable_augments() -> None:
    clean = np.zeros((48, 64), dtype=np.float32)
    clean[16, 20] = 1.0
    clean[28, 44] = 0.8
    clean[36, 32] = 0.6
    rng = np.random.default_rng(123)
    solvable = clean + rng.normal(0.0, 0.02, size=clean.shape)
    unsolvable = rng.normal(0.25, 0.02, size=clean.shape)
    assessment = {"regions": []}

    good = estimate_retrieval_quality(
        clean,
        solvable,
        artifact_assessment=assessment,
    )
    bad = estimate_retrieval_quality(
        clean,
        unsolvable,
        artifact_assessment=assessment,
    )

    assert good["solvable"] is True
    assert good["signal_to_noise"] > bad["signal_to_noise"]
    assert (
        good["retrievable_signal_fraction"]
        > bad["retrievable_signal_fraction"]
    )
    assert bad["solvable"] is False
    assert bad["warning_reasons"]


def test_detector_gap_mask_scales_pilatus_layout() -> None:
    preset = resolve_detector_preset("pilatus1m_pyfai")
    assert preset is not None

    mask = scaled_detector_gap_mask(preset, (128, 160))

    assert mask.shape == (128, 160)
    assert 0.0 < float(np.mean(mask)) < 0.08


def test_image_overlap_ranking_orders_best_match() -> None:
    observed = np.zeros((8, 8), dtype=np.float32)
    observed[2, 3] = 1.0
    close = observed.copy()
    far = np.zeros_like(observed)
    far[6, 6] = 1.0

    scores = rank_image_candidates(
        observed,
        {"far": far, "close": close},
    )

    assert scores[0].candidate_id == "close"
    assert scores[0].rank == 1
    assert scores[0].score > scores[1].score


def test_manifest_roundtrip(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tiff"
    label_path = tmp_path / "labels.json"
    image_path.write_bytes(b"demo")
    label_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    sample = DatasetSample(
        sample_id="s1",
        structure_id="str1",
        condition_id="c1",
        image_path="image.tiff",
        label_path="labels.json",
    )

    write_jsonl_manifest(manifest_path, [sample])
    loaded = read_jsonl_manifest(manifest_path)

    assert loaded[0].sample_id == "s1"
    assert loaded[0].image_path == "image.tiff"


def test_hybrid3_fixture_ingest_converts_geometry(tmp_path: Path) -> None:
    fixture_root = ROOT / "data_training" / "fixtures" / "hybrid3"
    records = load_fixture_records(fixture_root)

    summary = download_structure_files(
        records,
        output_root=tmp_path,
        fixture_root=fixture_root,
    )

    assert summary["ready"] == 1
    catalog_text = (tmp_path / "hybrid3_structure_catalog.yaml").read_text()
    assert "hybrid3_1635" in catalog_text
    assert (tmp_path / "structures" / "dataset_1635_geometry.vasp").exists()
    catalog = yaml.safe_load(catalog_text)
    metadata = catalog["structures"][0]["metadata"]
    assert metadata["components"]["inorganic_formula"] == "PbI2"
    assert metadata["components"]["element_counts"]["inorganic"] == {
        "I": 2,
        "Pb": 1,
    }
    assert metadata["structure_file"]["species_counts"] == {"I": 2, "Pb": 1}


def test_hybrid3_metadata_extracts_formula_and_cif_tags(
    tmp_path: Path,
) -> None:
    cif = tmp_path / "demo.cif"
    cif.write_text(
        "\n".join(
            [
                "data_demo",
                "_chemical_formula_sum 'C76 H72 I10 N12 O16 Pb3'",
                "_chemical_formula_moiety 'I10 Pb3, 4(C19 H18 N3 O4)'",
                "_cell_length_a 11.07924(16)",
                "_cell_length_b 11.1826(18)",
                "_cell_length_c 20.0231(3)",
                "_cell_angle_alpha 90.3834(13)",
                "_cell_angle_beta 93.2422(12)",
                "_cell_angle_gamma 114.3988(15)",
                "_space_group_crystal_system triclinic",
                "_symmetry_space_group_name_H-M 'P -1'",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "Pb1 Pb 0 0 0",
                "I1 I 0.5 0.5 0.25",
                "C1 C 0.1 0.2 0.3",
            ]
        ),
        encoding="utf-8",
    )

    metadata = extract_structure_file_metadata(cif)

    assert metadata["cell"]["a"] == 11.07924
    assert metadata["space_group_name"] == "P -1"
    assert metadata["formula_element_counts"]["Pb"] == 3
    assert metadata["formula_element_counts"]["I"] == 10
    assert metadata["atom_sites"]["element_site_counts"] == {
        "C": 1,
        "I": 1,
        "Pb": 1,
    }
    assert parse_formula_counts("4(C19H18N3O4)") == {
        "C": 76,
        "H": 72,
        "N": 12,
        "O": 16,
    }


def test_aims_geometry_conversion_writes_poscar(tmp_path: Path) -> None:
    source = ROOT / "data_training" / "fixtures" / "hybrid3" / "geometry.in"

    converted = convert_structure_file(source, tmp_path)

    assert converted is not None
    text = converted.read_text(encoding="utf-8")
    assert "Direct" in text
    assert "Pb" in text and "I" in text


def test_hybrid3_link_detection_accepts_common_variants() -> None:
    assert is_structure_like_path("/media/data_files/a/file.cif?download=1")
    assert is_structure_like_path("/media/data_files/a/geometry.in")
    assert is_structure_like_path("/media/data_files/a/POSCAR")
    assert is_structure_like_path("/media/data_files/a/CONTCAR")
    assert is_structure_like_path("/media/data_files/a/structures.zip")
    assert is_structure_like_path("/media/data_files/a/structure.cif.gz")
    assert is_structure_like_path("/media/data_files/a/POSCAR.gz")
    assert not is_structure_like_path("/media/data_files/a/plot.png")


def test_hybrid3_converts_format_variants(tmp_path: Path) -> None:
    poscar = tmp_path / "CONTCAR"
    poscar.write_text(_minimal_poscar(), encoding="utf-8")
    assert infer_structure_format(poscar) == "poscar"
    assert convert_structure_file(poscar, tmp_path / "out") is not None

    cell = tmp_path / "demo.cell"
    cell.write_text(
        "\n".join(
            [
                "%BLOCK LATTICE_CART",
                "4.0 0.0 0.0",
                "0.0 4.0 0.0",
                "0.0 0.0 6.0",
                "%ENDBLOCK LATTICE_CART",
                "%BLOCK POSITIONS_FRAC",
                "Pb 0.0 0.0 0.0",
                "I 0.5 0.5 0.25",
                "%ENDBLOCK POSITIONS_FRAC",
            ]
        ),
        encoding="utf-8",
    )
    converted_cell = convert_structure_file(cell, tmp_path / "cell_out")
    assert converted_cell is not None
    assert "Direct" in converted_cell.read_text(encoding="utf-8")

    xyz = tmp_path / "demo.xyz"
    xyz.write_text(
        "\n".join(
            [
                "2",
                'Lattice="4 0 0 0 4 0 0 0 6" Properties=species:S:1:pos:R:3',
                "Pb 0.0 0.0 0.0",
                "I 1.0 1.0 1.5",
            ]
        ),
        encoding="utf-8",
    )
    converted_xyz = convert_structure_file(xyz, tmp_path / "xyz_out")
    assert converted_xyz is not None
    assert "Cartesian" in converted_xyz.read_text(encoding="utf-8")

    gz_poscar = tmp_path / "POSCAR.gz"
    with gzip.open(gz_poscar, "wb") as handle:
        handle.write(_minimal_poscar().encode("utf-8"))
    converted_gz = convert_structure_file(gz_poscar, tmp_path / "gz_out")
    assert converted_gz is not None
    assert converted_gz.suffix == ".vasp"

    pdb = tmp_path / "demo.pdb"
    pdb.write_text(
        "\n".join(
            [
                "CRYST1    4.000    4.000    6.000  90.00  90.00  90.00 P 1",
                "HETATM    1 PB   UNK     1       0.000   0.000   0.000  1.00  0.00          Pb",
                "HETATM    2 I    UNK     1       1.000   1.000   1.500  1.00  0.00           I",
            ]
        ),
        encoding="utf-8",
    )
    assert convert_structure_file(pdb, tmp_path / "pdb_out") is not None

    shelx = tmp_path / "demo.res"
    shelx.write_text(
        "\n".join(
            [
                "TITL demo",
                "CELL 1.5418 4.0 4.0 6.0 90 90 90",
                "SFAC Pb I",
                "Pb1 1 0.0 0.0 0.0 1.0",
                "I1 2 0.5 0.5 0.25 1.0",
            ]
        ),
        encoding="utf-8",
    )
    assert convert_structure_file(shelx, tmp_path / "shelx_out") is not None


def test_hybrid3_extracts_structure_from_zip(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("notes.txt", "not a structure")
        handle.writestr("nested/POSCAR", _minimal_poscar())

    converted = convert_structure_file(archive, tmp_path / "zip_out")

    assert converted is not None
    assert converted.suffix == ".vasp"


def _minimal_poscar() -> str:
    return "\n".join(
        [
            "minimal",
            "1.0",
            "4.0 0.0 0.0",
            "0.0 4.0 0.0",
            "0.0 0.0 6.0",
            "Pb I",
            "1 1",
            "Direct",
            "0.0 0.0 0.0",
            "0.5 0.5 0.25",
            "",
        ]
    )
