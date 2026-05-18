"""Filename and folder metadata inference tests."""

from ewald.io.importers import build_data_group_from_paths
from ewald.io.metadata import infer_filename_metadata, infer_folder_metadata

EXAMPLE_NAME = (
    "sam22_1MAI1PbI2_unfilt_0p3M_5p0scfh_Si_30uL_043_2068.2s_"
    "x0.015_th0.300_0.49s_986546_001639_maxs.tiff"
)


def test_example_filename_metadata_inference(repo_root):
    result = infer_filename_metadata(repo_root / "example" / EXAMPLE_NAME)
    metadata = result.as_metadata()

    assert metadata["sample_number"] == 22
    assert metadata["sample_composition"] == "1MAI1PbI2"
    assert metadata["filtration_status"] == "unfiltered"
    assert metadata["concentration_molar"] == 0.3
    assert metadata["flow_rate_scfh"] == 5.0
    assert metadata["substrate"] == "Si"
    assert metadata["solution_volume_uL"] == 30.0
    assert metadata["x_position"] == 0.015
    assert metadata["incidence_angle_deg"] == 0.3
    assert metadata["frame_timestamp_s"] == 2068.2
    assert metadata["exposure_time_s"] == 0.49
    assert metadata["duration_candidates_s"] == [2068.2, 0.49]
    assert metadata["detector_type"] == "maxs"
    assert metadata["frame_number"] == 1639
    assert metadata["run_id"] == 986546
    assert "043" in metadata["_unresolved_tokens"]


def test_folder_report_flags_inconsistent_token_counts(tmp_path):
    paths = [
        tmp_path / EXAMPLE_NAME,
        tmp_path / ("extra_" + EXAMPLE_NAME),
    ]
    for path in paths:
        path.touch()

    report = infer_folder_metadata(paths)

    assert report.file_count == 2
    assert report.consistent_token_count is False
    assert len(report.files_requiring_metadata_input) == 2


def test_data_group_from_paths_stores_parse_report(repo_root):
    path = repo_root / "example" / EXAMPLE_NAME

    group, report = build_data_group_from_paths([path], group_name="Example")

    assert group.name == "Example"
    assert group.data_files[0].metadata["incidence_angle_deg"] == 0.3
    assert group.data_files[0].metadata["original_file_name"] == EXAMPLE_NAME
    assert group.parse_report["consistent_token_count"] is True
    assert report.recurrent_exposure_time_s == 0.49


def test_data_group_merges_sidecar_yaml_metadata(repo_root, tmp_path):
    path = repo_root / "example" / EXAMPLE_NAME
    sidecar = tmp_path / "metadata.yml"
    sidecar.write_text(
        """
defaults:
  beamline: testline
files:
  sam22_1MAI1PbI2_unfilt_0p3M_5p0scfh_Si_30uL_043_2068.2s_x0.015_th0.300_0.49s_986546_001639_maxs:
    operator: Keith
    incidence_angle_deg: 0.31
""".strip(),
        encoding="utf-8",
    )

    group, _ = build_data_group_from_paths(
        [path], group_name="Example", metadata_yml=sidecar
    )
    metadata = group.data_files[0].metadata

    assert metadata["beamline"] == "testline"
    assert metadata["operator"] == "Keith"
    assert metadata["incidence_angle_deg"] == 0.31
    assert metadata["_metadata_sidecar"] == str(sidecar)
