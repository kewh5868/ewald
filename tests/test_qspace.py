"""pyFAI q-space processing wrappers."""

from math import pi, radians, sin
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import ewald.processing.qspace as qspace
from ewald.processing.low_q import (
    build_low_q_features,
    critical_angle_deg_from_delta,
    estimate_bright_spot_centroid,
    estimate_refractive_index_delta,
    estimate_refractive_index_from_structure,
)
from ewald.processing.qspace import (
    CakingConfig,
    GrazingIncidenceConfig,
    cake_image,
    map_grazing_incidence_qspace,
    xray_energy_kev_from_wavelength_m,
    xray_energy_kev_to_wavelength_m,
)


def test_cake_image_returns_xarray_q_chi_map(repo_root):
    image = tifffile.imread(next((repo_root / "example").glob("*.tiff")))
    caked = cake_image(
        image,
        repo_root / "example" / "calib.poni",
        config=CakingConfig(npt_radial=24, npt_azimuthal=18),
    )

    assert caked.dims == ("azimuth", "radial")
    assert caked.shape == (18, 24)
    assert caked.attrs["radial_unit"] == "q_nm^-1"
    assert caked.attrs["azimuthal_unit"] == "chi_deg"


def test_grazing_incidence_map_returns_xarray_qip_qoop_map(repo_root):
    image = tifffile.imread(next((repo_root / "example").glob("*.tiff")))
    qmap = map_grazing_incidence_qspace(
        image,
        repo_root / "example" / "calib.poni",
        config=GrazingIncidenceConfig(
            npt_ip=20,
            npt_oop=16,
            xray_energy_kev=12.7,
            incident_angle_deg=0.3,
            sample_orientation=6,
            correct_solid_angle=False,
            polarization_factor=0.95,
            normalization_factor=2.0,
        ),
    )

    assert qmap.dims == ("q_oop", "q_ip")
    assert qmap.shape == (16, 20)
    assert qmap.attrs["q_ip_unit"] == "qip_A^-1"
    assert qmap.attrs["q_oop_unit"] == "qoop_A^-1"
    assert qmap.attrs["xray_energy_kev"] == 12.7
    assert qmap.attrs["correct_solid_angle"] is False
    assert qmap.attrs["polarization_factor"] == 0.95
    assert qmap.attrs["normalization_factor"] == 2.0
    assert qmap.attrs["incident_angle_deg"] == 0.3


def test_grazing_incidence_map_sets_wavelength_property(monkeypatch):
    class FakeFiberIntegrator:
        def __init__(self):
            self.wavelength_values = []

        @property
        def wavelength(self):
            if not self.wavelength_values:
                return None
            return self.wavelength_values[-1]

        @wavelength.setter
        def wavelength(self, value):
            self.wavelength_values.append(value)

        def set_wavelength(self, value):
            raise AssertionError("deprecated set_wavelength should not run")

        def integrate2d_grazing_incidence(self, *args, **kwargs):
            return SimpleNamespace(
                intensity=np.ones((2, 3), dtype=float),
                outofplane=np.array([0.0, 1.0]),
                inplane=np.array([0.0, 1.0, 2.0]),
                ip_unit="qip_A^-1",
                oop_unit="qoop_A^-1",
            )

    fake = FakeFiberIntegrator()
    monkeypatch.setattr(qspace, "load_fiber_integrator", lambda path: fake)

    qmap = map_grazing_incidence_qspace(
        np.ones((2, 3), dtype=float),
        "calib.poni",
        config=GrazingIncidenceConfig(xray_energy_kev=12.398419843320026),
    )

    assert fake.wavelength_values == pytest.approx([1.0e-10])
    assert qmap.attrs["wavelength_m"] == pytest.approx(1.0e-10)


def test_xray_energy_wavelength_conversions_round_trip():
    wavelength = xray_energy_kev_to_wavelength_m(12.398419843320026)

    assert wavelength == pytest.approx(1.0e-10)
    assert xray_energy_kev_from_wavelength_m(wavelength) == pytest.approx(
        12.398419843320026
    )


def test_low_q_feature_helpers_build_gi_markers():
    critical_angle = critical_angle_deg_from_delta(1.0e-6)

    features = build_low_q_features(
        incident_angle_deg=0.3,
        critical_angle_deg=critical_angle,
        xray_energy_kev=12.398419843320026,
        reflected_beam_x_px=12.0,
        reflected_beam_y_px=34.0,
    )
    by_kind = {feature.kind: feature for feature in features}

    assert by_kind["direct_beam"].qz == pytest.approx(0.0)
    assert by_kind["reflected_beam"].qz == pytest.approx(
        4.0 * pi * sin(radians(0.3))
    )
    assert by_kind["reflected_beam"].x_px == 12.0
    assert by_kind["yoneda_band"].display == "horizontal_line"
    assert "alpha_f = alpha_c" in by_kind["yoneda_band"].label
    assert by_kind["critical_q"].metadata["role"] == "film critical-angle edge"
    assert by_kind["critical_q"].qz > 0


def test_refractive_delta_estimate_uses_formula_density_and_energy():
    estimate = estimate_refractive_index_delta(
        "CH3NH3PbI3",
        density_g_cm3=4.16,
        xray_energy_kev=12.398419843320026,
    )

    assert estimate.normalized_formula == "H6PbCI3N"
    assert estimate.electrons_per_formula_unit == pytest.approx(260.0)
    assert estimate.electron_density_per_a3 > 0
    assert estimate.delta > 0
    assert estimate.critical_angle_deg == pytest.approx(
        critical_angle_deg_from_delta(estimate.delta)
    )


def test_refractive_delta_estimate_accepts_sample_alias_formula():
    estimate = estimate_refractive_index_delta(
        "1MAI1PbI2",
        density_g_cm3=4.16,
        xray_energy_kev=12.398419843320026,
    )

    assert estimate.normalized_formula == "H6PbCI3N"


def test_structure_refractive_index_estimate_loads_cif(tmp_path):
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    cif_path = tmp_path / "silicon.cif"
    cif_path.write_text(structure.to(fmt="cif"), encoding="utf-8")

    estimate = estimate_refractive_index_from_structure(
        cif_path,
        xray_energy_kev=12.398419843320026,
    )

    assert estimate.file_format == "CIF"
    assert estimate.normalized_formula == "Si"
    assert estimate.composition["Si"] == pytest.approx(2.0)
    assert estimate.atom_count == 2
    assert estimate.density_g_cm3 > 0
    assert estimate.delta > 0
    assert estimate.refractive_index_real == pytest.approx(
        1.0 - estimate.delta
    )


def test_structure_refractive_index_estimate_loads_poscar(tmp_path):
    poscar_path = _write_si_poscar(tmp_path / "POSCAR_Si")

    estimate = estimate_refractive_index_from_structure(
        poscar_path,
        xray_energy_kev=12.398419843320026,
    )

    assert estimate.file_format == "POSCAR"
    assert estimate.normalized_formula == "Si"
    assert estimate.composition["Si"] == pytest.approx(2.0)
    assert estimate.unit_cell_volume_a3 == pytest.approx(5.43**3)
    assert estimate.density_g_cm3 > 0
    assert estimate.critical_angle_deg > 0


def test_structure_refractive_index_estimate_loads_vasp_file(tmp_path):
    vasp_path = _write_si_poscar(tmp_path / "si.vasp")

    estimate = estimate_refractive_index_from_structure(
        vasp_path,
        xray_energy_kev=12.398419843320026,
    )

    assert estimate.file_format == "VASP"
    assert estimate.normalized_formula == "Si"
    assert estimate.delta > 0


def test_structure_refractive_index_estimate_rejects_bad_files(tmp_path):
    unsupported = tmp_path / "structure.txt"
    unsupported.write_text("not a structure", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported structure file type"):
        estimate_refractive_index_from_structure(
            unsupported,
            xray_energy_kev=12.398419843320026,
        )

    malformed = tmp_path / "POSCAR_broken"
    malformed.write_text("not a poscar", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not parse POSCAR"):
        estimate_refractive_index_from_structure(
            malformed,
            xray_energy_kev=12.398419843320026,
        )


def test_bright_spot_centroid_uses_high_intensity_pixels():
    image = [[0.0, 0.0, 0.0], [0.0, 20.0, 40.0], [0.0, 0.0, 0.0]]

    centroid = estimate_bright_spot_centroid(image, percentile=80.0)

    assert centroid is not None
    assert centroid[0] == pytest.approx(1.6667, abs=1.0e-4)
    assert centroid[1] == pytest.approx(1.0)


def test_bright_spot_centroid_respects_image_orientation():
    image = [[0.0, 10.0, 30.0], [0.0, 0.0, 0.0]]

    centroid = estimate_bright_spot_centroid(
        image,
        percentile=80.0,
        rotation_deg=90,
        mirrored_y=True,
    )

    assert centroid is not None
    assert centroid[0] == pytest.approx(0.0)
    assert centroid[1] == pytest.approx(1.75)


def _write_si_poscar(path):
    path.write_text(
        "\n".join(
            [
                "Si",
                "1.0",
                "5.43 0.0 0.0",
                "0.0 5.43 0.0",
                "0.0 0.0 5.43",
                "Si",
                "2",
                "Direct",
                "0.0 0.0 0.0",
                "0.25 0.25 0.25",
            ]
        ),
        encoding="utf-8",
    )
    return path
